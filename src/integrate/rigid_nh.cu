/*
   Adapted from LAMMPS src/RIGID/fix_rigid_nh_small.cpp (GNU GPL).
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ ; Sandia National Laboratories.
   Copyright (2003) Sandia Corporation. Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software. Contributing author: Trung Dac Nguyen (ORNL).
   References: Kamberaj et al., J. Chem. Phys. 122, 224114 (2005);
   Miller et al., J. Chem. Phys. 116, 8649-8659 (2002).
   See developers/rigid_nh.md for source correspondence and validation scope.
*/
#include "rigid_nh.cuh"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message)
{
  if (!condition) throw std::invalid_argument(message);
}
bool positive(double x) { return std::isfinite(x) && x > 0; }
void check_duration(double duration)
{
  require(std::isfinite(duration), "Rigid NH requires a finite propagation duration.");
}
// 保留 LAMMPS 的八阶 Maclaurin 近似，以便逐子步对照。
double maclaurin(double x)
{
  const double x2 = x * x, x4 = x2 * x2;
  return 1 + x2 / 6 + x4 / 120 + x2 * x4 / 5040 + x4 * x4 / 362880;
}
RigidThermostat::Chain make_chain(int length, double first_mass, double other_mass, double kT)
{
  require(positive(first_mass) && positive(other_mass),
    "Rigid NH masses are outside the finite positive range.");
  RigidThermostat::Chain chain(length);
  for (int i = 0; i < length; ++i) {
    chain[i].mass = i == 0 ? first_mass : other_mass;
    if (i > 0) chain[i].acceleration = -kT / other_mass;
  }
  return chain;
}
void require_finite(const RigidThermostat::Chain& chain)
{
  for (const auto& link : chain)
    if (!std::isfinite(link.rate) || !std::isfinite(link.coordinate) ||
        !std::isfinite(link.acceleration))
      throw std::overflow_error("Rigid NH chain overflow; reduce time step or check the state.");
}

std::vector<double> yoshida_weights(int order)
{
  if (order == 3) {
    const double w = 1 / (2 - std::pow(2.0, 1.0 / 3));
    return {w, 1 - 2 * w, w};
  }
  const double w = 1 / (4 - std::pow(4.0, 1.0 / 3));
  return {w, w, 1 - 4 * w, w, w};
}
void set_thermostat_masses(RigidThermostat::Chain& chain, int dof, double kT, double tau)
{
  const double mass = kT * tau * tau;
  require(positive(mass) && positive(dof * mass),
    "Rigid thermostat masses are outside the finite positive range.");
  for (std::size_t i = 0; i < chain.size(); ++i) chain[i].mass = i == 0 ? dof * mass : mass;
}
void check_kinetic_energy(double kt, double kr, int rotation_dof)
{
  require(std::isfinite(kt) && kt >= 0 && std::isfinite(kr) && kr >= 0 &&
      (rotation_dof != 0 || kr == 0), "Invalid rigid NH twice-kinetic energies.");
}
void advance_chain(
  RigidThermostat::Chain& chain, double duration, double kinetic, double target_dof,
  double damping, double kT, int subcycles, const std::vector<double>& weights)
{
  chain[0].acceleration = (kinetic - target_dof * kT) / chain[0].mass;
  for (std::size_t i = 1; i < chain.size(); ++i) {
    const auto& previous = chain[i - 1];
    chain[i].acceleration = (previous.mass * previous.rate * previous.rate - kT) / chain[i].mass;
  }
  for (int subcycle = 0; subcycle < subcycles; ++subcycle) {
    for (double weight : weights) {
      const double step = weight * duration / subcycles;
      const double half = 0.5 * step, quarter = 0.25 * step;
      auto update_rate = [&](std::size_t i) {
        const double x = quarter * chain[i + 1].rate;
        const double s = std::exp(-damping * x);
        chain[i].rate = chain[i].rate * s * s + half * chain[i].acceleration * s * maclaurin(x);
      };
      chain.back().rate += half * chain.back().acceleration;
      for (std::size_t i = chain.size() - 1; i > 0; --i) update_rate(i - 1);
      for (auto& link : chain) link.coordinate += step * link.rate;
      // LAMMPS 在这里先刷新所有高节驱动力；前向 sweep 每更新一节后又刷新
      // 下一节。直接在下方逐节刷新等价，避免重复计算和保存临时数组。
      for (std::size_t i = 0; i + 1 < chain.size(); ++i) {
        update_rate(i);
        chain[i + 1].acceleration =
          (chain[i].mass * chain[i].rate * chain[i].rate - kT) / chain[i + 1].mass;
      }
      chain.back().rate += half * chain.back().acceleration;
    }
  }
  require_finite(chain);
}

} // namespace

RigidThermostat::RigidThermostat(const Parameters& p) : parameters_(p)
{
  require(p.translation_dof > 0 && p.rotation_dof >= 0 &&
      static_cast<long long>(p.translation_dof) + p.rotation_dof <= std::numeric_limits<int>::max(),
    "Invalid rigid NH degrees of freedom.");
  require(positive(p.kT) && positive(p.temperature_time), "Invalid rigid thermostat target or coupling time.");
  require(p.chain_length >= 1 && p.subcycles >= 1 && (p.yoshida_order == 3 || p.yoshida_order == 5),
    "Invalid rigid NH chain integration settings.");
  const double mass = p.kT * p.temperature_time * p.temperature_time;
  translation_ = make_chain(p.chain_length, p.translation_dof * mass, mass, p.kT);
  if (p.rotation_dof) rotation_ = make_chain(p.chain_length, p.rotation_dof * mass, mass, p.kT);
  require_finite(translation_);
  require_finite(rotation_);
  weights_ = yoshida_weights(p.yoshida_order);
}

void RigidThermostat::advance(double duration, double kt, double kr)
{
  advance(duration, kt, kr, parameters_.kT);
}
void RigidThermostat::advance(double duration, double kt, double kr, double kT)
{
  check_duration(duration);
  check_kinetic_energy(kt, kr, parameters_.rotation_dof);
  require(positive(kT), "Rigid thermostat requires finite positive kT.");
  if (duration == 0) return;
  auto translation = translation_, rotation = rotation_;
  set_thermostat_masses(translation, parameters_.translation_dof, kT, parameters_.temperature_time);
  if (!rotation.empty()) set_thermostat_masses(rotation, parameters_.rotation_dof, kT, parameters_.temperature_time);
  advance_chain(translation, duration, kt, parameters_.translation_dof, 1, kT, parameters_.subcycles, weights_);
  if (!rotation.empty())
    advance_chain(rotation, duration, kr, parameters_.rotation_dof, 1, kT, parameters_.subcycles, weights_);
  translation_.swap(translation);
  rotation_.swap(rotation);
  parameters_.kT = kT;
}
RigidThermostat::MomentumScales RigidThermostat::momentum_scales(double duration) const
{
  check_duration(duration);
  const MomentumScales scales{std::exp(-duration * translation_[0].rate),
    rotation_.empty() ? 1 : std::exp(-duration * rotation_[0].rate)};
  if (!positive(scales.translation) || !positive(scales.rotation))
    throw std::overflow_error("Rigid thermostat momentum scales are outside the finite positive range.");
  return scales;
}

RigidNH::RigidNH(const Parameters& p)
  : parameters_(p), thermostat_({p.translation_dof, p.rotation_dof, p.kT, p.temperature_time,
      p.chain_length, p.subcycles, p.yoshida_order})
{
  require(positive(p.pressure_time) && std::isfinite(p.target_pressure), "Invalid rigid NH target or coupling time.");
  require(p.piston_chain_length >= 1, "Invalid rigid NH piston chain length.");
  const double qb = p.kT * p.pressure_time * p.pressure_time;
  piston_.mass = (degrees_of_freedom() + 3) * qb;
  require(positive(qb) && positive(9 * qb) && positive(piston_.mass),
    "Rigid NH masses are outside the finite positive range.");
  piston_chain_ = make_chain(p.piston_chain_length, 9 * qb, qb, p.kT);
  require_finite(piston_chain_);
  weights_ = yoshida_weights(p.yoshida_order);
}

double RigidNH::degrees_of_freedom() const
{
  return static_cast<double>(parameters_.translation_dof) + parameters_.rotation_dof;
}
void RigidNH::check_kinetic(double kt, double kr) const
{
  check_kinetic_energy(kt, kr, parameters_.rotation_dof);
}

void RigidNH::advance_thermostats(double duration, double kt, double kr)
{
  thermostat_.advance(duration, kt, kr);
}
void RigidNH::advance_piston_thermostat(double duration)
{
  check_duration(duration);
  if (duration == 0) return;
  auto chain = piston_chain_;
  // LAMMPS: sum_i(M_i*nu_i²)/pdim = M*nu²；第一节目标是 kT，不是 9*kT。
  advance_chain(chain, duration, piston_.mass * piston_.rate * piston_.rate, 1.0, 0.5,
    parameters_.kT, parameters_.subcycles, weights_);
  piston_chain_.swap(chain);
}
RigidNH::MTKTerms RigidNH::mtk_terms(double kt, double kr) const
{
  check_kinetic(kt, kr);
  // 先相除避免两个有限动能求和溢出。
  const double g = degrees_of_freedom();
  const MTKTerms terms{kt / g + kr / g, (3 / g) * piston_.rate};
  if (!std::isfinite(terms.energy) || !std::isfinite(terms.rate))
    throw std::overflow_error("Rigid NH MTK terms overflow.");
  return terms;
}
void RigidNH::advance_piston(double duration, double pressure, double volume, double kt, double kr)
{
  check_duration(duration);
  require(std::isfinite(pressure) && positive(volume), "Invalid rigid NH pressure or volume.");
  const auto terms = mtk_terms(kt, kr);
  if (duration == 0) return;
  const double acceleration =
    ((pressure - parameters_.target_pressure) * volume + terms.energy) / piston_.mass;
  const double next = (piston_.rate + duration * acceleration) *
    std::exp(-duration * piston_chain_[0].rate);
  if (!std::isfinite(next)) throw std::overflow_error("Rigid NH piston rate overflow.");
  piston_.rate = next;
}
RigidNH::MomentumScales RigidNH::momentum_scales(double duration) const
{
  check_duration(duration);
  const double correction = (3 / degrees_of_freedom()) * piston_.rate;
  const MomentumScales scales{
    std::exp(-duration * (translation_chain()[0].rate + piston_.rate + correction)),
    rotation_chain().empty() ? 1.0 : std::exp(-duration * (rotation_chain()[0].rate + 3 * correction))};
  if (!positive(scales.translation) || !positive(scales.rotation))
    throw std::overflow_error("Rigid NH momentum scales are outside the finite positive range.");
  return scales;
}
