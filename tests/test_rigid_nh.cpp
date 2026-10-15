#include "integrate/rigid_nh.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "fixtures/rigid_nh_reference.h"

namespace {
double max_error = 0;
void near(double a, double b, double tolerance = 3e-11)
{
  assert(std::isfinite(a) && std::isfinite(b));
  max_error = std::max(max_error, std::abs(a - b));
  if (std::abs(a - b) > tolerance * (1 + std::abs(b)))
    std::cerr << "Mismatch: " << a << " versus " << b << '\n';
  assert(std::abs(a - b) <= tolerance * (1 + std::abs(b)));
}
void compare_chain(const RigidNH::Chain& chain, const std::vector<double>& q,
  const std::vector<double>& eta, const std::vector<double>& rate, const std::vector<double>& acceleration)
{
  assert(chain.size() == q.size());
  for (std::size_t i = 0; i < chain.size(); ++i) {
    near(chain[i].mass, q[i]);
    near(chain[i].coordinate, eta[i]);
    near(chain[i].rate, rate[i]);
    near(chain[i].acceleration, acceleration[i]);
  }
}
std::vector<double> snapshot(const RigidNH& nh)
{
  std::vector<double> values{nh.piston().rate, nh.piston().mass};
  for (const auto* chain : {&nh.translation_chain(), &nh.rotation_chain(), &nh.piston_chain()})
    for (const auto& link : *chain)
      values.insert(values.end(), {link.coordinate, link.rate, link.mass, link.acceleration});
  return values;
}
template <class Exception, class Function> void rejects(Function f)
{
  bool caught = false;
  try { f(); } catch (const Exception&) { caught = true; }
  assert(caught);
}
void source_parity()
{
  std::mt19937 rng(20261015);
  std::uniform_real_distribution<double> random(0, 1);
  int cases = 0;
  for (int order : {3, 5}) for (int cycles : {1, 3})
    for (int length : {1, 3, 10}) for (int piston_length : {1, 4, 10}) {
      RigidNH::Parameters p{9, 6, 0.025, 0.2, 0.8, 0.001, length, piston_length, cycles, order};
      RigidNH nh(p);
      lammps_reference::Reference ref(p);
      for (int step = 0; step < 100; ++step) {
        const double dt = (0.0005 + 0.001 * random(rng)) * (step >= 90 ? -1 : 1);
        const double kt = p.kT * p.translation_dof * (0.5 + random(rng));
        const double kr = p.kT * p.rotation_dof * (0.5 + random(rng));
        const double pressure = 0.003 * random(rng), volume = 70 + 60 * random(rng);
        ref.set_duration(dt);
        ref.akin_t = kt; ref.akin_r = kr;
        nh.advance_thermostats(dt, kt, kr);
        ref.nhc_temp_integrate();
        compare_chain(nh.translation_chain(), ref.q_t, ref.eta_t, ref.eta_dot_t, ref.f_eta_t);
        compare_chain(nh.rotation_chain(), ref.q_r, ref.eta_r, ref.eta_dot_r, ref.f_eta_r);
        nh.advance_piston_thermostat(dt);
        ref.nhc_press_integrate();
        compare_chain(nh.piston_chain(), ref.q_b, ref.eta_b, ref.eta_dot_b, ref.f_eta_b);
        nh.advance_piston(dt / 2, pressure, volume, kt, kr);
        ref.cell.xprd = volume;
        for (double& value : ref.p_current) value = pressure;
        ref.nh_epsilon_dot();
        for (int axis = 0; axis < 3; ++axis) {
          near(nh.piston().mass, ref.epsilon_mass[axis]);
          near(nh.piston().rate, ref.epsilon_dot[axis]);
        }
        const auto mtk = nh.mtk_terms(kt, kr);
        near(mtk.energy, ref.mtk_term1);
        near(mtk.rate, ref.mtk_term2);
        const auto factors = nh.momentum_scales(dt / 2);
        // Literal scale_t/scale_r expressions in initial_integrate/final_integrate.
        near(factors.translation, std::exp(-ref.dtq * ref.eta_dot_t[0]) *
          std::exp(-ref.dtq * (ref.epsilon_dot[0] + ref.mtk_term2)));
        near(factors.rotation, std::exp(-ref.dtq * ref.eta_dot_r[0]) *
          std::exp(-ref.dtq * (ref.pdim * ref.mtk_term2)));
      }
      ++cases;
    }
  std::cout << "LAMMPS scalar parity: " << cases << " cases x 100 steps; max absolute error " << max_error << '\n';
}
void thermostat_ramp_parity()
{
  int cases = 0;
  for (int order : {3, 5}) for (int cycles : {1, 3}) for (int length : {1, 3, 10}) {
    RigidThermostat thermostat({9, 6, 0.025, 0.2, length, cycles, order});
    lammps_reference::Reference ref({9, 6, 0.025, 0.2, 0.8, 0, length, 1, cycles, order});
    for (int step = 0; step < 100; ++step) {
      const double kT = 0.025*(1+0.004*step), kt = 9*kT*0.8, kr = 6*kT*1.4;
      ref.t_target = kT; ref.akin_t = kt; ref.akin_r = kr; ref.set_duration(0.001);
      ref.nhc_temp_integrate();
      thermostat.advance(0.001, kt, kr, kT);
      compare_chain(thermostat.translation_chain(), ref.q_t, ref.eta_t, ref.eta_dot_t, ref.f_eta_t);
      compare_chain(thermostat.rotation_chain(), ref.q_r, ref.eta_r, ref.eta_dot_r, ref.f_eta_r);
      near(thermostat.parameters().kT, kT);
      const auto scales = thermostat.momentum_scales(0.0005);
      near(scales.translation, std::exp(-0.0005*ref.eta_dot_t[0]));
      near(scales.rotation, std::exp(-0.0005*ref.eta_dot_r[0]));
    }
    ++cases;
  }
  std::cout << "LAMMPS thermostat ramp parity: " << cases << " cases x 100 steps\n";
  RigidThermostat thermostat({6, 3, 0.5, 2});
  auto state = [&] {
    std::vector<double> values{thermostat.parameters().kT};
    for (const auto* chain : {&thermostat.translation_chain(), &thermostat.rotation_chain()})
      for (const auto& link : *chain)
        values.insert(values.end(), {link.mass, link.coordinate, link.rate, link.acceleration});
    return values;
  };
  const auto before = state();
  rejects<std::invalid_argument>([&] { thermostat.advance(0.01, 1, 1, 1e308); });
  rejects<std::overflow_error>([&] { thermostat.advance(1e300, 1e300, 1e300, 0.6); });
  assert(state() == before);
}

void analytical_and_boundaries()
{
  RigidNH::Parameters p{6, 3, 0.5, 2, 3, 0.01, 1, 1, 1, 3};
  RigidNH nh(p);
  near(nh.translation_chain()[0].mass, 12);
  near(nh.rotation_chain()[0].mass, 6);
  near(nh.piston().mass, 54);
  near(nh.piston_chain()[0].mass, 40.5);
  const double h = 0.02;
  nh.advance_thermostats(h, 6, 0.75);
  near(nh.translation_chain()[0].rate, h * 0.25);
  near(nh.translation_chain()[0].coordinate, h * h * 0.125);
  near(nh.rotation_chain()[0].rate, -h * 0.125);
  nh.advance_thermostats(-h, 6, 0.75);
  near(nh.translation_chain()[0].rate, 0);
  near(nh.translation_chain()[0].coordinate, 0);
  near(nh.rotation_chain()[0].rate, 0);
  // Source target is one kT even with Q_b0=9*kT*tau_P².
  nh.advance_piston_thermostat(h);
  near(nh.piston_chain()[0].rate, -h * 0.5 / 40.5);
  nh.advance_piston_thermostat(-h);
  near(nh.piston_chain()[0].rate, 0);
  // Pure MTK drive survives P=P0; no hidden additional factor 3 or dt/2.
  nh.advance_piston(h, p.target_pressure, 100, 3, 1.5);
  near(nh.piston().rate, h * 0.5 / 54);
  const auto factors = nh.momentum_scales(h);
  near(factors.translation, std::exp(-h * nh.piston().rate * (1 + 3.0 / 9)));
  near(factors.rotation, std::exp(-h * nh.piston().rate)); // 9*nu/g, g=9
  // Piston update is kick then damping, not a symmetric integrator on its own.
  nh.advance_piston_thermostat(h);
  const double old_rate = nh.piston().rate;
  const double bath_rate = nh.piston_chain()[0].rate;
  nh.advance_piston(h, 0.03, 100, 3, 1.5);
  near(nh.piston().rate, (old_rate + h * (2 + 0.5) / 54) * std::exp(-h * bath_rate));

  const auto before = snapshot(nh);
  nh.advance_thermostats(0, 3, 1.5);
  nh.advance_piston_thermostat(0);
  nh.advance_piston(0, 0.03, 100, 3, 1.5);
  assert(snapshot(nh) == before);
  rejects<std::invalid_argument>([&] { nh.advance_thermostats(h, -1, 1); });
  rejects<std::invalid_argument>([&] { nh.advance_piston(h, 1, 0, 1, 1); });
  rejects<std::invalid_argument>([&] { nh.advance_piston_thermostat(std::numeric_limits<double>::infinity()); });
  rejects<std::overflow_error>([&] { nh.advance_thermostats(1e300, 1e300, 1e300); });
  rejects<std::overflow_error>([&] { nh.advance_piston(h, 1e300, 1e300, 1, 1); });
  assert(snapshot(nh) == before);

  p.rotation_dof = 0;
  RigidNH free(p);
  assert(free.rotation_chain().empty());
  free.advance_thermostats(h, 3, 0);
  free.advance_piston_thermostat(h);
  free.advance_piston(h, p.target_pressure, 100, 3, 0);
  near(free.momentum_scales(h).rotation, 1);
  rejects<std::invalid_argument>([&] { free.advance_thermostats(h, 3, 1); });
  p.chain_length = 0;
  rejects<std::invalid_argument>([&] { RigidNH invalid(p); });
  p.chain_length = 1; p.yoshida_order = 7;
  rejects<std::invalid_argument>([&] { RigidNH invalid(p); });
  p.yoshida_order = 3; p.kT = 0;
  rejects<std::invalid_argument>([&] { RigidNH invalid(p); });
  p.kT = 0.5; p.translation_dof = 0;
  rejects<std::invalid_argument>([&] { RigidNH invalid(p); });
  p.translation_dof = 6; p.pressure_time = 1e300;
  rejects<std::invalid_argument>([&] { RigidNH invalid(p); });
  p.pressure_time = 3; p.target_pressure = std::numeric_limits<double>::quiet_NaN();
  rejects<std::invalid_argument>([&] { RigidNH invalid(p); });
}
}
int main()
{
  source_parity();
  thermostat_ramp_parity();
  analytical_and_boundaries();
  std::cout << "Rigid NH chains, piston and MTK checks passed.\n";
}
