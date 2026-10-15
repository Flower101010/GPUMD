#include "integrate/rigid_nh.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

// Minimal host stand-in for fields accessed by the original LAMMPS functions.
// The actual propagation bodies below are extracted, not reimplemented here.
namespace lammps_reference {
struct Reference {
  struct Domain { int dimension = 3; double xprd = 1, yprd = 1, zprd = 1; } cell;
  Domain* domain = &cell;
  int nf_t, nf_r, g_f, t_chain, p_chain, t_iter, t_order, pdim = 3;
  double boltz = 1, mvv2e = 1, nktv2p = 1, t_target, t_freq, p_freq_max;
  double akin_t = 0, akin_r = 0, dtq = 0, mtk_term1 = 0, mtk_term2 = 0, p_hydro;
  int p_flag[3] = {1, 1, 1};
  double p_freq[3], p_current[3]{}, epsilon_mass[3], epsilon_dot[3]{};
  std::vector<double> q_t, q_r, eta_t, eta_r, eta_dot_t, eta_dot_r, f_eta_t, f_eta_r;
  std::vector<double> q_b, eta_b, eta_dot_b, f_eta_b, weights, wdti1, wdti2, wdti4;
  explicit Reference(const RigidNH::Parameters& p)
    : nf_t(p.translation_dof), nf_r(p.rotation_dof), g_f(nf_t + nf_r),
      t_chain(p.chain_length), p_chain(p.piston_chain_length), t_iter(p.subcycles),
      t_order(p.yoshida_order), t_target(p.kT), t_freq(1 / p.temperature_time),
      p_freq_max(1 / p.pressure_time), p_hydro(p.target_pressure),
      q_t(t_chain), q_r(t_chain), eta_t(t_chain), eta_r(t_chain),
      eta_dot_t(t_chain), eta_dot_r(t_chain), f_eta_t(t_chain), f_eta_r(t_chain),
      q_b(p_chain), eta_b(p_chain), eta_dot_b(p_chain), f_eta_b(p_chain),
      wdti1(t_order), wdti2(t_order), wdti4(t_order)
  {
    const double tm = t_target / (t_freq * t_freq);
    q_t[0] = nf_t * tm;
    q_r[0] = nf_r * tm;
    for (int i = 1; i < t_chain; ++i) {
      q_t[i] = q_r[i] = tm;
      f_eta_t[i] = f_eta_r[i] = -t_target / tm;
    }
    const double bm = t_target / (p_freq_max * p_freq_max);
    q_b[0] = 9 * bm;
    for (int i = 1; i < p_chain; ++i) { q_b[i] = bm; f_eta_b[i] = -t_target / bm; }
    for (int i = 0; i < 3; ++i) {
      p_freq[i] = p_freq_max;
      epsilon_mass[i] = (g_f + 3) * t_target / (p_freq[i] * p_freq[i]);
    }
    if (t_order == 3) {
      const double w = 1 / (2 - std::pow(2.0, 1.0 / 3));
      weights = {w, 1 - 2 * w, w};
    } else {
      const double w = 1 / (4 - std::pow(4.0, 1.0 / 3));
      weights = {w, w, 1 - 4 * w, w, w};
    }
  }
  void set_duration(double duration)
  {
    dtq = duration / 2;
    for (int j = 0; j < t_order; ++j) {
      wdti1[j] = weights[j] * duration / t_iter;
      wdti2[j] = wdti1[j] / 2;
      wdti4[j] = wdti1[j] / 4;
    }
  }
  void nhc_temp_integrate();
  void nhc_press_integrate();
  void nh_epsilon_dot();
  double maclaurin_series(double x);
};
#include "fixtures/rigid_nh_lammps_functions.h"
}

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
  analytical_and_boundaries();
  std::cout << "Rigid NH chains, piston and MTK checks passed.\n";
}
