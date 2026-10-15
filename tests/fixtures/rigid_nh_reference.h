#pragma once
#include "integrate/rigid_nh.cuh"
#include <cmath>
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
#include "rigid_nh_lammps_functions.h"
}

