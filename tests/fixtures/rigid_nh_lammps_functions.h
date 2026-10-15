// Generated test oracle; DO NOT use in production.
// LAMMPS commit: c71ef819ab3785bba81c31c0ef5fce84603608d3
// fix_rigid_nh_small.cpp SHA256: de8d0f6119d3124bc3ef86264f684e723dd98fc7a0250d29c851c95f71214aa4
// fix_rigid_nh_small.h SHA256: 2d81e06561490a4203268d2192c7c4f72cb2254fa929bb8904efd55bceea9969
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Trung Dac Nguyen (ORNL)
   references: Kamberaj et al., J. Chem. Phys. 122, 224114 (2005)
               Miller et al., J Chem Phys. 116, 8649-8659 (2002)
------------------------------------------------------------------------- */

inline void Reference::nhc_temp_integrate()
{
  if (g_f == 0) return;

  int i,j,k;
  double kt,gfkt_t,gfkt_r,tmp,ms,s,s2;

  kt = boltz * t_target;
  gfkt_t = nf_t * kt;
  gfkt_r = nf_r * kt;

  // update thermostat masses

  double t_mass = boltz * t_target / (t_freq * t_freq);
  q_t[0] = nf_t * t_mass;
  q_r[0] = nf_r * t_mass;
  for (i = 1; i < t_chain; i++)
    q_t[i] = q_r[i] = t_mass;

  // update force of thermostats coupled to particles

  f_eta_t[0] = (akin_t * mvv2e - gfkt_t) / q_t[0];
  f_eta_r[0] = (akin_r * mvv2e - gfkt_r) / q_r[0];

  // multiple timestep iteration

  for (i = 0; i < t_iter; i++) {
    for (j = 0; j < t_order; j++) {

      // update thermostat velocities half step

      eta_dot_t[t_chain-1] += wdti2[j] * f_eta_t[t_chain-1];
      eta_dot_r[t_chain-1] += wdti2[j] * f_eta_r[t_chain-1];

      for (k = 1; k < t_chain; k++) {
        tmp = wdti4[j] * eta_dot_t[t_chain-k];
        ms = maclaurin_series(tmp);
        s = exp(-1.0 * tmp);
        s2 = s * s;
        eta_dot_t[t_chain-k-1] = eta_dot_t[t_chain-k-1] * s2 +
          wdti2[j] * f_eta_t[t_chain-k-1] * s * ms;

        tmp = wdti4[j] * eta_dot_r[t_chain-k];
        ms = maclaurin_series(tmp);
        s = exp(-1.0 * tmp);
        s2 = s * s;
        eta_dot_r[t_chain-k-1] = eta_dot_r[t_chain-k-1] * s2 +
          wdti2[j] * f_eta_r[t_chain-k-1] * s * ms;
      }

      // update thermostat positions a full step

      for (k = 0; k < t_chain; k++) {
        eta_t[k] += wdti1[j] * eta_dot_t[k];
        eta_r[k] += wdti1[j] * eta_dot_r[k];
      }

      // update thermostat forces

      for (k = 1; k < t_chain; k++) {
        f_eta_t[k] = q_t[k-1] * eta_dot_t[k-1] * eta_dot_t[k-1] - kt;
        f_eta_t[k] /= q_t[k];
        f_eta_r[k] = q_r[k-1] * eta_dot_r[k-1] * eta_dot_r[k-1] - kt;
        f_eta_r[k] /= q_r[k];
      }

      // update thermostat velocities a full step

      for (k = 0; k < t_chain-1; k++) {
        tmp = wdti4[j] * eta_dot_t[k+1];
        ms = maclaurin_series(tmp);
        s = exp(-1.0 * tmp);
        s2 = s * s;
        eta_dot_t[k] = eta_dot_t[k] * s2 + wdti2[j] * f_eta_t[k] * s * ms;
        tmp = q_t[k] * eta_dot_t[k] * eta_dot_t[k] - kt;
        f_eta_t[k+1] = tmp / q_t[k+1];

        tmp = wdti4[j] * eta_dot_r[k+1];
        ms = maclaurin_series(tmp);
        s = exp(-1.0 * tmp);
        s2 = s * s;
        eta_dot_r[k] = eta_dot_r[k] * s2 + wdti2[j] * f_eta_r[k] * s * ms;
        tmp = q_r[k] * eta_dot_r[k] * eta_dot_r[k] - kt;
        f_eta_r[k+1] = tmp / q_r[k+1];
      }

      eta_dot_t[t_chain-1] += wdti2[j] * f_eta_t[t_chain-1];
      eta_dot_r[t_chain-1] += wdti2[j] * f_eta_r[t_chain-1];
    }
  }
}
inline void Reference::nhc_press_integrate()
{
  int i,j,k;
  double tmp,s,s2,ms,kecurrent;
  double kt = boltz * t_target;
  double lkt_press = kt;

  // update thermostat masses

  int dimension = domain->dimension;
  double tb_mass = kt / (p_freq_max * p_freq_max);
  q_b[0] = dimension * dimension * tb_mass;
  for (i = 1; i < p_chain; i++) {
    q_b[i] = tb_mass;
    f_eta_b[i] = q_b[i-1] * eta_dot_b[i-1] * eta_dot_b[i-1] - kt;
    f_eta_b[i] /= q_b[i];
  }

  // update forces acting on thermostat

  kecurrent = 0.0;
  for (i = 0; i < 3; i++)
    if (p_flag[i]) {
      epsilon_mass[i] = (g_f + dimension) * kt / (p_freq[i] * p_freq[i]);
      kecurrent += epsilon_mass[i] * epsilon_dot[i] * epsilon_dot[i];
    }
  kecurrent /= pdim;

  f_eta_b[0] = (kecurrent - lkt_press) / q_b[0];

  // multiple timestep iteration

  for (i = 0; i < t_iter; i++) {
    for (j = 0; j < t_order; j++) {

      // update thermostat velocities a half step

      eta_dot_b[p_chain-1] += wdti2[j] * f_eta_b[p_chain-1];

      for (k = 1; k < p_chain; k++) {
        tmp = wdti4[j] * eta_dot_b[p_chain-k];
        ms = maclaurin_series(tmp);
        s = exp(-0.5 * tmp);
        s2 = s * s;
        eta_dot_b[p_chain-k-1] = eta_dot_b[p_chain-k-1] * s2 +
          wdti2[j] * f_eta_b[p_chain-k-1] * s * ms;
      }

      // update thermostat positions

      for (k = 0; k < p_chain; k++)
        eta_b[k] += wdti1[j] * eta_dot_b[k];

      // update thermostat forces

      for (k = 1; k < p_chain; k++) {
        f_eta_b[k] = q_b[k-1] * eta_dot_b[k-1] * eta_dot_b[k-1] - kt;
        f_eta_b[k] /= q_b[k];
      }

      // update thermostat velocites a full step

      for (k = 0; k < p_chain-1; k++) {
        tmp = wdti4[j] * eta_dot_b[k+1];
        ms = maclaurin_series(tmp);
        s = exp(-0.5 * tmp);
        s2 = s * s;
        eta_dot_b[k] = eta_dot_b[k] * s2 + wdti2[j] * f_eta_b[k] * s * ms;
        tmp = q_b[k] * eta_dot_b[k] * eta_dot_b[k] - kt;
        f_eta_b[k+1] = tmp / q_b[k+1];
      }

      eta_dot_b[p_chain-1] += wdti2[j] * f_eta_b[p_chain-1];
    }
  }
}
inline void Reference::nh_epsilon_dot()
{
  if (g_f == 0) return;

  int i;
  double volume,scale,f_epsilon;

  if (domain->dimension == 2) volume = domain->xprd*domain->yprd;
  else volume = domain->xprd*domain->yprd*domain->zprd;

  // MTK terms

  mtk_term1 = (akin_t + akin_r) * mvv2e / g_f;

  scale = exp(-1.0 * dtq * eta_dot_b[0]);

  for (i = 0; i < 3; i++)
    if (p_flag[i]) {
      f_epsilon = (p_current[i]-p_hydro)*volume / nktv2p + mtk_term1;
      f_epsilon /= epsilon_mass[i];
      epsilon_dot[i] += dtq * f_epsilon;
      epsilon_dot[i] *= scale;
    }

  mtk_term2 = 0.0;
  for (i = 0; i < 3; i++)
    if (p_flag[i]) mtk_term2 += epsilon_dot[i];
  mtk_term2 /= g_f;
}
inline double Reference::maclaurin_series(double x)
{
  double x2, x4;
  x2 = x * x;
  x4 = x2 * x2;
  return (1.0 + (1.0 / 6.0) * x2 + (1.0 / 120.0) * x4 + (1.0 / 5040.0) * x2 * x4 +
          (1.0 / 362880.0) * x4 * x4);
}
