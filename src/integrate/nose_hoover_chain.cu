/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

// CPU Nose-Hoover chain propagator extracted from ensemble_nhc.cu.
// Splitting coefficients: M. E. Tuckerman, Statistical Mechanics: Theory and
// Molecular Simulation, Oxford University Press (2010).
#include "nose_hoover_chain.cuh"
#include <cmath>
#include <stdexcept>

double nose_hoover_chain::propagate(
  int length,
  double* position,
  double* momentum,
  const double* mass,
  double twice_kinetic_energy,
  double kT,
  double degrees_of_freedom,
  double duration)
{
  if (length < 2) {
    throw std::invalid_argument("Nose-Hoover chain requires at least two thermostats");
  }
  if (!position || !momentum || !mass) {
    throw std::invalid_argument("Nose-Hoover chain state must not be null");
  }
  if (!std::isfinite(twice_kinetic_energy) || twice_kinetic_energy < 0.0 ||
      !std::isfinite(kT) || kT <= 0.0 ||
      !std::isfinite(degrees_of_freedom) || degrees_of_freedom <= 0.0 ||
      !std::isfinite(duration)) {
    throw std::invalid_argument("Invalid Nose-Hoover chain propagation parameters");
  }
  for (int m = 0; m < length; ++m) {
    if (!std::isfinite(mass[m]) || mass[m] <= 0.0 ||
        !std::isfinite(position[m]) || !std::isfinite(momentum[m])) {
      throw std::invalid_argument("Invalid Nose-Hoover chain state or mass");
    }
  }

  // These constants are taken from Tuckerman's book
  constexpr int n_sy = 7;
  constexpr int n_respa = 4;
  const double w[7] = {
    0.784513610477560,
    0.235573213359357,
    -1.17767998417887,
    1.31518632068391,
    -1.17767998417887,
    0.235573213359357,
    0.784513610477560};

  double factor = 1.0; // to be accumulated

  for (int n1 = 0; n1 < n_sy; n1++) {
    double dt2 = duration * w[n1] / n_respa;
    double dt4 = dt2 * 0.5;
    double dt8 = dt4 * 0.5;
    for (int n2 = 0; n2 < n_respa; n2++) {

      // update momentum of the last (length - 1) thermostat:
      double G = momentum[length - 2] * momentum[length - 2] / mass[length - 2] - kT;
      momentum[length - 1] += dt4 * G;

      // update thermostat momenta from length - 2 to 0:
      for (int m = length - 2; m >= 0; m--) {
        double tmp = exp(-dt8 * momentum[m + 1] / mass[m + 1]);
        G = m == 0 ? twice_kinetic_energy - degrees_of_freedom * kT
                   : momentum[m - 1] * momentum[m - 1] / mass[m - 1] - kT;
        momentum[m] = tmp * (tmp * momentum[m] + dt4 * G);
      }

      // update thermostat positions from length - 1 to 0:
      for (int m = length - 1; m >= 0; m--) {
        position[m] += dt2 * momentum[m] / mass[m];
      }

      // compute the scale factor
      double factor_local = exp(-dt2 * momentum[0] / mass[0]);
      twice_kinetic_energy *= factor_local * factor_local;
      factor *= factor_local;

      // update thermostat momenta from 0 to length - 2:
      for (int m = 0; m < length - 1; m++) {
        double tmp = exp(-dt8 * momentum[m + 1] / mass[m + 1]);
        G = m == 0 ? twice_kinetic_energy - degrees_of_freedom * kT
                   : momentum[m - 1] * momentum[m - 1] / mass[m - 1] - kT;
        momentum[m] = tmp * (tmp * momentum[m] + dt4 * G);
      }

      // update momentum of the last (length - 1) thermostat:
      G = momentum[length - 2] * momentum[length - 2] / mass[length - 2] - kT;
      momentum[length - 1] += dt4 * G;
    }
  }
  return factor;
}
