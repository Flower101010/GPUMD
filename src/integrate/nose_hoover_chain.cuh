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

#pragma once

namespace nose_hoover_chain {

// Advance a thermostat chain on the CPU and return the physical momentum scale.
// position = eta; momentum = pi (NOT friction xi); mass = Q; xi = pi / Q.
// twice_kinetic_energy = 2K of all coupled physical degrees of freedom.
// kT = Boltzmann constant * target temperature; degrees_of_freedom = coupled f.
// duration is the actual propagation interval (usually dt/2), in GPUMD time units.
// The caller must apply the returned factor to the coupled physical momenta.
// For a rigid body, these are M*vcm and Miller's conjugate momentum p; do not
// scale quat or independently scale the constrained bead velocities.
//
// Requires length >= 2, positive finite Q/kT/f, finite eta/pi/duration and
// finite nonnegative 2K. Throws std::invalid_argument before modifying state
// for invalid input. Negative duration is supported for backward propagation.
// Arrays have length entries and must not overlap; mass is unchanged.
// Initialization is owned by the caller, conventionally Q[0] = f*kT*tau^2,
// Q[j>0] = kT*tau^2. This function does not choose the thermostat coupling.
double propagate(
  int length,
  double* position,
  double* momentum,
  const double* mass,
  double twice_kinetic_energy,
  double kT,
  double degrees_of_freedom,
  double duration);

} // namespace nose_hoover_chain
