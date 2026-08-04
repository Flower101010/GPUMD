/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
    even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details. You should have received a copy of the GNU General
    Public License along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "topology.cuh"
#include <string>
#include <vector>

// Internal units: equilibrium distance in angstrom, force constant in eV/angstrom^2.
// The corresponding potential is U(r) = 0.5 * force_constant * (r - equilibrium_distance)^2.
struct HarmonicBondParameter {
  double equilibrium_distance;
  double force_constant;
};

class ForceFieldParameters
{
public:
  // During the harmonic-bond milestone, Bond::type directly indexes this table. A function-type
  // registry can be added later without changing the Topology interaction records.
  std::vector<HarmonicBondParameter> harmonic_bond_parameters;

  // Validate both the parameter values and all Topology references to this parameter table.
  std::vector<std::string> validate(const Topology& topology) const;

  // Throw std::runtime_error containing every topology and parameter validation error.
  void validate_or_throw(const Topology& topology) const;

  void clear();
};
