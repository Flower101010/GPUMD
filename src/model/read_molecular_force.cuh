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

#include "force_field_parameters.cuh"
#include "topology.cuh"
#include <string>

struct MolecularForceDefinition {
  Topology topology;
  ForceFieldParameters parameters;
};

// Version-1 file format (indices are zero-based; distances are angstrom; force constants are
// eV/angstrom^2):
//
//   gpumd_molecular_force 1
//   number_of_atoms N
//   harmonic_bond_parameters M
//   equilibrium_distance force_constant  # repeated M times
//   bonds B
//   atom_i atom_j type                    # repeated B times
//
// Blank lines and comments beginning with # are ignored.
MolecularForceDefinition read_molecular_force(const std::string& filename);
