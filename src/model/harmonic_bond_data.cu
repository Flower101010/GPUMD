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

#include "harmonic_bond_data.cuh"
#include <vector>

namespace
{
template <typename T>
void upload_vector(const std::vector<T>& host, GPU_Vector<T>& device)
{
  device.resize(host.size());
  if (!host.empty()) {
    device.copy_from_host(host.data());
  }
}
} // namespace

void HarmonicBondData::upload(
  const Topology& topology, const ForceFieldParameters& parameters)
{
  parameters.validate_or_throw(topology);

  std::vector<int> atom_i(topology.bonds.size());
  std::vector<int> atom_j(topology.bonds.size());
  std::vector<int> type(topology.bonds.size());
  for (size_t i = 0; i < topology.bonds.size(); ++i) {
    atom_i[i] = topology.bonds[i].atom_i;
    atom_j[i] = topology.bonds[i].atom_j;
    type[i] = topology.bonds[i].type;
  }

  std::vector<double> equilibrium_distance(parameters.harmonic_bond_parameters.size());
  std::vector<double> force_constant(parameters.harmonic_bond_parameters.size());
  for (size_t i = 0; i < parameters.harmonic_bond_parameters.size(); ++i) {
    equilibrium_distance[i] = parameters.harmonic_bond_parameters[i].equilibrium_distance;
    force_constant[i] = parameters.harmonic_bond_parameters[i].force_constant;
  }

  upload_vector(atom_i, atom_i_);
  upload_vector(atom_j, atom_j_);
  upload_vector(type, type_);
  upload_vector(equilibrium_distance, equilibrium_distance_);
  upload_vector(force_constant, force_constant_);
}

void HarmonicBondData::clear()
{
  atom_i_.clear();
  atom_j_.clear();
  type_.clear();
  equilibrium_distance_.clear();
  force_constant_.clear();
}
