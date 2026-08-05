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

#include "harmonic_angle_data.cuh"
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

void HarmonicAngleData::upload(
  const Topology& topology, const ForceFieldParameters& parameters)
{
  parameters.validate_or_throw(topology);
  number_of_atoms_ = topology.number_of_atoms;

  std::vector<int> atom_i(topology.angles.size());
  std::vector<int> atom_j(topology.angles.size());
  std::vector<int> atom_k(topology.angles.size());
  std::vector<int> type(topology.angles.size());
  for (size_t i = 0; i < topology.angles.size(); ++i) {
    atom_i[i] = topology.angles[i].atom_i;
    atom_j[i] = topology.angles[i].atom_j;
    atom_k[i] = topology.angles[i].atom_k;
    type[i] = topology.angles[i].type;
  }

  std::vector<double> equilibrium_angle(parameters.harmonic_angle_parameters.size());
  std::vector<double> angle_constant(parameters.harmonic_angle_parameters.size());
  for (size_t i = 0; i < parameters.harmonic_angle_parameters.size(); ++i) {
    equilibrium_angle[i] = parameters.harmonic_angle_parameters[i].equilibrium_angle;
    angle_constant[i] = parameters.harmonic_angle_parameters[i].angle_constant;
  }

  upload_vector(atom_i, atom_i_);
  upload_vector(atom_j, atom_j_);
  upload_vector(atom_k, atom_k_);
  upload_vector(type, type_);
  upload_vector(equilibrium_angle, equilibrium_angle_);
  upload_vector(angle_constant, angle_constant_);
}

void HarmonicAngleData::clear()
{
  number_of_atoms_ = 0;
  atom_i_.clear();
  atom_j_.clear();
  atom_k_.clear();
  type_.clear();
  equilibrium_angle_.clear();
  angle_constant_.clear();
}
