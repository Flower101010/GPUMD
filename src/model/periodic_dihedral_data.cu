/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "periodic_dihedral_data.cuh"
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

void PeriodicDihedralData::upload(
  const Topology& topology, const ForceFieldParameters& parameters)
{
  parameters.validate_or_throw(topology);
  number_of_atoms_ = topology.number_of_atoms;

  std::vector<int> atom_i(topology.dihedrals.size());
  std::vector<int> atom_j(topology.dihedrals.size());
  std::vector<int> atom_k(topology.dihedrals.size());
  std::vector<int> atom_l(topology.dihedrals.size());
  std::vector<int> type(topology.dihedrals.size());
  for (size_t i = 0; i < topology.dihedrals.size(); ++i) {
    atom_i[i] = topology.dihedrals[i].atom_i;
    atom_j[i] = topology.dihedrals[i].atom_j;
    atom_k[i] = topology.dihedrals[i].atom_k;
    atom_l[i] = topology.dihedrals[i].atom_l;
    type[i] = topology.dihedrals[i].type;
  }

  const size_t number_of_parameters = parameters.periodic_dihedral_parameters.size();
  std::vector<double> force_constant(number_of_parameters);
  std::vector<int> multiplicity(number_of_parameters);
  std::vector<double> phase(number_of_parameters);
  for (size_t i = 0; i < number_of_parameters; ++i) {
    force_constant[i] = parameters.periodic_dihedral_parameters[i].force_constant;
    multiplicity[i] = parameters.periodic_dihedral_parameters[i].multiplicity;
    phase[i] = parameters.periodic_dihedral_parameters[i].phase;
  }

  upload_vector(atom_i, atom_i_);
  upload_vector(atom_j, atom_j_);
  upload_vector(atom_k, atom_k_);
  upload_vector(atom_l, atom_l_);
  upload_vector(type, type_);
  upload_vector(force_constant, force_constant_);
  upload_vector(multiplicity, multiplicity_);
  upload_vector(phase, phase_);
}

void PeriodicDihedralData::clear()
{
  number_of_atoms_ = 0;
  atom_i_.clear();
  atom_j_.clear();
  atom_k_.clear();
  atom_l_.clear();
  type_.clear();
  force_constant_.clear();
  multiplicity_.clear();
  phase_.clear();
}
