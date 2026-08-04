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

#include "molecular_force.cuh"

void MolecularForce::initialize(
  const Topology& topology, const ForceFieldParameters& parameters)
{
  harmonic_bond_data_.upload(topology, parameters);
  initialized_ = true;
}

void MolecularForce::clear()
{
  harmonic_bond_data_.clear();
  initialized_ = false;
}

void MolecularForce::compute(
  const Box& box,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom) const
{
  if (!initialized_) {
    return;
  }

  harmonic_bond_.compute(
    box,
    harmonic_bond_data_,
    position_per_atom,
    potential_per_atom,
    force_per_atom,
    virial_per_atom);
}
