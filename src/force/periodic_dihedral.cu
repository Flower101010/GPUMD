/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "periodic_dihedral.cuh"
#include "bonded_accumulation.cuh"
#include "bonded_geometry.cuh"
#include "utilities/error.cuh"
#include <stdexcept>

namespace
{
constexpr int BLOCK_SIZE = 128;

__global__ void gpu_compute_periodic_dihedrals(
  const int number_of_atoms,
  const int number_of_dihedrals,
  const Box box,
  const int* dihedral_atom_i,
  const int* dihedral_atom_j,
  const int* dihedral_atom_k,
  const int* dihedral_atom_l,
  const int* dihedral_type,
  const double* force_constant,
  const int* multiplicity,
  const double* phase,
  const double* position,
  double* potential,
  double* force,
  double* virial,
  int* errors)
{
  const int dihedral_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (dihedral_index >= number_of_dihedrals) {
    return;
  }

  const int atom_i = dihedral_atom_i[dihedral_index];
  const int atom_j = dihedral_atom_j[dihedral_index];
  const int atom_k = dihedral_atom_k[dihedral_index];
  const int atom_l = dihedral_atom_l[dihedral_index];
  const int type = dihedral_type[dihedral_index];
  if (force_constant[type] == 0.0) return;

  // Consecutive minimum-image bond vectors unwrap the four-atom chain without a neighbor list.
  double b1[3] = {
    position[atom_j] - position[atom_i],
    position[atom_j + number_of_atoms] - position[atom_i + number_of_atoms],
    position[atom_j + 2 * number_of_atoms] - position[atom_i + 2 * number_of_atoms]};
  double b2[3] = {
    position[atom_k] - position[atom_j],
    position[atom_k + number_of_atoms] - position[atom_j + number_of_atoms],
    position[atom_k + 2 * number_of_atoms] - position[atom_j + 2 * number_of_atoms]};
  double b3[3] = {
    position[atom_l] - position[atom_k],
    position[atom_l + number_of_atoms] - position[atom_k + number_of_atoms],
    position[atom_l + 2 * number_of_atoms] - position[atom_k + 2 * number_of_atoms]};
  apply_mic(box, b1[0], b1[1], b1[2]);
  apply_mic(box, b2[0], b2[1], b2[2]);
  apply_mic(box, b3[0], b3[1], b3[2]);

  double phi = 0.0, energy = 0.0, forces[4][3];
  const int atoms[4] = {atom_i, atom_j, atom_k, atom_l};
  const double r[4][3] = {{-b1[0],-b1[1],-b1[2]}, {0.0,0.0,0.0},
    {b2[0],b2[1],b2[2]}, {b2[0]+b3[0],b2[1]+b3[1],b2[2]+b3[2]}};
  if (!bonded_geometry::periodic_dihedral(b1, b2, b3, force_constant[type], multiplicity[type],
        phase[type], phi, energy, forces[0], forces[1], forces[2], forces[3]) ||
      !add_bonded_interaction(number_of_atoms, atoms, r, forces, energy, potential, force, virial)) {
    atomicMin(errors+2, dihedral_index);
  }

}
} // namespace

void PeriodicDihedral::compute(
  const Box& box,
  const PeriodicDihedralData& dihedral_data,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  BondedErrorState* shared_errors) const
{
  const int number_of_atoms = dihedral_data.number_of_atoms();
  if (position_per_atom.size() != static_cast<size_t>(3 * number_of_atoms) ||
      potential_per_atom.size() != static_cast<size_t>(number_of_atoms) ||
      force_per_atom.size() != static_cast<size_t>(3 * number_of_atoms) ||
      virial_per_atom.size() != static_cast<size_t>(9 * number_of_atoms)) {
    throw std::invalid_argument(
      "PeriodicDihedral output array sizes do not match number_of_atoms.");
  }

  const int number_of_dihedrals = static_cast<int>(dihedral_data.number_of_dihedrals());
  if (number_of_dihedrals == 0) {
    return;
  }

  BondedErrorState& errors = shared_errors ? *shared_errors : errors_;
  if (!shared_errors) errors.reset();

  const int grid_size = (number_of_dihedrals + BLOCK_SIZE - 1) / BLOCK_SIZE;
  gpu_compute_periodic_dihedrals<<<grid_size, BLOCK_SIZE>>>(
    number_of_atoms,
    number_of_dihedrals,
    box,
    dihedral_data.atom_i().data(),
    dihedral_data.atom_j().data(),
    dihedral_data.atom_k().data(),
    dihedral_data.atom_l().data(),
    dihedral_data.type().data(),
    dihedral_data.force_constant().data(),
    dihedral_data.multiplicity().data(),
    dihedral_data.phase().data(),
    position_per_atom.data(),
    potential_per_atom.data(),
    force_per_atom.data(),
    virial_per_atom.data(),
    errors.data());
  GPU_CHECK_KERNEL
  if (!shared_errors) errors.check();
}
