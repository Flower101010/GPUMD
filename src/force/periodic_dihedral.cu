/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "periodic_dihedral.cuh"
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
  double* virial)
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

  double phi = 0.0;
  double energy = 0.0;
  double force_i[3];
  double force_j[3];
  double force_k[3];
  double force_l[3];
  if (!bonded_geometry::periodic_dihedral(
        b1,
        b2,
        b3,
        force_constant[type],
        multiplicity[type],
        phase[type],
        phi,
        energy,
        force_i,
        force_j,
        force_k,
        force_l)) {
    return;
  }

  const int atoms[4] = {atom_i, atom_j, atom_k, atom_l};
  const double* forces[4] = {force_i, force_j, force_k, force_l};
  for (int p = 0; p < 4; ++p) {
    atomicAdd(&force[atoms[p]], forces[p][0]);
    atomicAdd(&force[atoms[p] + number_of_atoms], forces[p][1]);
    atomicAdd(&force[atoms[p] + 2 * number_of_atoms], forces[p][2]);
    atomicAdd(&potential[atoms[p]], energy * 0.25);
  }

  // Use atom j as origin: r_i=-b1, r_k=b2, r_l=b2+b3.
  const double relative_i[3] = {-b1[0], -b1[1], -b1[2]};
  const double relative_k[3] = {b2[0], b2[1], b2[2]};
  const double relative_l[3] = {b2[0] + b3[0], b2[1] + b3[1], b2[2] + b3[2]};
  const double* relative[3] = {relative_i, relative_k, relative_l};
  const double* relative_force[3] = {force_i, force_k, force_l};
  const int row[9] = {0, 1, 2, 0, 0, 1, 1, 2, 2};
  const int column[9] = {0, 1, 2, 1, 2, 2, 0, 0, 1};
  for (int component = 0; component < 9; ++component) {
    double interaction_virial = 0.0;
    for (int p = 0; p < 3; ++p) {
      interaction_virial += relative[p][row[component]] * relative_force[p][column[component]];
    }
    const double per_atom_virial = interaction_virial * 0.25;
    for (int p = 0; p < 4; ++p) {
      atomicAdd(&virial[atoms[p] + component * number_of_atoms], per_atom_virial);
    }
  }
}
} // namespace

void PeriodicDihedral::compute(
  const Box& box,
  const PeriodicDihedralData& dihedral_data,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom) const
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
    virial_per_atom.data());
  GPU_CHECK_KERNEL
}
