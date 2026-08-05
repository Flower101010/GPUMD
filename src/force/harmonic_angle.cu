/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "harmonic_angle.cuh"
#include "bonded_geometry.cuh"
#include "utilities/error.cuh"
#include <stdexcept>

namespace
{
constexpr int BLOCK_SIZE = 128;

__global__ void gpu_compute_harmonic_angles(
  const int number_of_atoms,
  const int number_of_angles,
  const Box box,
  const int* angle_atom_i,
  const int* angle_atom_j,
  const int* angle_atom_k,
  const int* angle_type,
  const double* equilibrium_angle,
  const double* angle_constant,
  const double* position,
  double* potential,
  double* force,
  double* virial)
{
  const int angle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (angle_index >= number_of_angles) {
    return;
  }

  const int atom_i = angle_atom_i[angle_index];
  const int atom_j = angle_atom_j[angle_index];
  const int atom_k = angle_atom_k[angle_index];
  const int type = angle_type[angle_index];

  double a[3] = {
    position[atom_i] - position[atom_j],
    position[atom_i + number_of_atoms] - position[atom_j + number_of_atoms],
    position[atom_i + 2 * number_of_atoms] - position[atom_j + 2 * number_of_atoms]};
  double b[3] = {
    position[atom_k] - position[atom_j],
    position[atom_k + number_of_atoms] - position[atom_j + number_of_atoms],
    position[atom_k + 2 * number_of_atoms] - position[atom_j + 2 * number_of_atoms]};
  apply_mic(box, a[0], a[1], a[2]);
  apply_mic(box, b[0], b[1], b[2]);

  double energy = 0.0;
  double force_i[3];
  double force_k[3];
  double force_j[3];
  if (!bonded_geometry::harmonic_angle(
        a,
        b,
        equilibrium_angle[type],
        angle_constant[type],
        energy,
        force_i,
        force_j,
        force_k)) {
    return;
  }

  const int atoms[3] = {atom_i, atom_j, atom_k};
  const double* forces[3] = {force_i, force_j, force_k};
  for (int p = 0; p < 3; ++p) {
    atomicAdd(&force[atoms[p]], forces[p][0]);
    atomicAdd(&force[atoms[p] + number_of_atoms], forces[p][1]);
    atomicAdd(&force[atoms[p] + 2 * number_of_atoms], forces[p][2]);
    atomicAdd(&potential[atoms[p]], energy / 3.0);
  }

  // The interaction virial is sum_a r_a tensor F_a. Use atom j as the origin and
  // distribute the resulting tensor equally among the three participating atoms.
  const double interaction_virial[9] = {
    a[0] * force_i[0] + b[0] * force_k[0],
    a[1] * force_i[1] + b[1] * force_k[1],
    a[2] * force_i[2] + b[2] * force_k[2],
    a[0] * force_i[1] + b[0] * force_k[1],
    a[0] * force_i[2] + b[0] * force_k[2],
    a[1] * force_i[2] + b[1] * force_k[2],
    a[1] * force_i[0] + b[1] * force_k[0],
    a[2] * force_i[0] + b[2] * force_k[0],
    a[2] * force_i[1] + b[2] * force_k[1]};
  for (int component = 0; component < 9; ++component) {
    const double per_atom_virial = interaction_virial[component] / 3.0;
    for (int p = 0; p < 3; ++p) {
      atomicAdd(&virial[atoms[p] + component * number_of_atoms], per_atom_virial);
    }
  }
}
} // namespace

void HarmonicAngle::compute(
  const Box& box,
  const HarmonicAngleData& angle_data,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom) const
{
  const int number_of_atoms = angle_data.number_of_atoms();
  if (position_per_atom.size() != static_cast<size_t>(3 * number_of_atoms) ||
      potential_per_atom.size() != static_cast<size_t>(number_of_atoms) ||
      force_per_atom.size() != static_cast<size_t>(3 * number_of_atoms) ||
      virial_per_atom.size() != static_cast<size_t>(9 * number_of_atoms)) {
    throw std::invalid_argument("HarmonicAngle output array sizes do not match number_of_atoms.");
  }

  const int number_of_angles = static_cast<int>(angle_data.number_of_angles());
  if (number_of_angles == 0) {
    return;
  }

  const int grid_size = (number_of_angles + BLOCK_SIZE - 1) / BLOCK_SIZE;
  gpu_compute_harmonic_angles<<<grid_size, BLOCK_SIZE>>>(
    number_of_atoms,
    number_of_angles,
    box,
    angle_data.atom_i().data(),
    angle_data.atom_j().data(),
    angle_data.atom_k().data(),
    angle_data.type().data(),
    angle_data.equilibrium_angle().data(),
    angle_data.angle_constant().data(),
    position_per_atom.data(),
    potential_per_atom.data(),
    force_per_atom.data(),
    virial_per_atom.data());
  GPU_CHECK_KERNEL
}
