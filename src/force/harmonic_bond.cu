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

#include "harmonic_bond.cuh"
#include "utilities/error.cuh"
#include <stdexcept>

namespace
{
constexpr int BLOCK_SIZE = 128;

__global__ void gpu_compute_harmonic_bonds(
  const int number_of_atoms,
  const int number_of_bonds,
  const Box box,
  const int* bond_atom_i,
  const int* bond_atom_j,
  const int* bond_type,
  const double* equilibrium_distance,
  const double* force_constant,
  const double* position,
  double* potential,
  double* force,
  double* virial)
{
  const int bond_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (bond_index >= number_of_bonds) {
    return;
  }

  const int atom_i = bond_atom_i[bond_index];
  const int atom_j = bond_atom_j[bond_index];
  const int type = bond_type[bond_index];

  double dx = position[atom_j] - position[atom_i];
  double dy = position[atom_j + number_of_atoms] - position[atom_i + number_of_atoms];
  double dz = position[atom_j + 2 * number_of_atoms] - position[atom_i + 2 * number_of_atoms];
  apply_mic(box, dx, dy, dz);

  const double distance_squared = dx * dx + dy * dy + dz * dz;
  if (distance_squared == 0.0) {
    return;
  }

  const double distance = sqrt(distance_squared);
  const double displacement = distance - equilibrium_distance[type];
  const double energy = 0.5 * force_constant[type] * displacement * displacement;
  const double force_over_distance = force_constant[type] * displacement / distance;
  const double force_x = force_over_distance * dx;
  const double force_y = force_over_distance * dy;
  const double force_z = force_over_distance * dz;

  atomicAdd(&force[atom_i], force_x);
  atomicAdd(&force[atom_i + number_of_atoms], force_y);
  atomicAdd(&force[atom_i + 2 * number_of_atoms], force_z);
  atomicAdd(&force[atom_j], -force_x);
  atomicAdd(&force[atom_j + number_of_atoms], -force_y);
  atomicAdd(&force[atom_j + 2 * number_of_atoms], -force_z);

  const double half_energy = 0.5 * energy;
  atomicAdd(&potential[atom_i], half_energy);
  atomicAdd(&potential[atom_j], half_energy);

  // GPUMD virial layout: xx, yy, zz, xy, xz, yz, yx, zx, zy. Split the pair virial
  // -r_ij tensor F_i equally between the two atoms.
  const double half_virial[9] = {
    -0.5 * dx * force_x,
    -0.5 * dy * force_y,
    -0.5 * dz * force_z,
    -0.5 * dx * force_y,
    -0.5 * dx * force_z,
    -0.5 * dy * force_z,
    -0.5 * dy * force_x,
    -0.5 * dz * force_x,
    -0.5 * dz * force_y};

  for (int component = 0; component < 9; ++component) {
    atomicAdd(&virial[atom_i + component * number_of_atoms], half_virial[component]);
    atomicAdd(&virial[atom_j + component * number_of_atoms], half_virial[component]);
  }
}
} // namespace

void HarmonicBond::compute(
  const Box& box,
  const HarmonicBondData& bond_data,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom) const
{
  const int number_of_atoms = bond_data.number_of_atoms();
  if (position_per_atom.size() != static_cast<size_t>(3 * number_of_atoms) ||
      potential_per_atom.size() != static_cast<size_t>(number_of_atoms) ||
      force_per_atom.size() != static_cast<size_t>(3 * number_of_atoms) ||
      virial_per_atom.size() != static_cast<size_t>(9 * number_of_atoms)) {
    throw std::invalid_argument("HarmonicBond output array sizes do not match number_of_atoms.");
  }

  const int number_of_bonds = static_cast<int>(bond_data.number_of_bonds());
  if (number_of_bonds == 0) {
    return;
  }

  const int grid_size = (number_of_bonds + BLOCK_SIZE - 1) / BLOCK_SIZE;
  gpu_compute_harmonic_bonds<<<grid_size, BLOCK_SIZE>>>(
    number_of_atoms,
    number_of_bonds,
    box,
    bond_data.atom_i().data(),
    bond_data.atom_j().data(),
    bond_data.type().data(),
    bond_data.equilibrium_distance().data(),
    bond_data.force_constant().data(),
    position_per_atom.data(),
    potential_per_atom.data(),
    force_per_atom.data(),
    virial_per_atom.data());
  GPU_CHECK_KERNEL
}
