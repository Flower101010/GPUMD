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
#include "bonded_accumulation.cuh"
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
  double* virial,
  int* errors)
{
  const int bond_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (bond_index >= number_of_bonds) {
    return;
  }

  const int atom_i = bond_atom_i[bond_index];
  const int atom_j = bond_atom_j[bond_index];
  const int type = bond_type[bond_index];
  if (force_constant[type] == 0.0) return;

  double dx = position[atom_j] - position[atom_i];
  double dy = position[atom_j + number_of_atoms] - position[atom_i + number_of_atoms];
  double dz = position[atom_j + 2 * number_of_atoms] - position[atom_i + 2 * number_of_atoms];
  apply_mic(box, dx, dy, dz);

  const double r[2][3] = {{0.0, 0.0, 0.0}, {dx, dy, dz}};
  double forces[2][3], energy = 0.0;
  const int atoms[2] = {atom_i, atom_j};
  if (!bonded_geometry::harmonic_bond(r[1], equilibrium_distance[type], force_constant[type],
        energy, forces[0], forces[1]) ||
      !add_bonded_interaction(number_of_atoms, atoms, r, forces, energy, potential, force, virial)) {
    atomicMin(errors, bond_index);
  }

}
} // namespace

void HarmonicBond::compute(
  const Box& box,
  const HarmonicBondData& bond_data,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  BondedErrorState* shared_errors) const
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

  BondedErrorState& errors = shared_errors ? *shared_errors : errors_;
  if (!shared_errors) errors.reset();

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
    virial_per_atom.data(),
    errors.data());
  GPU_CHECK_KERNEL
  if (!shared_errors) errors.check();
}
