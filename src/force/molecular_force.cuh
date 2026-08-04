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

#include "harmonic_bond.cuh"
#include "model/force_field_parameters.cuh"
#include "model/topology.cuh"

// Entry point for fixed-topology molecular mechanics forces. Additional bonded and non-bonded
// modules can be added here without exposing them individually to the simulation loop.
class MolecularForce
{
public:
  void initialize(const Topology& topology, const ForceFieldParameters& parameters);
  void clear();

  bool is_initialized() const { return initialized_; }
  size_t number_of_harmonic_bonds() const { return harmonic_bond_data_.number_of_bonds(); }

  // Add molecular-mechanics contributions to pre-existing per-atom output arrays. Calling compute
  // before initialize is a no-op so existing GPUMD simulations remain unaffected when this module
  // is connected to the main loop.
  void compute(
    const Box& box,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom) const;

private:
  bool initialized_ = false;
  HarmonicBondData harmonic_bond_data_;
  HarmonicBond harmonic_bond_;
};
