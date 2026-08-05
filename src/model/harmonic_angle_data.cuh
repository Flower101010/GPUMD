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

#include "force_field_parameters.cuh"
#include "utilities/gpu_vector.cuh"

// GPU-ready structure-of-arrays representation for harmonic angles.
class HarmonicAngleData
{
public:
  HarmonicAngleData() = default;
  HarmonicAngleData(const HarmonicAngleData&) = delete;
  HarmonicAngleData& operator=(const HarmonicAngleData&) = delete;

  void upload(const Topology& topology, const ForceFieldParameters& parameters);
  void clear();

  int number_of_atoms() const { return number_of_atoms_; }
  size_t number_of_angles() const { return atom_i_.size(); }
  size_t number_of_parameters() const { return equilibrium_angle_.size(); }

  const GPU_Vector<int>& atom_i() const { return atom_i_; }
  const GPU_Vector<int>& atom_j() const { return atom_j_; }
  const GPU_Vector<int>& atom_k() const { return atom_k_; }
  const GPU_Vector<int>& type() const { return type_; }
  const GPU_Vector<double>& equilibrium_angle() const { return equilibrium_angle_; }
  const GPU_Vector<double>& angle_constant() const { return angle_constant_; }

private:
  int number_of_atoms_ = 0;
  GPU_Vector<int> atom_i_;
  GPU_Vector<int> atom_j_;
  GPU_Vector<int> atom_k_;
  GPU_Vector<int> type_;
  GPU_Vector<double> equilibrium_angle_;
  GPU_Vector<double> angle_constant_;
};
