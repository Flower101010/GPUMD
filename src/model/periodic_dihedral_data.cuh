/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once

#include "force_field_parameters.cuh"
#include "utilities/gpu_vector.cuh"

// GPU-ready structure-of-arrays representation for periodic proper dihedrals.
class PeriodicDihedralData
{
public:
  PeriodicDihedralData() = default;
  PeriodicDihedralData(const PeriodicDihedralData&) = delete;
  PeriodicDihedralData& operator=(const PeriodicDihedralData&) = delete;

  void upload(const Topology& topology, const ForceFieldParameters& parameters);
  void clear();

  int number_of_atoms() const { return number_of_atoms_; }
  size_t number_of_dihedrals() const { return atom_i_.size(); }
  size_t number_of_parameters() const { return force_constant_.size(); }

  const GPU_Vector<int>& atom_i() const { return atom_i_; }
  const GPU_Vector<int>& atom_j() const { return atom_j_; }
  const GPU_Vector<int>& atom_k() const { return atom_k_; }
  const GPU_Vector<int>& atom_l() const { return atom_l_; }
  const GPU_Vector<int>& type() const { return type_; }
  const GPU_Vector<double>& force_constant() const { return force_constant_; }
  const GPU_Vector<int>& multiplicity() const { return multiplicity_; }
  const GPU_Vector<double>& phase() const { return phase_; }

private:
  int number_of_atoms_ = 0;
  GPU_Vector<int> atom_i_;
  GPU_Vector<int> atom_j_;
  GPU_Vector<int> atom_k_;
  GPU_Vector<int> atom_l_;
  GPU_Vector<int> type_;
  GPU_Vector<double> force_constant_;
  GPU_Vector<int> multiplicity_;
  GPU_Vector<double> phase_;
};
