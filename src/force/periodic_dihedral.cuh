/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once

#include "model/box.cuh"
#include "model/periodic_dihedral_data.cuh"
#include "utilities/gpu_vector.cuh"

class PeriodicDihedral
{
public:
  void compute(
    const Box& box,
    const PeriodicDihedralData& dihedral_data,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom) const;
};
