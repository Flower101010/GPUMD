#pragma once

#include "bonded_geometry.cuh"
#include "utilities/gpu_macro.cuh"

template <int N>
__device__ inline bool add_bonded_interaction(
  int number_of_atoms,
  const int (&atoms)[N],
  const double (&r)[N][3],
  const double (&forces)[N][3],
  double energy,
  double* potential,
  double* force,
  double* virial)
{
  double tensor[9];
  if (!bonded_geometry::interaction_virial(r, forces, tensor))
    return false;
  for (int i = 0; i < N; ++i) {
    atomicAdd(&potential[atoms[i]], energy / N);
    for (int d = 0; d < 3; ++d)
      atomicAdd(&force[atoms[i] + d * number_of_atoms], forces[i][d]);
    for (int c = 0; c < 9; ++c)
      atomicAdd(&virial[atoms[i] + c * number_of_atoms], tensor[c] / N);
  }
  return true;
}
