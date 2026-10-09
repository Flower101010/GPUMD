#pragma once

#include "utilities/gpu_vector.cuh"
#include <climits>
#include <stdexcept>
#include <string>

// One shared readback for all three MolecularForce kernels; storage is reused.
// Standalone calculators use their own state and perform one readback per call.
class BondedErrorState
{
public:
  void reset()
  {
    indices_.resize(3);
    indices_.fill(INT_MAX);
  }
  void clear() { indices_.clear(); }
  int* data() { return indices_.data(); }
  void check() const
  {
    int indices[3];
    indices_.copy_to_host(indices);
    const char* names[3] = {"harmonic bond", "harmonic angle", "periodic dihedral"};
    for (int family = 0; family < 3; ++family) {
      if (indices[family] != INT_MAX) {
        throw std::runtime_error(
          std::string("Invalid geometry in ") + names[family] + " interaction " +
          std::to_string(indices[family]) +
          " (0-based): zero-length, undefined collinear, or nonfinite bonded geometry/result.");
      }
    }
  }

private:
  GPU_Vector<int> indices_;
};

