#include "force/periodic_dihedral.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace
{
struct Result {
  std::vector<double> potential;
  std::vector<double> force;
  std::vector<double> virial;
};

bool gpu_is_available()
{
  int number_of_devices = 0;
  return gpuGetDeviceCount(&number_of_devices) == gpuSuccess && number_of_devices > 0;
}

bool nearly_equal(double actual, double expected, double tolerance = 1.0e-10)
{
  return std::abs(actual - expected) <= tolerance;
}

Box make_box(double length = 20.0, bool periodic = false)
{
  Box box;
  box.pbc_x = periodic;
  box.pbc_y = periodic;
  box.pbc_z = periodic;
  box.cpu_h[0] = length;
  box.cpu_h[4] = length;
  box.cpu_h[8] = length;
  box.is_orthogonal = true;
  return box;
}

Result compute(const Box& box, const std::vector<double>& position)
{
  Topology topology;
  topology.number_of_atoms = 4;
  topology.dihedrals = {{0, 1, 2, 3, 0}};
  ForceFieldParameters parameters;
  parameters.periodic_dihedral_parameters = {{1.7, 3, 0.4}};
  PeriodicDihedralData data;
  data.upload(topology, parameters);

  GPU_Vector<double> gpu_position(position.size());
  gpu_position.copy_from_host(position.data());
  GPU_Vector<double> potential(4, 0.0);
  GPU_Vector<double> force(12, 0.0);
  GPU_Vector<double> virial(36, 0.0);
  PeriodicDihedral dihedral;
  dihedral.compute(box, data, gpu_position, potential, force, virial);

  Result result;
  result.potential.resize(4);
  result.force.resize(12);
  result.virial.resize(36);
  potential.copy_to_host(result.potential.data());
  force.copy_to_host(result.force.data());
  virial.copy_to_host(result.virial.data());
  return result;
}

double total_energy(const Result& result)
{
  return std::accumulate(result.potential.begin(), result.potential.end(), 0.0);
}

void test_known_signed_dihedral_energy()
{
  // Consecutive bonds x, y, z give phi=+pi/2 with the documented convention.
  const std::vector<double> position =
    {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const Result result = compute(make_box(), position);
  const double expected = 1.7 * (1.0 + std::cos(3.0 * PI / 2.0 - 0.4));
  assert(nearly_equal(total_energy(result), expected));
  for (double atom_energy : result.potential) {
    assert(nearly_equal(atom_energy, expected * 0.25));
  }
}

void test_force_matches_finite_difference_total_force_and_pbc()
{
  // Atom 0 is across x PBC from atom 1; consecutive MIC vectors reconstruct the same chain.
  const Box box = make_box(10.0, true);
  std::vector<double> position = {
    9.1, 0.0, 0.4, 1.0, 0.2, 0.0, 1.1, 1.3, 9.9, 0.0, 0.2, 1.0};
  const Result center = compute(box, position);
  const double h = 1.0e-6;
  for (size_t coordinate = 0; coordinate < position.size(); ++coordinate) {
    std::vector<double> plus = position;
    std::vector<double> minus = position;
    plus[coordinate] += h;
    minus[coordinate] -= h;
    const double numerical_force = -(total_energy(compute(box, plus)) -
                                     total_energy(compute(box, minus))) /
      (2.0 * h);
    assert(nearly_equal(center.force[coordinate], numerical_force, 5.0e-8));
  }
  for (int d = 0; d < 3; ++d) {
    double total_force = 0.0;
    for (int atom = 0; atom < 4; ++atom) {
      total_force += center.force[d * 4 + atom];
    }
    assert(nearly_equal(total_force, 0.0));
  }
}
} // namespace

int main()
{
  if (!gpu_is_available()) {
    std::cout << "SKIP: no accessible GPU for PeriodicDihedral runtime tests.\n";
    return 0;
  }
  test_known_signed_dihedral_energy();
  test_force_matches_finite_difference_total_force_and_pbc();
  CHECK(gpuDeviceSynchronize());
  std::cout << "PASS: PeriodicDihedral energy, force, PBC, and finite-difference tests.\n";
  return 0;
}
