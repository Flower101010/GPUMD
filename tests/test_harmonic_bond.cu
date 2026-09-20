#include "force/harmonic_bond.cuh"
#include "utilities/gpu_macro.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace
{
constexpr double TOLERANCE = 1.0e-10;

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

bool nearly_equal(const double actual, const double expected, const double tolerance = TOLERANCE)
{
  return std::abs(actual - expected) <= tolerance;
}

Box make_orthogonal_box(const double length, const bool periodic)
{
  Box box;
  box.pbc_x = periodic ? 1 : 0;
  box.pbc_y = periodic ? 1 : 0;
  box.pbc_z = periodic ? 1 : 0;
  box.cpu_h[0] = length;
  box.cpu_h[4] = length;
  box.cpu_h[8] = length;
  box.is_orthogonal = true;
  return box;
}

Result compute(
  const Topology& topology,
  const ForceFieldParameters& parameters,
  const Box& box,
  const std::vector<double>& position)
{
  HarmonicBondData bond_data;
  bond_data.upload(topology, parameters);

  GPU_Vector<double> gpu_position(position.size());
  gpu_position.copy_from_host(position.data());

  GPU_Vector<double> gpu_potential(topology.number_of_atoms, 0.0);
  GPU_Vector<double> gpu_force(3 * topology.number_of_atoms, 0.0);
  GPU_Vector<double> gpu_virial(9 * topology.number_of_atoms, 0.0);

  HarmonicBond harmonic_bond;
  harmonic_bond.compute(
    box, bond_data, gpu_position, gpu_potential, gpu_force, gpu_virial);

  Result result;
  result.potential.resize(gpu_potential.size());
  result.force.resize(gpu_force.size());
  result.virial.resize(gpu_virial.size());
  gpu_potential.copy_to_host(result.potential.data());
  gpu_force.copy_to_host(result.force.data());
  gpu_virial.copy_to_host(result.virial.data());
  return result;
}

double total_energy(const Result& result)
{
  return std::accumulate(result.potential.begin(), result.potential.end(), 0.0);
}

void test_equilibrium_stretched_and_compressed_bonds()
{
  Topology topology;
  topology.number_of_atoms = 2;
  topology.bonds = {{0, 1, 0}};

  ForceFieldParameters parameters;
  parameters.harmonic_bond_parameters = {{1.5, 20.0}};
  const Box box = make_orthogonal_box(20.0, false);

  Result result = compute(topology, parameters, box, {0.0, 1.5, 0.0, 0.0, 0.0, 0.0});
  assert(nearly_equal(total_energy(result), 0.0));
  assert(nearly_equal(result.force[0], 0.0));
  assert(nearly_equal(result.force[1], 0.0));

  result = compute(topology, parameters, box, {0.0, 2.0, 0.0, 0.0, 0.0, 0.0});
  assert(nearly_equal(result.potential[0], 1.25));
  assert(nearly_equal(result.potential[1], 1.25));
  assert(nearly_equal(result.force[0], 10.0));
  assert(nearly_equal(result.force[1], -10.0));
  assert(nearly_equal(result.virial[0], -10.0));
  assert(nearly_equal(result.virial[1], -10.0));

  result = compute(topology, parameters, box, {0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
  assert(nearly_equal(total_energy(result), 2.5));
  assert(nearly_equal(result.force[0], -10.0));
  assert(nearly_equal(result.force[1], 10.0));
  assert(nearly_equal(result.virial[0], 5.0));
  assert(nearly_equal(result.virial[1], 5.0));
}

void test_periodic_boundary_bond()
{
  Topology topology;
  topology.number_of_atoms = 2;
  topology.bonds = {{0, 1, 0}};

  ForceFieldParameters parameters;
  parameters.harmonic_bond_parameters = {{0.5, 10.0}};
  const Box box = make_orthogonal_box(10.0, true);

  const Result result =
    compute(topology, parameters, box, {0.2, 9.8, 0.0, 0.0, 0.0, 0.0});
  assert(nearly_equal(total_energy(result), 0.05));
  assert(nearly_equal(result.force[0], 1.0));
  assert(nearly_equal(result.force[1], -1.0));
  assert(nearly_equal(result.virial[0], 0.2));
  assert(nearly_equal(result.virial[1], 0.2));
}

void test_multiple_bonds_accumulate_on_shared_atom()
{
  Topology topology;
  topology.number_of_atoms = 3;
  topology.bonds = {{0, 1, 0}, {1, 2, 0}};

  ForceFieldParameters parameters;
  parameters.harmonic_bond_parameters = {{1.5, 20.0}};
  const Box box = make_orthogonal_box(20.0, false);

  const Result result =
    compute(topology, parameters, box, {0.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  assert(nearly_equal(total_energy(result), 5.0));
  assert(nearly_equal(result.force[0], 10.0));
  assert(nearly_equal(result.force[1], 0.0));
  assert(nearly_equal(result.force[2], -10.0));
  assert(nearly_equal(result.virial[0], -10.0));
  assert(nearly_equal(result.virial[1], -20.0));
  assert(nearly_equal(result.virial[2], -10.0));
}

void test_force_matches_finite_difference_energy_gradient()
{
  Topology topology;
  topology.number_of_atoms = 2;
  topology.bonds = {{0, 1, 0}};

  ForceFieldParameters parameters;
  parameters.harmonic_bond_parameters = {{1.5, 20.0}};
  const Box box = make_orthogonal_box(20.0, false);
  const double h = 1.0e-6;

  const Result center =
    compute(topology, parameters, box, {0.0, 2.0, 0.0, 0.0, 0.0, 0.0});
  const Result plus =
    compute(topology, parameters, box, {0.0, 2.0 + h, 0.0, 0.0, 0.0, 0.0});
  const Result minus =
    compute(topology, parameters, box, {0.0, 2.0 - h, 0.0, 0.0, 0.0, 0.0});

  const double numerical_force_on_j = -(total_energy(plus) - total_energy(minus)) / (2.0 * h);
  assert(nearly_equal(center.force[1], numerical_force_on_j, 1.0e-8));
}
} // namespace

int main()
{
  if (!gpu_is_available()) {
    std::cout << "SKIP: no accessible GPU for HarmonicBond runtime tests.\n";
    return 0;
  }

  test_equilibrium_stretched_and_compressed_bonds();
  test_periodic_boundary_bond();
  test_multiple_bonds_accumulate_on_shared_atom();
  test_force_matches_finite_difference_energy_gradient();
  CHECK(gpuDeviceSynchronize());

  std::cout << "PASS: HarmonicBond energy, force, virial, PBC, and finite-difference tests.\n";
  return 0;
}
