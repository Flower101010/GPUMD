#include "force/molecular_force.cuh"
#include "utilities/gpu_macro.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace
{
constexpr double TOLERANCE = 1.0e-10;

bool gpu_is_available()
{
  int number_of_devices = 0;
  return gpuGetDeviceCount(&number_of_devices) == gpuSuccess && number_of_devices > 0;
}

bool nearly_equal(const double actual, const double expected)
{
  return std::abs(actual - expected) <= TOLERANCE;
}

Box make_box()
{
  Box box;
  box.pbc_x = 0;
  box.pbc_y = 0;
  box.pbc_z = 0;
  box.cpu_h[0] = 20.0;
  box.cpu_h[4] = 20.0;
  box.cpu_h[8] = 20.0;
  box.is_orthogonal = true;
  return box;
}

template <typename T>
std::vector<T> copy_to_host(const GPU_Vector<T>& device)
{
  std::vector<T> host(device.size());
  if (!host.empty()) {
    device.copy_to_host(host.data());
  }
  return host;
}

void test_uninitialized_module_is_no_op()
{
  MolecularForce molecular_force;
  assert(!molecular_force.is_initialized());
  assert(molecular_force.number_of_harmonic_bonds() == 0);
  assert(molecular_force.number_of_harmonic_angles() == 0);
  assert(molecular_force.number_of_periodic_dihedrals() == 0);

  GPU_Vector<double> empty;
  molecular_force.compute(make_box(), empty, empty, empty, empty);
}

void test_end_to_end_initialization_compute_accumulation_and_clear()
{
  Topology topology;
  topology.number_of_atoms = 2;
  topology.bonds = {{0, 1, 0}};

  ForceFieldParameters parameters;
  parameters.harmonic_bond_parameters = {{1.5, 20.0}};

  MolecularForce molecular_force;
  molecular_force.initialize(topology, parameters);
  assert(molecular_force.is_initialized());
  assert(molecular_force.number_of_harmonic_bonds() == 1);
  assert(molecular_force.number_of_harmonic_angles() == 0);
  assert(molecular_force.number_of_periodic_dihedrals() == 0);

  const std::vector<double> position = {0.0, 2.0, 0.0, 0.0, 0.0, 0.0};
  GPU_Vector<double> gpu_position(position.size());
  gpu_position.copy_from_host(position.data());

  // MolecularForce adds to existing values rather than clearing contributions from other forces.
  GPU_Vector<double> potential(2, 1.0);
  GPU_Vector<double> force(6, 2.0);
  GPU_Vector<double> virial(18, 3.0);
  molecular_force.compute(make_box(), gpu_position, potential, force, virial);

  const std::vector<double> host_potential = copy_to_host(potential);
  const std::vector<double> host_force = copy_to_host(force);
  const std::vector<double> host_virial = copy_to_host(virial);
  assert(nearly_equal(host_potential[0], 2.25));
  assert(nearly_equal(host_potential[1], 2.25));
  assert(nearly_equal(host_force[0], 12.0));
  assert(nearly_equal(host_force[1], -8.0));
  assert(nearly_equal(host_force[2], 2.0));
  assert(nearly_equal(host_virial[0], -7.0));
  assert(nearly_equal(host_virial[1], -7.0));
  assert(nearly_equal(host_virial[2], 3.0));

  molecular_force.clear();
  assert(!molecular_force.is_initialized());
  assert(molecular_force.number_of_harmonic_bonds() == 0);
  assert(molecular_force.number_of_harmonic_angles() == 0);
  assert(molecular_force.number_of_periodic_dihedrals() == 0);
}

void test_all_bonded_styles_are_orchestrated_once()
{
  Topology topology;
  topology.number_of_atoms = 4;
  topology.bonds = {{0, 1, 0}};
  topology.angles = {{0, 1, 2, 0}};
  topology.dihedrals = {{0, 1, 2, 3, 0}};

  ForceFieldParameters parameters;
  parameters.harmonic_bond_parameters = {{1.0, 10.0}};
  parameters.harmonic_angle_parameters = {{std::acos(-1.0) / 3.0, 2.0}};
  parameters.periodic_dihedral_parameters = {{1.7, 3, 0.4}};

  MolecularForce molecular_force;
  molecular_force.initialize(topology, parameters);
  assert(molecular_force.number_of_harmonic_bonds() == 1);
  assert(molecular_force.number_of_harmonic_angles() == 1);
  assert(molecular_force.number_of_periodic_dihedrals() == 1);

  const std::vector<double> position =
    {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  GPU_Vector<double> gpu_position(position.size());
  gpu_position.copy_from_host(position.data());
  GPU_Vector<double> potential(4, 0.0);
  GPU_Vector<double> force(12, 0.0);
  GPU_Vector<double> virial(36, 0.0);
  molecular_force.compute(make_box(), gpu_position, potential, force, virial);

  const std::vector<double> host_potential = copy_to_host(potential);
  double total_energy = 0.0;
  for (double atom_energy : host_potential) {
    total_energy += atom_energy;
  }
  const double pi = std::acos(-1.0);
  const double expected_angle = (pi / 6.0) * (pi / 6.0);
  const double expected_dihedral = 1.7 * (1.0 + std::cos(3.0 * pi / 2.0 - 0.4));
  assert(nearly_equal(total_energy, expected_angle + expected_dihedral));
}
} // namespace

int main()
{
  test_uninitialized_module_is_no_op();

  if (!gpu_is_available()) {
    std::cout << "SKIP: no accessible GPU for MolecularForce runtime tests.\n";
    return 0;
  }

  test_end_to_end_initialization_compute_accumulation_and_clear();
  test_all_bonded_styles_are_orchestrated_once();
  CHECK(gpuDeviceSynchronize());

  std::cout << "PASS: MolecularForce topology-to-kernel pipeline tests.\n";
  return 0;
}
