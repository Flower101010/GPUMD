#include "model/periodic_dihedral_data.cuh"
#include "utilities/gpu_macro.cuh"
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
bool gpu_is_available()
{
  int number_of_devices = 0;
  return gpuGetDeviceCount(&number_of_devices) == gpuSuccess && number_of_devices > 0;
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

void test_invalid_input_is_rejected_before_upload()
{
  Topology topology;
  topology.number_of_atoms = 4;
  topology.dihedrals = {{0, 1, 2, 3, 1}};
  ForceFieldParameters parameters;
  parameters.periodic_dihedral_parameters = {{1.0, 3, 0.0}};

  PeriodicDihedralData data;
  bool threw = false;
  try {
    data.upload(topology, parameters);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
  assert(data.number_of_dihedrals() == 0);
}

void test_upload_and_clear()
{
  Topology topology;
  topology.number_of_atoms = 5;
  topology.dihedrals = {{0, 1, 2, 3, 1}, {1, 2, 3, 4, 0}};
  ForceFieldParameters parameters;
  parameters.periodic_dihedral_parameters = {{1.0, 3, 0.2}, {2.0, 2, -0.4}};

  PeriodicDihedralData data;
  data.upload(topology, parameters);
  assert(data.number_of_atoms() == 5);
  assert(data.number_of_dihedrals() == 2);
  assert(data.number_of_parameters() == 2);
  assert(copy_to_host(data.atom_i()) == std::vector<int>({0, 1}));
  assert(copy_to_host(data.atom_l()) == std::vector<int>({3, 4}));
  assert(copy_to_host(data.type()) == std::vector<int>({1, 0}));
  assert(copy_to_host(data.force_constant()) == std::vector<double>({1.0, 2.0}));
  assert(copy_to_host(data.multiplicity()) == std::vector<int>({3, 2}));
  assert(copy_to_host(data.phase()) == std::vector<double>({0.2, -0.4}));

  data.clear();
  assert(data.number_of_atoms() == 0);
  assert(data.number_of_dihedrals() == 0);
  assert(data.atom_i().data() == nullptr);
}
} // namespace

int main()
{
  test_invalid_input_is_rejected_before_upload();
  if (!gpu_is_available()) {
    std::cout << "SKIP: no accessible GPU for PeriodicDihedralData upload test.\n";
    return 0;
  }
  test_upload_and_clear();
  std::cout << "PASS: PeriodicDihedralData GPU upload tests.\n";
  return 0;
}
