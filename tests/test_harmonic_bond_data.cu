#include "model/harmonic_bond_data.cuh"
#include "utilities/gpu_macro.cuh"
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

static bool gpu_is_available()
{
  int number_of_devices = 0;
  return gpuGetDeviceCount(&number_of_devices) == gpuSuccess && number_of_devices > 0;
}

static void test_empty_gpu_vector_is_safe_without_a_device()
{
  GPU_Vector<int> empty;
  empty.resize(0);

  assert(empty.size() == 0);
  assert(empty.data() == nullptr);
  empty.clear();
}

static void test_invalid_input_is_rejected_before_upload()
{
  Topology topology;
  topology.number_of_atoms = 2;
  topology.bonds = {{0, 1, 1}};

  ForceFieldParameters parameters;
  parameters.harmonic_bond_parameters = {{1.5, 20.0}};

  HarmonicBondData data;
  bool threw = false;
  try {
    data.upload(topology, parameters);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
  assert(data.number_of_bonds() == 0);
}

template <typename T>
static std::vector<T> copy_to_host(const GPU_Vector<T>& device)
{
  std::vector<T> host(device.size());
  if (!host.empty()) {
    device.copy_to_host(host.data());
  }
  return host;
}

static void test_upload_and_clear()
{
  Topology topology;
  topology.number_of_atoms = 3;
  topology.bonds = {{0, 1, 1}, {1, 2, 0}};

  ForceFieldParameters parameters;
  parameters.harmonic_bond_parameters = {{1.5, 20.0}, {1.3, 30.0}};

  HarmonicBondData data;
  data.upload(topology, parameters);

  assert(data.number_of_bonds() == 2);
  assert(data.number_of_parameters() == 2);
  assert(copy_to_host(data.atom_i()) == std::vector<int>({0, 1}));
  assert(copy_to_host(data.atom_j()) == std::vector<int>({1, 2}));
  assert(copy_to_host(data.type()) == std::vector<int>({1, 0}));
  assert(copy_to_host(data.equilibrium_distance()) == std::vector<double>({1.5, 1.3}));
  assert(copy_to_host(data.force_constant()) == std::vector<double>({20.0, 30.0}));

  data.clear();
  assert(data.number_of_bonds() == 0);
  assert(data.number_of_parameters() == 0);
  assert(data.atom_i().data() == nullptr);
  assert(data.equilibrium_distance().data() == nullptr);
}

static void test_reupload_empty_data()
{
  Topology topology;
  topology.number_of_atoms = 2;
  topology.bonds = {{0, 1, 0}};

  ForceFieldParameters parameters;
  parameters.harmonic_bond_parameters = {{1.5, 20.0}};

  HarmonicBondData data;
  data.upload(topology, parameters);

  topology.bonds.clear();
  parameters.harmonic_bond_parameters.clear();
  data.upload(topology, parameters);

  assert(data.number_of_bonds() == 0);
  assert(data.number_of_parameters() == 0);
  assert(data.atom_i().data() == nullptr);
  assert(data.force_constant().data() == nullptr);
}

int main()
{
  test_empty_gpu_vector_is_safe_without_a_device();
  test_invalid_input_is_rejected_before_upload();

  if (!gpu_is_available()) {
    std::cout << "SKIP: no accessible GPU for HarmonicBondData upload test.\n";
    return 0;
  }

  test_upload_and_clear();
  test_reupload_empty_data();
  return 0;
}
