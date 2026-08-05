#include "model/harmonic_angle_data.cuh"
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
  topology.number_of_atoms = 3;
  topology.angles = {{0, 1, 2, 1}};
  ForceFieldParameters parameters;
  parameters.harmonic_angle_parameters = {{1.0, 2.0}};

  HarmonicAngleData data;
  bool threw = false;
  try {
    data.upload(topology, parameters);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
  assert(data.number_of_angles() == 0);
}

void test_upload_clear_and_empty_reupload()
{
  Topology topology;
  topology.number_of_atoms = 4;
  topology.angles = {{0, 1, 2, 1}, {1, 2, 3, 0}};
  ForceFieldParameters parameters;
  parameters.harmonic_angle_parameters = {{1.0, 2.0}, {1.5, 3.0}};

  HarmonicAngleData data;
  data.upload(topology, parameters);
  assert(data.number_of_atoms() == 4);
  assert(data.number_of_angles() == 2);
  assert(data.number_of_parameters() == 2);
  assert(copy_to_host(data.atom_i()) == std::vector<int>({0, 1}));
  assert(copy_to_host(data.atom_j()) == std::vector<int>({1, 2}));
  assert(copy_to_host(data.atom_k()) == std::vector<int>({2, 3}));
  assert(copy_to_host(data.type()) == std::vector<int>({1, 0}));
  assert(copy_to_host(data.equilibrium_angle()) == std::vector<double>({1.0, 1.5}));
  assert(copy_to_host(data.angle_constant()) == std::vector<double>({2.0, 3.0}));

  topology.angles.clear();
  parameters.harmonic_angle_parameters.clear();
  data.upload(topology, parameters);
  assert(data.number_of_atoms() == 4);
  assert(data.number_of_angles() == 0);
  assert(data.number_of_parameters() == 0);

  data.clear();
  assert(data.number_of_atoms() == 0);
  assert(data.atom_i().data() == nullptr);
}
} // namespace

int main()
{
  test_invalid_input_is_rejected_before_upload();
  if (!gpu_is_available()) {
    std::cout << "SKIP: no accessible GPU for HarmonicAngleData upload test.\n";
    return 0;
  }
  test_upload_clear_and_empty_reupload();
  std::cout << "PASS: HarmonicAngleData GPU upload tests.\n";
  return 0;
}
