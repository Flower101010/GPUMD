#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

static bool gpu_is_available()
{
  int number_of_devices = 0;
  return gpuGetDeviceCount(&number_of_devices) == gpuSuccess && number_of_devices > 0;
}

static void test_default_and_zero_size_states()
{
  GPU_Vector<int> default_vector;
  assert(default_vector.size() == 0);
  assert(default_vector.data() == nullptr);

  GPU_Vector<int> sized_zero(0);
  assert(sized_zero.size() == 0);
  assert(sized_zero.data() == nullptr);

  GPU_Vector<int> filled_zero(0, 7);
  assert(filled_zero.size() == 0);
  assert(filled_zero.data() == nullptr);

  default_vector.clear();
  default_vector.clear();
  assert(default_vector.size() == 0);
  assert(default_vector.data() == nullptr);
}

static void test_global_memory_copy_fill_and_resize()
{
  GPU_Vector<int> device(4);
  const std::vector<int> input = {1, 2, 3, 4};
  device.copy_from_host(input.data());

  std::vector<int> output(4);
  device.copy_to_host(output.data());
  assert(output == input);

  device.fill(9);
  device.copy_to_host(output.data());
  assert(output == std::vector<int>({9, 9, 9, 9}));

  const std::vector<int> patch = {5, 6};
  device.copy_from_host(patch.data(), patch.size(), 1);
  device.copy_to_host(output.data());
  assert(output == std::vector<int>({9, 5, 6, 9}));

  std::vector<int> partial(2);
  device.copy_to_host(partial.data(), partial.size(), 1);
  assert(partial == patch);

  device.resize(7, 3);
  output.resize(7);
  device.copy_to_host(output.data());
  assert(output == std::vector<int>({3, 3, 3, 3, 3, 3, 3}));

  int* allocation = device.data();
  device.resize(0);
  assert(device.size() == 0);
  assert(device.data() == allocation);

  device.resize(2);
  assert(device.data() == allocation);
  const std::vector<int> second_input = {10, 11};
  device.copy_from_host(second_input.data());
  output.resize(2);
  device.copy_to_host(output.data());
  assert(output == second_input);

  device.clear();
  device.clear();
  assert(device.size() == 0);
  assert(device.data() == nullptr);
}

static void test_managed_memory()
{
  GPU_Vector<double> managed(3, 2.5, Memory_Type::managed);
  assert(managed.size() == 3);
  assert(managed[0] == 2.5);
  assert(managed[1] == 2.5);
  assert(managed[2] == 2.5);

  managed[1] = 4.5;
  CHECK(gpuDeviceSynchronize());
  assert(managed[1] == 4.5);

  double* allocation = managed.data();
  managed.resize(0, Memory_Type::managed);
  assert(managed.size() == 0);
  assert(managed.data() == allocation);
  managed.resize(2, Memory_Type::managed);
  assert(managed.data() == allocation);
  assert(managed[1] == 4.5);
  managed.clear();
  assert(managed.data() == nullptr);
}

static void test_repeated_allocation_and_release()
{
  GPU_Vector<double> device;
  for (int iteration = 0; iteration < 1000; ++iteration) {
    const size_t size = static_cast<size_t>(iteration % 31 + 1);
    device.resize(size, static_cast<double>(iteration));
    device.clear();
    assert(device.size() == 0);
    assert(device.data() == nullptr);
  }
}

int main(int argc, char** argv)
{
  // An availability probe must not depend on the container regression tests.
  if (argc == 2 && std::strcmp(argv[1], "--probe") == 0) {
    std::cout << (gpu_is_available() ? "PASS: GPU is accessible.\n" :
      "SKIP: no accessible GPU.\n");
    return 0;
  }
  if (argc != 1) {
    std::cerr << "Usage: test_gpu_vector [--probe]\n";
    return 2;
  }
  test_default_and_zero_size_states();

  if (!gpu_is_available()) {
    std::cout << "SKIP: no accessible GPU for GPU_Vector runtime tests.\n";
    return 0;
  }

  test_global_memory_copy_fill_and_resize();
  test_managed_memory();
  test_repeated_allocation_and_release();
  CHECK(gpuDeviceSynchronize());

  std::cout << "PASS: GPU_Vector runtime tests.\n";
  return 0;
}
