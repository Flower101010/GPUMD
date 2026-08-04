#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include <cassert>
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

  device.resize(0);
  assert(device.size() == 0);
  assert(device.data() == nullptr);

  device.resize(2);
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

  managed.resize(0, Memory_Type::managed);
  assert(managed.size() == 0);
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

int main()
{
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
