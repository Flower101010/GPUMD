#include "utilities/math.cuh"
#include "utilities/gpu_vector.cuh"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace rigid_math;

struct Result {
  Quaternion q, p, product, conjugated, permuted;
};

// 故意在 math.cu 之外调用设备函数，防止 CPU 测试通过但设备链接失败。
static __global__ void evaluate(Result* output)
{
  const int i = threadIdx.x;
  if (i >= 3) return;
  const PrincipalAxis axis = static_cast<PrincipalAxis>(i + 1);
  Quaternion q{1, 0, 0, 0}, p{0, 0, 0, 0};
  if (i == 0) p.x = 4;
  if (i == 1) p.y = 4;
  if (i == 2) p.z = 4;
  integrate_free_rotation(q, p, {2, 2, 2}, 0.2);
  output[i] = {q, p, multiply({0, 1, 0, 0}, {0, 0, 1, 0}),
    conjugate({1, 2, 3, 4}), permute({1, 2, 3, 4}, axis)};
}

static void expect(Quaternion actual, Quaternion expected)
{
  assert(std::abs(actual.w - expected.w) <= 1e-12);
  assert(std::abs(actual.x - expected.x) <= 1e-12);
  assert(std::abs(actual.y - expected.y) <= 1e-12);
  assert(std::abs(actual.z - expected.z) <= 1e-12);
}

int main()
{
  int devices = 0;
  if (gpuGetDeviceCount(&devices) != gpuSuccess || devices == 0) {
    std::cout << "SKIP: no accessible GPU\n";
    return 0;
  }
  GPU_Vector<Result> device(3);
  evaluate<<<1, 32>>>(device.data());
  GPU_CHECK_KERNEL
  Result result[3];
  device.copy_to_host(result);
  const double c = std::cos(0.1), s = std::sin(0.1);
  const Quaternion permutations[3] = {{-2, 1, 4, -3}, {-3, -4, 1, 2}, {-4, 3, -2, 1}};
  for (int i = 0; i < 3; ++i) {
    Quaternion q{c, 0, 0, 0}, p{-4*s, 0, 0, 0};
    if (i == 0) { q.x = s; p.x = 4*c; }
    if (i == 1) { q.y = s; p.y = 4*c; }
    if (i == 2) { q.z = s; p.z = 4*c; }
    expect(result[i].q, q);
    expect(result[i].p, p);
    expect(result[i].product, {0, 0, 0, 1});
    expect(result[i].conjugated, {1, -2, -3, -4});
    expect(result[i].permuted, permutations[i]);
    assert(std::abs(dot(result[i].q, result[i].q) - 1) <= 1e-12);
    assert(std::abs(dot(result[i].q, result[i].p)) <= 1e-12);
  }
}
