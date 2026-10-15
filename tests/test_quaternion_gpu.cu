#include "utilities/rigid_math.cuh"
#include "utilities/gpu_vector.cuh"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace rigid_math;

struct Result {
  Quaternion q, p, product, conjugated, permuted;
  Vec3 sum, difference, scaled, crossed, unit, rotated, inverse, momentum, omega, singular_omega, kicked;
  Quaternion unit_q, converted;
  double vector_dot, vector_squared, quaternion_squared, vector_norm, quaternion_norm, stable_large, stable_small;
  bool normalized, normalized_q, invalid_unchanged;
};

// 故意在 rigid_math.cu 之外调用设备函数，防止 CPU 测试通过但设备链接失败。
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
  Result& result = output[i];
  const Vec3 v{1, 2, 3}, b{-2, 4, 1};
  result.sum = add(v, b);
  result.difference = subtract(v, b);
  result.scaled = scale(v, -2);
  result.crossed = cross(v, b);
  result.vector_dot = dot(v, b);
  result.vector_squared = norm_squared(v);
  result.quaternion_squared = norm_squared(Quaternion{1, 2, 2, 4});
  result.vector_norm = norm(Vec3{3, 4, 0});
  result.quaternion_norm = norm(Quaternion{1, 2, 2, 4});
  result.stable_large = norm(Vec3{1e300, 1e300, 0}) / 1e300;
  result.stable_small = norm(Quaternion{1e-300, 1e-300, 0, 0}) / 1e-300;
  const double magnitude = i == 0 ? DBL_MAX : i == 1 ? 0x0.0000000000001p-1022 : 1e-300;
  result.unit = {magnitude, -magnitude, 0};
  result.unit_q = {magnitude, 0, -magnitude, 0};
  result.normalized = normalize(result.unit);
  result.normalized_q = normalize(result.unit_q);
  Vec3 zero{0, 0, 0};
  Quaternion invalid{1, 2, INFINITY, 4};
  result.invalid_unchanged = !normalize(zero) && !normalize(invalid) &&
    zero.x == 0 && zero.y == 0 && zero.z == 0 &&
    invalid.w == 1 && invalid.x == 2 && invalid.y == INFINITY && invalid.z == 4;
  const double h = sqrt(0.5);
  const Quaternion orientation{h, 0, 0, h};
  result.rotated = rotate(orientation, v);
  result.inverse = inverse_rotate(orientation, {-2, 1, 3});
  result.converted = angular_momentum_to_conjqm(orientation, v);
  result.momentum = conjqm_to_angular_momentum(orientation, result.converted);
  result.omega = angular_momentum_to_omega(orientation, v, {2, 4, 6});
  result.singular_omega = angular_momentum_to_omega(orientation, v, {0, 4, 6});
  Quaternion kicked = result.converted;
  kick_rotation(orientation, kicked, {4, -2, 1}, 0.25);
  result.kicked = conjqm_to_angular_momentum(orientation, kicked);
}

static void expect(Quaternion actual, Quaternion expected)
{
  assert(std::abs(actual.w - expected.w) <= 1e-12);
  assert(std::abs(actual.x - expected.x) <= 1e-12);
  assert(std::abs(actual.y - expected.y) <= 1e-12);
  assert(std::abs(actual.z - expected.z) <= 1e-12);
}

static void expect(Vec3 actual, Vec3 expected)
{
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
    const Result& r = result[i];
    const double h = std::sqrt(0.5);
    expect(r.sum, Vec3{-1, 6, 4});
    expect(r.difference, Vec3{3, -2, 2});
    expect(r.scaled, Vec3{-2, -4, -6});
    expect(r.crossed, Vec3{-10, -7, 8});
    assert(r.vector_dot == 9 && r.vector_squared == 14 && r.quaternion_squared == 25);
    assert(r.vector_norm == 5 && r.quaternion_norm == 5);
    assert(std::abs(r.stable_large - std::sqrt(2.0)) <= 1e-12);
    assert(std::abs(r.stable_small - std::sqrt(2.0)) <= 1e-12);
    assert(r.normalized && r.normalized_q && r.invalid_unchanged);
    expect(r.unit, Vec3{h, -h, 0});
    expect(r.unit_q, {h, 0, -h, 0});
    expect(r.rotated, Vec3{-2, 1, 3});
    expect(r.inverse, Vec3{1, 2, 3});
    expect(r.converted, {-6*h, 6*h, 2*h, 6*h});
    expect(r.momentum, Vec3{1, 2, 3});
    expect(r.omega, Vec3{0.25, 1, 0.5});
    expect(r.singular_omega, Vec3{0.25, 0, 0.5});
    expect(r.kicked, Vec3{2, 1.5, 3.25});
  }
}
