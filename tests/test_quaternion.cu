#include "utilities/rigid_math.cuh"
#include <cassert>
#include <cmath>
#include <limits>
#include <initializer_list>

using namespace rigid_math;

static void expect_quaternion(Quaternion actual, Quaternion expected)
{
  const double tol = 1e-12;
  assert(std::abs(actual.w - expected.w) <= tol);
  assert(std::abs(actual.x - expected.x) <= tol);
  assert(std::abs(actual.y - expected.y) <= tol);
  assert(std::abs(actual.z - expected.z) <= tol);
}

static void expect_vector(Vec3 actual, Vec3 expected)
{
  assert(std::abs(actual.x - expected.x) <= 1e-12);
  assert(std::abs(actual.y - expected.y) <= 1e-12);
  assert(std::abs(actual.z - expected.z) <= 1e-12);
}

static void test_vector_algebra_and_norms()
{
  const Vec3 a{1, 2, 3}, b{-2, 4, 1};
  expect_vector(add(a, b), {-1, 6, 4});
  expect_vector(subtract(a, b), {3, -2, 2});
  expect_vector(scale(a, -2), {-2, -4, -6});
  expect_vector(cross(a, b), {-10, -7, 8});
  assert(dot(a, b) == 9);
  assert(norm_squared(a) == 14);
  assert(norm(Vec3{3, 4, 0}) == 5);
  assert(norm_squared(Quaternion{1, 2, 2, 4}) == 25);
  assert(norm(Quaternion{1, 2, 2, 4}) == 5);
  // 范数不能先平方：这些输入的范数可表示，但平方会溢出/下溢。
  for (double magnitude : {1e300, 1e-300}) {
    assert(std::abs(norm(Vec3{magnitude, magnitude, 0}) / magnitude - std::sqrt(2.0)) < 1e-14);
    assert(std::abs(norm(Quaternion{magnitude, magnitude, 0, 0}) / magnitude - std::sqrt(2.0)) < 1e-14);
  }
}

static void test_checked_normalization()
{
  Vec3 v{3, 4, 0};
  assert(normalize(v));
  expect_vector(v, {0.6, 0.8, 0});
  Quaternion q{1, 2, 2, 4};
  assert(normalize(q));
  expect_quaternion(q, {0.2, 0.4, 0.4, 0.8});
  const double h = std::sqrt(0.5);
  // 真正范数溢出及次正规输入仍能归一化。
  for (double magnitude : {std::numeric_limits<double>::max(), std::numeric_limits<double>::denorm_min()}) {
    Vec3 huge{magnitude, -magnitude, 0};
    Quaternion tiny{magnitude, 0, -magnitude, 0};
    assert(normalize(huge));
    assert(normalize(tiny));
    expect_vector(huge, {h, -h, 0});
    expect_quaternion(tiny, {h, 0, -h, 0});
  }
  Vec3 zero{0, 0, 0};
  Quaternion zero_q{0, 0, 0, 0};
  assert(!normalize(zero));
  assert(!normalize(zero_q));
  expect_vector(zero, {0, 0, 0});
  expect_quaternion(zero_q, {0, 0, 0, 0});
  Vec3 bad{1, std::numeric_limits<double>::infinity(), 3};
  Quaternion bad_q{1, 2, std::numeric_limits<double>::quiet_NaN(), 4};
  assert(!normalize(bad));
  assert(!normalize(bad_q));
  assert(bad.x == 1 && std::isinf(bad.y) && bad.z == 3);
  assert(bad_q.w == 1 && bad_q.x == 2 && std::isnan(bad_q.y) && bad_q.z == 4);
}

static void test_rotations_and_momentum_conventions()
{
  const double h = std::sqrt(0.5);
  const Quaternion q{h, 0, 0, h}; // +90 degrees about z.
  const Vec3 v{1, 2, 3};
  expect_vector(rotate(q, v), {-2, 1, 3});
  expect_vector(rotate({-h, 0, 0, -h}, v), {-2, 1, 3});
  expect_vector(inverse_rotate(q, {-2, 1, 3}), v);
  expect_vector(rotate({0.5, 0.5, 0.5, 0.5}, v), {3, 1, 2});
  // 独立手算非平凡方向，检查空间/体坐标和因子 2，而非仅做往返。
  const Quaternion p = angular_momentum_to_conjqm(q, v);
  expect_quaternion(p, {-6*h, 6*h, 2*h, 6*h});
  expect_vector(conjqm_to_angular_momentum(q, p), v);
  assert(std::abs(dot(q, p)) <= 1e-12);
  expect_vector(angular_momentum_to_omega(q, v, {2, 4, 6}), {0.25, 1, 0.5});
  expect_vector(angular_momentum_to_omega(q, v, {0, 4, 6}), {0.25, 0, 0.5});
  expect_vector(angular_momentum_to_omega(q, v, {0, 0, 0}), {0, 0, 0});
  Quaternion kicked = p;
  kick_rotation(q, kicked, {4, -2, 1}, 0.25);
  expect_vector(conjqm_to_angular_momentum(q, kicked), {2, 1.5, 3.25});
  assert(std::abs(dot(q, kicked)) <= 1e-12);
  kick_rotation(q, kicked, {4, -2, 1}, -0.25);
  expect_quaternion(kicked, p);
  kick_rotation(q, kicked, {0, 0, 0}, 0.25);
  expect_quaternion(kicked, p);
}

static void test_miller_with_constant_axial_torque()
{
  Quaternion q{1, 0, 0, 0}, p{0, 0, 0, 1.4};
  const Vec3 torque{0, 0, 0.4}, inertia{2, 2, 2};
  const double dt = 0.01;
  for (int step = 0; step < 100; ++step) {
    kick_rotation(q, p, torque, dt / 2);
    integrate_free_rotation(q, p, inertia, dt);
    kick_rotation(q, p, torque, dt / 2);
  }
  // t=1, Lz=0.7+0.4*t；theta=(0.7*t+0.4*t*t/2)/2=0.45。
  expect_quaternion(q, {std::cos(0.225), 0, 0, std::sin(0.225)});
  expect_vector(conjqm_to_angular_momentum(q, p), {0, 0, 1.1});
  assert(std::abs(dot(q, q) - 1) <= 1e-12);
  assert(std::abs(dot(q, p)) <= 1e-12);
}

static void test_multiply_and_conjugate()
{
  const Quaternion q{1, 2, 3, 4};
  const Quaternion identity{1, 0, 0, 0};
  expect_quaternion(multiply(identity, q), q);
  expect_quaternion(multiply(q, identity), q);
  expect_quaternion(conjugate(q), {1, -2, -3, -4});
  expect_quaternion(multiply(q, conjugate(q)), {30, 0, 0, 0});

  const Quaternion i{0, 1, 0, 0}, j{0, 0, 1, 0};
  expect_quaternion(multiply(i, j), {0, 0, 0, 1});
  expect_quaternion(multiply(j, i), {0, 0, 0, -1});
}

static void test_permute()
{
  const Quaternion q{1, 2, 3, 4};
  expect_quaternion(permute(q, PrincipalAxis::x), {-2, 1, 4, -3});
  expect_quaternion(permute(q, PrincipalAxis::y), {-3, -4, 1, 2});
  expect_quaternion(permute(q, PrincipalAxis::z), {-4, 3, -2, 1});
}

static void test_rotate_axis_known_solution()
{
  Quaternion q{1, 0, 0, 0}, p{0, 0, 0, 4};
  const Vec3 inertia{1, 1, 2};
  const double dt = 0.2;
  // Rotation about axis 3: phi = 4 / (4 * 2) = 0.5.
  const double c = std::cos(0.1), s = std::sin(0.1);
  rotate_axis(q, p, inertia, PrincipalAxis::z, dt);
  expect_quaternion(q, {c, 0, 0, s});
  expect_quaternion(p, {-4 * s, 0, 0, 4 * c});
}

static void test_integrate_free_rotation_reversible()
{
  const Quaternion q0{1, 0, 0, 0}, p0{0, 1, 2, 3};
  Quaternion q = q0, p = p0;
  const Vec3 inertia{1, 2, 3};
  integrate_free_rotation(q, p, inertia, 0.1);
  assert(std::abs(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z - 1) <= 1e-12);
  assert(std::abs(q.w * p.w + q.x * p.x + q.y * p.y + q.z * p.z) <= 1e-12);
  integrate_free_rotation(q, p, inertia, -0.1);
  expect_quaternion(q, q0);
  expect_quaternion(p, p0);
}

int main()
{
  test_vector_algebra_and_norms();
  test_checked_normalization();
  test_rotations_and_momentum_conventions();
  test_miller_with_constant_axial_torque();
  test_multiply_and_conjugate();
  test_rotate_axis_known_solution();
  test_integrate_free_rotation_reversible();
  test_permute();
  return 0;
}
