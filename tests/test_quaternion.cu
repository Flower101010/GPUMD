#include "utilities/math.cuh"
#include <cassert>
#include <cmath>

using namespace rigid_math;

static void expect_quaternion(Quaternion actual, Quaternion expected)
{
  const double tol = 1e-12;
  assert(std::abs(actual.w - expected.w) <= tol);
  assert(std::abs(actual.x - expected.x) <= tol);
  assert(std::abs(actual.y - expected.y) <= tol);
  assert(std::abs(actual.z - expected.z) <= tol);
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
  test_multiply_and_conjugate();
  test_rotate_axis_known_solution();
  test_integrate_free_rotation_reversible();
  test_permute();
  return 0;
}
