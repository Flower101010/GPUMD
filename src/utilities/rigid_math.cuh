#pragma once

#include <cassert>
#include <cfloat>
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define RIGID_MATH_HD __host__ __device__
#else
#define RIGID_MATH_HD
#endif

namespace rigid_math
{

struct Vec3 {
  double x, y, z;
};
struct Quaternion {
  double w, x, y, z;
};
enum class PrincipalAxis { x = 1, y = 2, z = 3 };

RIGID_MATH_HD inline Vec3 add(const Vec3& a, const Vec3& b);
RIGID_MATH_HD inline Vec3 subtract(const Vec3& a, const Vec3& b);
RIGID_MATH_HD inline Vec3 scale(const Vec3& v, double factor);
RIGID_MATH_HD inline double dot(const Vec3& a, const Vec3& b);
RIGID_MATH_HD inline Vec3 cross(const Vec3& a, const Vec3& b);
RIGID_MATH_HD inline double norm_squared(const Vec3& v);
RIGID_MATH_HD inline double norm_squared(const Quaternion& q);
RIGID_MATH_HD inline double norm(const Vec3& v);
RIGID_MATH_HD inline double norm(const Quaternion& q);

RIGID_MATH_HD inline bool normalize(Vec3& v);
RIGID_MATH_HD inline bool normalize(Quaternion& q);

RIGID_MATH_HD inline Vec3 rotate(const Quaternion& q, const Vec3& v_body);
RIGID_MATH_HD inline Vec3 inverse_rotate(const Quaternion& q, const Vec3& v_space);

RIGID_MATH_HD inline Quaternion
angular_momentum_to_conjqm(const Quaternion& q, const Vec3& angular_momentum_space);

RIGID_MATH_HD inline Vec3 conjqm_to_angular_momentum(const Quaternion& q, const Quaternion& p);

RIGID_MATH_HD inline Vec3 angular_momentum_to_omega(
  const Quaternion& q, const Vec3& angular_momentum_space, const Vec3& inertia);

RIGID_MATH_HD inline void
kick_rotation(const Quaternion& q, Quaternion& p, const Vec3& torque_space, double dt);
} // namespace rigid_math

namespace rigid_math
{

/* ----------------------------------------------------------------------
   quaternion-quaternion multiply: c = a*b
------------------------------------------------------------------------- */

RIGID_MATH_HD inline Quaternion multiply(const Quaternion& a, const Quaternion& b)
{
  return {
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
    a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x};
};

/* ----------------------------------------------------------------------
   conjugate of a quaternion
------------------------------------------------------------------------- */

RIGID_MATH_HD inline Quaternion conjugate(const Quaternion& q) { return {q.w, -q.x, -q.y, -q.z}; };

/* ----------------------------------------------------------------------
   apply evolution operators to quat, quat momentum
   Miller et al., J Chem Phys. 116, 8649-8659 (2002)
------------------------------------------------------------------------- */

// 1. Miller permutation operator
RIGID_MATH_HD inline Quaternion permute(const Quaternion& q, PrincipalAxis axis)
{
  switch (axis) {
    case PrincipalAxis::x:
      return {-q.x, q.w, q.z, -q.y};

    case PrincipalAxis::y:
      return {-q.y, -q.z, q.w, q.x};

    case PrincipalAxis::z:
      return {-q.z, q.y, -q.x, q.w};

    default:
      assert(false && "Invalid principal axis");
      return {0.0, 0.0, 0.0, 0.0};
  }
}
// ----------------------------------------------------------------------
// Quaternion dot product
// ----------------------------------------------------------------------

RIGID_MATH_HD inline double dot(const Quaternion& a, const Quaternion& b)
{
  return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

RIGID_MATH_HD inline Vec3 add(const Vec3& a, const Vec3& b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

RIGID_MATH_HD inline Vec3 subtract(const Vec3& a, const Vec3& b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

RIGID_MATH_HD inline Vec3 scale(const Vec3& v, double factor)
{
  return {factor * v.x, factor * v.y, factor * v.z};
}

RIGID_MATH_HD inline double dot(const Vec3& a, const Vec3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

RIGID_MATH_HD inline Vec3 cross(const Vec3& a, const Vec3& b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// 真正平方范数超出 double 范围时可溢出；norm 不先计算平方范数。
RIGID_MATH_HD inline double norm_squared(const Vec3& v) { return dot(v, v); }
RIGID_MATH_HD inline double norm_squared(const Quaternion& q) { return dot(q, q); }
RIGID_MATH_HD inline double norm(const Vec3& v) { return hypot(hypot(v.x, v.y), v.z); }
RIGID_MATH_HD inline double norm(const Quaternion& q)
{
  return hypot(hypot(q.w, q.x), hypot(q.y, q.z));
}

namespace detail
{
RIGID_MATH_HD inline bool finite(double value) { return value >= -DBL_MAX && value <= DBL_MAX; }
} // namespace detail

RIGID_MATH_HD inline bool normalize(Vec3& v)
{
  if (!detail::finite(v.x) || !detail::finite(v.y) || !detail::finite(v.z))
    return false;
  const double largest = fmax(fabs(v.x), fmax(fabs(v.y), fabs(v.z)));
  if (largest == 0.0)
    return false;
  // 先缩放分量，避免大数范数溢出或次正规数的倒数溢出。
  const Vec3 scaled{v.x / largest, v.y / largest, v.z / largest};
  const double length = norm(scaled);
  v = {scaled.x / length, scaled.y / length, scaled.z / length};
  return true;
}

RIGID_MATH_HD inline bool normalize(Quaternion& q)
{
  if (!detail::finite(q.w) || !detail::finite(q.x) || !detail::finite(q.y) || !detail::finite(q.z))
    return false;
  const double largest = fmax(fmax(fabs(q.w), fabs(q.x)), fmax(fabs(q.y), fabs(q.z)));
  if (largest == 0.0)
    return false;
  const Quaternion scaled{q.w / largest, q.x / largest, q.y / largest, q.z / largest};
  const double length = norm(scaled);
  q = {scaled.w / length, scaled.x / length, scaled.y / length, scaled.z / length};
  return true;
}

RIGID_MATH_HD inline Vec3 rotate(const Quaternion& q, const Vec3& v_body)
{//没有对四元数进行归一化检查，所以必须传入单位四元数
  const Vec3 imaginary{q.x, q.y, q.z};
  const Vec3 t = scale(cross(imaginary, v_body), 2.0);
  return add(v_body, add(scale(t, q.w), cross(imaginary, t)));
}

RIGID_MATH_HD inline Vec3 inverse_rotate(const Quaternion& q, const Vec3& v_space)
{
  return rotate(conjugate(q), v_space);
}

RIGID_MATH_HD inline Quaternion
angular_momentum_to_conjqm(const Quaternion& q, const Vec3& angular_momentum_space)
{
  const Vec3 body = inverse_rotate(q, angular_momentum_space);
  const Quaternion product = multiply(q, {0.0, body.x, body.y, body.z});
  return {2.0 * product.w, 2.0 * product.x, 2.0 * product.y, 2.0 * product.z};
}

RIGID_MATH_HD inline Vec3 conjqm_to_angular_momentum(const Quaternion& q, const Quaternion& p)
{
  const Quaternion body = multiply(conjugate(q), p);
  return rotate(q, {0.5 * body.x, 0.5 * body.y, 0.5 * body.z});
}

RIGID_MATH_HD inline Vec3 angular_momentum_to_omega(
  const Quaternion& q, const Vec3& angular_momentum_space, const Vec3& inertia)
{
  assert(detail::finite(inertia.x) && inertia.x >= 0.0);
  assert(detail::finite(inertia.y) && inertia.y >= 0.0);
  assert(detail::finite(inertia.z) && inertia.z >= 0.0);
  const Vec3 body = inverse_rotate(q, angular_momentum_space);
  const Vec3 omega_body{
    inertia.x == 0.0 ? 0.0 : body.x / inertia.x,
    inertia.y == 0.0 ? 0.0 : body.y / inertia.y,
    inertia.z == 0.0 ? 0.0 : body.z / inertia.z};
  return rotate(q, omega_body);
}

RIGID_MATH_HD inline void
kick_rotation(const Quaternion& q, Quaternion& p, const Vec3& torque_space, double dt)
{
  const Quaternion rate = angular_momentum_to_conjqm(q, torque_space);
  p = {p.w + dt * rate.w, p.x + dt * rate.x, p.y + dt * rate.y, p.z + dt * rate.z};
}

// ----------------------------------------------------------------------
// Miller single-axis exact evolution
// q: orientation quaternion
// p: canonical conjugate momentum
// inertia: principal moments of inertia
// axis: principal axis (x, y, or z)
// dt: evolution time
// ----------------------------------------------------------------------

RIGID_MATH_HD inline void
rotate_axis(Quaternion& q, Quaternion& p, const Vec3& inertia, PrincipalAxis axis, double dt)
{
  assert(axis == PrincipalAxis::x || axis == PrincipalAxis::y || axis == PrincipalAxis::z);

  // Principal moment of inertia
  const double I = (axis == PrincipalAxis::x)   ? inertia.x
                   : (axis == PrincipalAxis::y) ? inertia.y
                                                : inertia.z;
  assert(I >= 0.0 && I <= DBL_MAX);

  // Match LAMMPS handling of zero inertia
  if (I == 0.0)
    return;

  // Apply Miller permutation operators
  const Quaternion kq = permute(q, axis);
  const Quaternion kp = permute(p, axis);

  // Angular frequency
  const double phi = dot(p, kq) / (4.0 * I);

  // Exact evolution
  const double c = cos(dt * phi);
  const double s = sin(dt * phi);

  // Update orientation quaternion
  q = {c * q.w + s * kq.w, c * q.x + s * kq.x, c * q.y + s * kq.y, c * q.z + s * kq.z};

  // Update canonical conjugate momentum
  p = {c * p.w + s * kp.w, c * p.x + s * kp.x, c * p.y + s * kp.y, c * p.z + s * kp.z};
}

// ----------------------------------------------------------------------
// Miller symmetric five-stage splitting
// Sequence: 3-2-1-2-3
// ----------------------------------------------------------------------

RIGID_MATH_HD inline void
integrate_free_rotation(Quaternion& q, Quaternion& p, const Vec3& inertia, double dt)
{
  const double half_dt = 0.5 * dt;

  rotate_axis(q, p, inertia, PrincipalAxis::z, half_dt);
  rotate_axis(q, p, inertia, PrincipalAxis::y, half_dt);
  rotate_axis(q, p, inertia, PrincipalAxis::x, dt);
  rotate_axis(q, p, inertia, PrincipalAxis::y, half_dt);
  rotate_axis(q, p, inertia, PrincipalAxis::z, half_dt);
}
} // namespace rigid_math

#undef RIGID_MATH_HD
