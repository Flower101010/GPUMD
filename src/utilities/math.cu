#include "math.cuh"
#include <cmath>

namespace rigid_math
{
/* ----------------------------------------------------------------------
   quaternion-quaternion multiply: c = a*b
------------------------------------------------------------------------- */

__host__ __device__ Quaternion multiply(const Quaternion& a, const Quaternion& b)
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

__host__ __device__ Quaternion conjugate(const Quaternion& q) { return {q.w, -q.x, -q.y, -q.z}; };

/* ----------------------------------------------------------------------
   apply evolution operators to quat, quat momentum
   Miller et al., J Chem Phys. 116, 8649-8659 (2002)
------------------------------------------------------------------------- */

// 1. Miller permutation operator
__host__ __device__ Quaternion permute(const Quaternion& q, int k)
{
  switch (k) {
    case 1:
      return {-q.x, q.w, q.z, -q.y};

    case 2:
      return {-q.y, -q.z, q.w, q.x};

    case 3:
      return {-q.z, q.y, -q.x, q.w};

    default:
      return {0.0, 0.0, 0.0, 0.0};
  }
}
// ----------------------------------------------------------------------
// Quaternion dot product
// ----------------------------------------------------------------------

__host__ __device__ double dot(const Quaternion& a, const Quaternion& b)
{
  return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

// ----------------------------------------------------------------------
// Miller single-axis exact evolution
// q: orientation quaternion
// p: canonical conjugate momentum
// inertia: principal moments of inertia
// k: principal axis (1, 2, or 3)
// dt: evolution time
// ----------------------------------------------------------------------

__host__ __device__ void
rotate_axis(Quaternion& q, Quaternion& p, const Vec3& inertia, int k, double dt)
{
  if (k < 1 || k > 3)
    return;

  // Principal moment of inertia
  const double I = (k == 1) ? inertia.x : (k == 2) ? inertia.y : inertia.z;

  // Match LAMMPS handling of zero inertia
  if (I == 0.0)
    return;

  // Apply Miller permutation operators
  const Quaternion kq = permute(q, k);
  const Quaternion kp = permute(p, k);

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

__host__ __device__ void
integrate_rotation(Quaternion& q, Quaternion& p, const Vec3& inertia, double dt)
{
  const double half_dt = 0.5 * dt;

  rotate_axis(q, p, inertia, 3, half_dt);
  rotate_axis(q, p, inertia, 2, half_dt);
  rotate_axis(q, p, inertia, 1, dt);
  rotate_axis(q, p, inertia, 2, half_dt);
  rotate_axis(q, p, inertia, 3, half_dt);
}
} // namespace rigid_math