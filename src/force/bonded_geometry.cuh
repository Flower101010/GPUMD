/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once

#include <cfloat>
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define BONDED_HD __host__ __device__
#else
#define BONDED_HD
#endif

namespace bonded_geometry
{
constexpr double pi = 3.14159265358979323846;
BONDED_HD inline bool finite(double x) { return x >= -DBL_MAX && x <= DBL_MAX; }
BONDED_HD inline bool finite_vector(const double x[3])
{
  return finite(x[0]) && finite(x[1]) && finite(x[2]);
}
BONDED_HD inline double dot(const double a[3], const double b[3])
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
BONDED_HD inline void cross(const double a[3], const double b[3], double result[3])
{
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];
}
BONDED_HD inline double length(const double a[3]) { return hypot(hypot(a[0], a[1]), a[2]); }
BONDED_HD inline bool normalize(const double a[3], double unit[3], double& norm)
{
  if (!finite_vector(a))
    return false;
  norm = length(a);
  if (!finite(norm) || norm == 0.0)
    return false;
  for (int d = 0; d < 3; ++d)
    unit[d] = a[d] / norm;
  return true;
}

// r = r_j-r_i; all evaluators return false for undefined/nonfinite geometry.
BONDED_HD inline bool
harmonic_bond(const double r[3], double r0, double k, double& energy, double fi[3], double fj[3])
{
  energy = 0.0;
  for (int d = 0; d < 3; ++d)
    fi[d] = fj[d] = 0.0;
  if (k == 0.0)
    return true;
  double unit[3], norm;
  if (!normalize(r, unit, norm))
    return false;
  const double delta = norm - r0;
  energy = 0.5 * k * delta * delta;
  for (int d = 0; d < 3; ++d) {
    fi[d] = k * delta * unit[d];
    fj[d] = -fi[d];
  }
  return finite(energy) && finite_vector(fi) && finite_vector(fj);
}

BONDED_HD inline bool harmonic_angle(
  const double a[3],
  const double b[3],
  double theta0,
  double k,
  double& energy,
  double fi[3],
  double fj[3],
  double fk[3])
{
  energy = 0.0;
  for (int d = 0; d < 3; ++d)
    fi[d] = fj[d] = fk[d] = 0.0;
  if (k == 0.0)
    return true;
  double u[3], v[3], la, lb, normal[3];
  if (!normalize(a, u, la) || !normalize(b, v, lb))
    return false;
  cross(u, v, normal);
  const double sine = length(normal);
  const double cosine = fmax(-1.0, fmin(1.0, dot(u, v)));
  // Compute the small distance from the nearest angle endpoint without acos cancellation.
  const double delta =
    theta0 > pi * 0.5 ? (pi - theta0) - atan2(sine, -cosine) : atan2(sine, cosine) - theta0;
  energy = 0.5 * k * delta * delta;
  if (sine == 0.0) {
    // At an equilibrium endpoint the energy has a well-defined zero gradient.
    // A non-equilibrium collinear angle has no unique Cartesian gradient.
    return finite(energy) && fabs(delta) <= 8.0 * DBL_EPSILON * pi;
  }
  for (int d = 0; d < 3; ++d)
    normal[d] /= sine;
  double direction_i[3], direction_k[3];
  cross(normal, u, direction_i);
  cross(v, normal, direction_k);
  for (int d = 0; d < 3; ++d) {
    fi[d] = k * delta * direction_i[d] / la;
    fk[d] = k * delta * direction_k[d] / lb;
    fj[d] = -fi[d] - fk[d];
  }
  return finite(energy) && finite_vector(fi) && finite_vector(fj) && finite_vector(fk);
}

BONDED_HD inline bool periodic_dihedral(
  const double b1[3],
  const double b2[3],
  const double b3[3],
  double k,
  int multiplicity,
  double phase,
  double& phi,
  double& energy,
  double fi[3],
  double fj[3],
  double fk[3],
  double fl[3])
{
  energy = phi = 0.0;
  for (int d = 0; d < 3; ++d)
    fi[d] = fj[d] = fk[d] = fl[d] = 0.0;
  if (k == 0.0)
    return true;
  double u1[3], u2[3], u3[3], l1, l2, l3, n1[3], n2[3];
  if (!normalize(b1, u1, l1) || !normalize(b2, u2, l2) || !normalize(b3, u3, l3))
    return false;
  cross(u1, u2, n1);
  cross(u2, u3, n2);
  const double n1_squared = dot(n1, n1), n2_squared = dot(n2, n2);
  if (n1_squared <= 1.0e-24 || n2_squared <= 1.0e-24)
    return false;
  phi = atan2(dot(u1, n2), dot(n1, n2));
  const double argument = multiplicity * phi - phase;
  energy = k * (1.0 + cos(argument));
  const double derivative = -k * multiplicity * sin(argument);
  const double p1 = (l1 / l2) * dot(u1, u2), p3 = (l3 / l2) * dot(u3, u2);
  for (int d = 0; d < 3; ++d) {
    fi[d] = derivative * n1[d] / (l1 * n1_squared);
    fl[d] = -derivative * n2[d] / (l3 * n2_squared);
    fj[d] = -(1.0 + p1) * fi[d] + p3 * fl[d];
    fk[d] = p1 * fi[d] - (1.0 + p3) * fl[d];
  }
  return finite(phi) && finite(energy) && finite_vector(fi) && finite_vector(fj) &&
         finite_vector(fk) && finite_vector(fl);
}

// GPUMD tensor order: xx, yy, zz, xy, xz, yz, yx, zx, zy.
// Positions must be locally unwrapped relative to a participating atom.
template <int N>
BONDED_HD inline bool
interaction_virial(const double (&r)[N][3], const double (&force)[N][3], double tensor[9])
{
  const int row[9] = {0, 1, 2, 0, 0, 1, 1, 2, 2};
  const int column[9] = {0, 1, 2, 1, 2, 2, 0, 0, 1};
  for (int c = 0; c < 9; ++c) {
    tensor[c] = 0.0;
    for (int i = 0; i < N; ++i)
      tensor[c] += r[i][row[c]] * force[i][column[c]];
    if (!finite(tensor[c]))
      return false;
  }
  return true;
}
} // namespace bonded_geometry
#undef BONDED_HD
