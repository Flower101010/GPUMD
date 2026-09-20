/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once

#include <cmath>

namespace bonded_geometry
{
inline __host__ __device__ double dot(const double a[3], const double b[3])
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline __host__ __device__ void cross(const double a[3], const double b[3], double result[3])
{
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];
}

inline __host__ __device__ bool harmonic_angle(
  const double a[3],
  const double b[3],
  const double equilibrium_angle,
  const double angle_constant,
  double& energy,
  double force_i[3],
  double force_j[3],
  double force_k[3])
{
  const double a_squared = dot(a, a);
  const double b_squared = dot(b, b);
  if (a_squared == 0.0 || b_squared == 0.0) {
    return false;
  }

  const double inverse_ab = 1.0 / sqrt(a_squared * b_squared);
  double cosine = dot(a, b) * inverse_ab;
  cosine = fmax(-1.0, fmin(1.0, cosine));
  const double sine_squared = fmax(0.0, 1.0 - cosine * cosine);
  if (sine_squared <= 1.0e-24) {
    return false;
  }

  const double displacement = acos(cosine) - equilibrium_angle;
  energy = 0.5 * angle_constant * displacement * displacement;
  const double factor = angle_constant * displacement / sqrt(sine_squared);
  for (int d = 0; d < 3; ++d) {
    force_i[d] = factor * (b[d] * inverse_ab - cosine * a[d] / a_squared);
    force_k[d] = factor * (a[d] * inverse_ab - cosine * b[d] / b_squared);
    force_j[d] = -force_i[d] - force_k[d];
  }
  return true;
}

inline __host__ __device__ bool periodic_dihedral(
  const double b1[3],
  const double b2[3],
  const double b3[3],
  const double force_constant,
  const int multiplicity,
  const double phase,
  double& phi,
  double& energy,
  double force_i[3],
  double force_j[3],
  double force_k[3],
  double force_l[3])
{
  double normal_1[3];
  double normal_2[3];
  cross(b1, b2, normal_1);
  cross(b2, b3, normal_2);
  const double b1_squared = dot(b1, b1);
  const double normal_1_squared = dot(normal_1, normal_1);
  const double normal_2_squared = dot(normal_2, normal_2);
  const double b2_squared = dot(b2, b2);
  const double b3_squared = dot(b3, b3);
  if (
    b1_squared == 0.0 || b2_squared == 0.0 || b3_squared == 0.0 ||
    normal_1_squared <= 1.0e-24 * b1_squared * b2_squared ||
    normal_2_squared <= 1.0e-24 * b2_squared * b3_squared) {
    return false;
  }

  const double b2_length = sqrt(b2_squared);
  phi = atan2(b2_length * dot(b1, normal_2), dot(normal_1, normal_2));
  const double argument = multiplicity * phi - phase;
  energy = force_constant * (1.0 + cos(argument));
  const double energy_derivative = -force_constant * multiplicity * sin(argument);
  const double scale_i = energy_derivative * b2_length / normal_1_squared;
  const double scale_l = -energy_derivative * b2_length / normal_2_squared;
  const double projection_1 = dot(b1, b2) / b2_squared;
  const double projection_3 = dot(b3, b2) / b2_squared;
  for (int d = 0; d < 3; ++d) {
    force_i[d] = scale_i * normal_1[d];
    force_l[d] = scale_l * normal_2[d];
    force_j[d] = -(1.0 + projection_1) * force_i[d] + projection_3 * force_l[d];
    force_k[d] = projection_1 * force_i[d] - (1.0 + projection_3) * force_l[d];
  }
  return true;
}
} // namespace bonded_geometry
