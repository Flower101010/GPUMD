#pragma once

namespace rigid_math
{

struct Vec3 {
  double x, y, z;
};

struct Quaternion {
  double w, x, y, z;
};

// 基础四元数操作
__host__ __device__ Quaternion multiply(const Quaternion& a, const Quaternion& b);

__host__ __device__ Quaternion conjugate(const Quaternion& q);

// 1. Miller permutation operator
__host__ __device__ Quaternion permute(const Quaternion& q, int k);

// Miller 单轴解析积分
__host__ __device__ void rotate_axis(
    Quaternion& q, Quaternion& p,
    const Vec3& inertia, int k, double dt);

// Miller symmetric five-stage splitting
__host__ __device__ void integrate_rotation(
    Quaternion& q, Quaternion& p,
    const Vec3& inertia, double dt);

} // namespace rigid_math
