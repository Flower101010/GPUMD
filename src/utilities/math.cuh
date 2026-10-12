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

// 值类型，用于线程局部计算，也可作为 GPU_Vector 的元素；不持有 GPU 内存。
struct Vec3 { double x, y, z; };
struct Quaternion { double w, x, y, z; };
enum class PrincipalAxis { x = 1, y = 2, z = 3 };

// TODO：以下接口只声明，函数体由你补在本头文件中，保持 HD inline。
// 不使用空函数体/假结果占位；尚未实现的接口不能被调用。
// Vec3 的基本代数。
RIGID_MATH_HD inline Vec3 add(const Vec3& a, const Vec3& b);
RIGID_MATH_HD inline Vec3 subtract(const Vec3& a, const Vec3& b);
RIGID_MATH_HD inline Vec3 scale(const Vec3& v, double factor);
RIGID_MATH_HD inline double dot(const Vec3& a, const Vec3& b);
RIGID_MATH_HD inline Vec3 cross(const Vec3& a, const Vec3& b);
RIGID_MATH_HD inline double norm_squared(const Vec3& v);
RIGID_MATH_HD inline double norm_squared(const Quaternion& q);
RIGID_MATH_HD inline double norm(const Vec3& v);
RIGID_MATH_HD inline double norm(const Quaternion& q);

// 归一化：零范数/非有限输入返回 false，输入保持原值；成功返回 true。
// norm/normalize 应避免可避免的中间平方溢出；热路径不会自动归一化 q。
RIGID_MATH_HD inline bool normalize(Vec3& v);
RIGID_MATH_HD inline bool normalize(Quaternion& q);

// q 必须是单位四元数；rotate 为 body -> space，inverse_rotate 为 space -> body。
// 不在每次旋转时归一化；用已知轴旋转和 q/-q 等价性验证约定。
RIGID_MATH_HD inline Vec3 rotate(const Quaternion& q, const Vec3& v_body);
RIGID_MATH_HD inline Vec3 inverse_rotate(const Quaternion& q, const Vec3& v_space);

// 刚体力学表示转换，只做单个刚体的数学计算，不访问 Atom/GPU_Vector。
// L_body=R(q)^T*L_space；p=2*q*(0,L_body)。
RIGID_MATH_HD inline Quaternion angular_momentum_to_conjqm(
  const Quaternion& q, const Vec3& angular_momentum_space);
// L_space=R(q)*(0.5*Im(conjugate(q)*p))，要求 q dot p=0。
RIGID_MATH_HD inline Vec3 conjqm_to_angular_momentum(
  const Quaternion& q, const Quaternion& p);
// omega_space=R(q)*(I_body^+*R(q)^T*L_space)。
// 主惯量已由初始化验证：有限、非负，零轴精确存 0；零轴角速度为 0。
RIGID_MATH_HD inline Vec3 angular_momentum_to_omega(
  const Quaternion& q, const Vec3& angular_momentum_space, const Vec3& inertia);

} // namespace rigid_math

// Miller 力矩 kick（TODO）：位置/quat 固定，tau_body=R(q)^T*tau_space，
// p += 2*dt*q*(0,tau_body)，dt 为本次 kick 的时间（半步调用传 dt/2）。
// 该接口不自动漂移 q，不计算 bead 力，也不归一化 q/p。
namespace rigid_math {
RIGID_MATH_HD inline void kick_rotation(
  const Quaternion& q, Quaternion& p, const Vec3& torque_space, double dt);
}

// 以下是你已完成的实现，从 math.cu 移入以支持跨 .cu 的设备调用。
// 函数定义必须在调用方可见；无需为这些小函数启用设备分离编译。
// multiply/conjugate/dot 是基础代数；permute/rotate_axis/free_rotation 是 Miller 算法。
// 自由漂移前提：|q|=1，q dot p=0，q 与 p 不别名，inertia 已验证，dt 为内部时间单位。
// 不包含外力矩 kick，不等同于完整 NVE；零惯量跳过该轴，不代替退化几何验证。
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
  const double I = (axis == PrincipalAxis::x) ? inertia.x :
                   (axis == PrincipalAxis::y) ? inertia.y : inertia.z;
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
