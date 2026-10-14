#pragma once

#include "integrate/rigid_body.cuh"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define RIGID_INTEGRATOR_HD __host__ __device__
#else
#define RIGID_INTEGRATOR_HD
#endif

namespace rigid_integrator
{
// h 是本操作实际执行的时间长度，使用 GPUMD 内部单位。
RIGID_INTEGRATOR_HD inline void
kick_translation(double h, double mass, const Vec3& force, Vec3& velocity)
{
  assert(std::isfinite(mass) && mass > 0.0);
  assert(std::isfinite(h));
  velocity = rigid_math::add(velocity, rigid_math::scale(force, h / mass));
}
RIGID_INTEGRATOR_HD inline void drift_translation(double h, Vec3& position, const Vec3& velocity)
{
  assert(std::isfinite(h));
  position = rigid_math::add(position, rigid_math::scale(velocity, h));
}

// GPU 操作同时覆盖质心与自由粒子；受管理 bead 不重复积分。
// 转动操作仅更新 quat/conjqm。均不计算力、包装坐标或重建 bead。
// 默认 GPU stream；支持有限负 h，以便验证可逆性；空体系不 launch。
void kick_translation(double h, Atom& atom, RigidBodySystem& rigid);
void drift_translation(double h, Atom& atom, RigidBodySystem& rigid);
void kick_rotation(double h, RigidBodySystem& rigid);
void drift_rotation(double h, RigidBodySystem& rigid);

// 缩放自由粒子 v、刚体 vcm 和 Miller 共轭动量 p；不改变 xcm/quat。
// 受管理 bead 不直接缩放；调用者随后 rebuild_velocities，使两层状态一致。
void scale_momenta(double factor, Atom& atom, RigidBodySystem& rigid);

// 保留原有 Verlet 接口：step1 半 kick + 整 drift，step2 只有半 kick。
// GPU 版本仍将平动 kick/drift 融合为一次 launch；新 ensemble 可独立编排上述操作。
RIGID_INTEGRATOR_HD inline void velocity_verlet(
  bool is_step1, double dt, double mass, const Vec3& force, Vec3& position, Vec3& velocity)
{
  kick_translation(0.5 * dt, mass, force, velocity);
  if (is_step1)
    drift_translation(dt, position, velocity);
}
RIGID_INTEGRATOR_HD inline void velocity_verlet(
  bool is_step1,
  double dt,
  const RigidBodySystem::BodyGeometry& geometry,
  const RigidBodySystem::BodyLoad& load,
  RigidBodySystem::BodyState& state)
{
  velocity_verlet(is_step1, dt, geometry.mass, load.fcm, state.xcm, state.vcm);
}
void velocity_verlet(bool is_step1, double dt, Atom& atom, RigidBodySystem& rigid);
} // namespace rigid_integrator

#undef RIGID_INTEGRATOR_HD
