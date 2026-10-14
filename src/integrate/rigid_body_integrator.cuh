#pragma once

#include "integrate/rigid_body.cuh"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define RIGID_INTEGRATOR_HD __host__ __device__
#else
#define RIGID_INTEGRATOR_HD
#endif

namespace rigid_integrator
{
// 共用平动公式：step1 半步 kick + 整步 drift；step2 只有半步 kick。
// dt 使用 GPUMD 内部单位；调用者在两个阶段之间重新计算力。
RIGID_INTEGRATOR_HD inline void velocity_verlet(
  bool is_step1, double dt, double mass, const Vec3& force, Vec3& position, Vec3& velocity)
{
  assert(std::isfinite(mass) && mass > 0.0);
  assert(std::isfinite(dt));
  velocity = rigid_math::add(velocity, rigid_math::scale(force, 0.5 * dt / mass));
  if (is_step1)
    position = rigid_math::add(position, rigid_math::scale(velocity, dt));
}

// 刚体质心只是质量为 M、受合力 F 的平动粒子；不更新 quat/conjqm。
RIGID_INTEGRATOR_HD inline void velocity_verlet(
  bool is_step1,
  double dt,
  const RigidBodySystem::BodyGeometry& geometry,
  const RigidBodySystem::BodyLoad& load,
  RigidBodySystem::BodyState& state)
{
  velocity_verlet(is_step1, dt, geometry.mass, load.fcm, state.xcm, state.vcm);
}

// 一个 GPU launch 更新所有质心与自由粒子。受管理 bead 不单独积分。
// 前置条件：rigid 已初始化；body loads 和 Atom forces 对应当前阶段。
// 不汇总力、不更新转动、不 remap、不重建 bead；这些由 ensemble 排序。
// 默认 GPU stream；允许有限的负 dt 以验证可逆性，空体系直接返回。
void velocity_verlet(bool is_step1, double dt, Atom& atom, RigidBodySystem& rigid);
} // namespace rigid_integrator

#undef RIGID_INTEGRATOR_HD
