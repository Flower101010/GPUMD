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

// 平动（自由粒子 + 刚体 COM）与转动（Miller p）使用独立正因子。
// 支持分开的 NHC 或 MTK 动量耦合；不选择耦合系数、不重建 bead 速度。
void scale_momenta(
  double translation_factor, double rotation_factor, Atom& atom, RigidBodySystem& rigid);

struct IsotropicDriftFactors {
  double box_scale;
  double translation_time;
};

/**
 * @brief 计算各向同性盒子伸缩时的盒长缩放系数和位置漂移系数。
 *
 * @param duration 漂移时长 h，使用 GPUMD 内部时间单位。
 * @param dilation_rate 盒长相对变化率 nu = (dL/dt)/L，
 *        单位为内部时间的倒数；正值表示膨胀，负值表示收缩。
 *
 * @return 两个系数：
 *         - box_scale：盒长缩放倍数 exp(nu*h)。
 *         - translation_time：速度项系数 expm1(nu*h)/nu；
 *           nu=0 时为 h。
 *
 * @note 对应更新公式：
 *       L_new = box_scale * L_old；
 *       R_new = box_scale * R_old + translation_time * V。
 *       假设本子步内 V 和 nu 保持不变，以坐标原点为伸缩中心。
 *
 * @note 仅计算系数，不修改状态或执行周期性包装。
 *       适用于刚体质心和自由粒子，不缩放刚体内部相对坐标。
 *       若拆分 remap/drift 子步，需根据调用顺序重新组合系数。
 */
IsotropicDriftFactors isotropic_drift_factors(double duration, double dilation_rate);

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
