#pragma once

#include "integrate/rigid_body.cuh"

// 共用刚体温度自由度与约束 virial；不执行积分，也不计算相互作用。
// 离散窗口只包含物理 kick/drift；NPT 可改用瞬时机械约束力采样。
class RigidThermo
{
public:
  struct KineticSample {
    // 对应 FixRigidNHSmall 的 akin_t/akin_r（转换到内部能量单位后）。
    // 我们的平动项额外包含自由粒子，LAMMPS 刚体 fix 本身只统计刚体。
    double twice_translation; // 2K，自由粒子 + 刚体 COM；内部能量单位
    double twice_rotation;    // 2K，刚体转动
    double twice_total() const { return twice_translation + twice_rotation; }
  };
  void initialize(const RigidBodySystem& rigid);
  void clear();
  int degrees_of_freedom() const { return degrees_of_freedom_; }
  int translational_degrees_of_freedom() const { return 3 * (atoms_ - members_ + bodies_); }
  int rotational_degrees_of_freedom() const { return 3 * bodies_; }
  // GPU 归约独立状态，不要求 bead 速度已重建；仅将两个标量同步到 HOST。
  // 不推进约束 virial 窗口。当前只支持初始化器接受的非线性刚体。
  KineticSample sample_kinetic_energy(const RigidBodySystem& rigid, const Atom& atom);
  const GPU_Vector<double>& constraint_virial_per_body() const { return constraint_virial_; }

  // 整步开始，保存旧姿态；两个 record 均须在 rebuild_velocities 覆盖 bead 速度前调用。
  // h 是物理半步的实际时长，而非整步 dt；atom.velocity 是该次 kick 前的速度。
  void begin_step(const RigidBodySystem& rigid);
  void record_first_half(double h, const RigidBodySystem& rigid, const Atom& atom);
  void record_second_half(double h, const RigidBodySystem& rigid, const Atom& atom);

  // 当前时间层的机械约束 virial（覆盖原窗口），用于变盒 NPT 的 setup 和压力反馈。
  // 调用前先更新 load，且 bead 速度须与状态一致；不计恒温/活塞摩擦。
  // 由 Euler 方程求角加速度，再用 fc=m*(a_cm+a_rel)-f，避免差分混入耦合冲量。
  void sample_constraint_virial(const RigidBodySystem& rigid, const Atom& atom);

  // 加到新计算的基础 thermo 压力上，只调用一次；不会改写模型的 Atom virial。
  // 必须已完成两半步或一次瞬时采样；6*B SoA: xx,yy,zz,xy,xz,yz。
  void add_constraint_pressure(double volume, GPU_Vector<double>& thermo) const;

private:
  enum class Stage { uninitialized, ready, step_started, first_half_recorded, complete };
  Stage stage_ = Stage::uninitialized;
  int atoms_ = 0;
  int bodies_ = 0;
  int members_ = 0;
  int degrees_of_freedom_ = 0;
  double half_step_ = 0.0;
  GPU_Vector<Quaternion> previous_quats_;
  GPU_Vector<double> constraint_virial_;
  GPU_Vector<double> kinetic_sample_; // 两个归约结果；initialize 时分配

  RigidBodySystem::ConstDeviceView checked_view(const RigidBodySystem& rigid) const;
  void check_sample(double h, const RigidBodySystem& rigid, const Atom& atom) const;
};
