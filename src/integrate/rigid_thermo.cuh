#pragma once

#include "integrate/rigid_body.cuh"

// 共用刚体温度自由度与离散约束 virial；不执行积分，也不计算相互作用。
// 一个统计窗口只包含物理 kick/drift：thermostat/barostat 更新应置于窗口之外。
class RigidThermo
{
public:
  void initialize(const RigidBodySystem& rigid);
  void clear();
  int degrees_of_freedom() const { return degrees_of_freedom_; }
  const GPU_Vector<double>& constraint_virial_per_body() const { return constraint_virial_; }

  // 整步开始，保存旧姿态；两个 record 均须在 rebuild_velocities 覆盖 bead 速度前调用。
  // h 是物理半步的实际时长，而非整步 dt；atom.velocity 是该次 kick 前的速度。
  void begin_step(const RigidBodySystem& rigid);
  void record_first_half(double h, const RigidBodySystem& rigid, const Atom& atom);
  void record_second_half(double h, const RigidBodySystem& rigid, const Atom& atom);

  // 加到新计算的基础 thermo 压力上，只调用一次；不会改写模型的 Atom virial。
  // 必须已完成两半步；6*B SoA: xx,yy,zz,xy,xz,yz。
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

  RigidBodySystem::ConstDeviceView checked_view(const RigidBodySystem& rigid) const;
  void check_sample(double h, const RigidBodySystem& rigid, const Atom& atom) const;
};
