#pragma once
#include "ensemble.cuh"
#include "rigid_body.cuh"
#include "rigid_thermo.cuh"

class Ensemble_Rigid_NVT : public Ensemble
{
public:
  // 独立的分子 grouping method；单成员组由现有 adapter 转为自由粒子。
  explicit Ensemble_Rigid_NVT(int molecule_grouping_method);
  ~Ensemble_Rigid_NVT() override = default;

  // 每次 run 开始、首次 Force::compute 之前调用；不读取尚未计算的力。
  void initialize(Atom& atom, const Box& box, const std::vector<Group>& groups);

  // 汇总旧力 -> 平动/转动半 kick -> 整步 drift -> remap/rebuild。
  void compute1(
    double time_step,
    const std::vector<Group>& groups,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) override;

  // 中间 Force::compute 后：汇总新力 -> 后半 kick -> rebuild v -> thermo。
  void compute2(
    double time_step,
    const std::vector<Group>& groups,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) override;

  // 3*N_free + 6*N_nonlinear_body，不自动扣除全体系动量自由度。
  int thermal_degrees_of_freedom() const override;
  const RigidBodySystem& rigid_bodies() const { return rigid_; }
  // 总压力诊断用的每体约束 virial，6*B SoA：xx,yy,zz,xy,xz,yz。
  // initialize 后为零；compute1 为第一半贡献，compute2 后为完整一步贡献。
  // 不等同于 LAMMPS 的每原子 stress 分配；Atom 的模型 virial 保持原值。
  const GPU_Vector<double>& constraint_virial_per_body() const
  {
    return thermo_state_.constraint_virial_per_body();
  }

private:
  enum class Stage { uninitialized, full_step, half_step };
  int molecule_grouping_method_;
  Stage stage_ = Stage::uninitialized;
  double step_time_step_ = 0.0; // 同一 run 内固定步长；两个半步必须一致
  RigidBodySystem rigid_;
  RigidThermo thermo_state_;

  void check_step(double time_step, const Atom& atom, const GPU_Vector<double>& thermo) const;
};
