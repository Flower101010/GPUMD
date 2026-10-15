#pragma once
#include "ensemble.cuh"
#include "rigid_body.cuh"
#include "rigid_nh.cuh"
#include "rigid_thermo.cuh"
#include <memory>

class Ensemble_Rigid_NVT : public Ensemble
{
public:
  // 独立的分子 grouping method；单成员组由现有 adapter 转为自由粒子。
  // temperature_coupling 以步数计；实际 tau = time_step * temperature_coupling。
  Ensemble_Rigid_NVT(
    int molecule_grouping_method, double target_temperature, double temperature_coupling,
    double time_step);
  ~Ensemble_Rigid_NVT() override = default;

  // 每次 run 开始、首次 Force::compute 之前调用；不读取尚未计算的力。
  void initialize(Atom& atom, const Box& box, const std::vector<Group>& groups);

  // 旧力半 kick -> 独立平动/转动缩放 -> drift -> 中间动能推进双链整步。
  void compute1(
    double time_step,
    const std::vector<Group>& groups,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) override;

  // 中间 Force::compute 后：独立缩放 -> 新力半 kick -> 机械约束压力/thermo。
  void compute2(
    double time_step,
    const std::vector<Group>& groups,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) override;

  // 3*N_free + 6*N_nonlinear_body，不自动扣除全体系动量自由度。
  int thermal_degrees_of_freedom() const override;
  const RigidBodySystem& rigid_bodies() const { return rigid_; }
  const RigidThermostat& thermostat() const;
  // 总压力诊断用的每体约束 virial，6*B SoA：xx,yy,zz,xy,xz,yz。
  // initialize 后为零；compute2 更新为该完整时间层的瞬时机械约束贡献。
  // 不等同于 LAMMPS 的每原子 stress 分配；Atom 的模型 virial 保持原值。
  const GPU_Vector<double>& constraint_virial_per_body() const
  {
    return thermo_state_.constraint_virial_per_body();
  }

private:
  enum class Stage { uninitialized, full_step, half_step };
  int molecule_grouping_method_;
  Stage stage_ = Stage::uninitialized;
  const double thermostat_time_step_;
  const double temperature_coupling_; // 固定 tau；温度斜坡中 Q 随当前目标温度更新
  RigidBodySystem rigid_;
  RigidThermo thermo_state_;
  std::unique_ptr<RigidThermostat> thermostat_;

  void check_step(double time_step, const Atom& atom, const GPU_Vector<double>& thermo) const;
};
