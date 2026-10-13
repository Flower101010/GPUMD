#pragma once
#include "ensemble.cuh"
#include "rigid_body.cuh"

class Ensemble_Rigid_NVE : public Ensemble
{
public:
  // 独立的分子 grouping method；单成员组由现有 adapter 转为自由粒子。
  explicit Ensemble_Rigid_NVE(int molecule_grouping_method);
  ~Ensemble_Rigid_NVE() override = default;

  // 每次 run 开始、首次 Force::compute 之前调用；不读取尚未计算的力。
  void initialize(Atom& atom, const Box& box, const std::vector<Group>& groups);

  // 汇总旧力 -> 平动/转动半 kick -> 整步 drift -> remap/rebuild。
  void compute1(
    double time_step, const std::vector<Group>& groups,
    Box& box, Atom& atom, GPU_Vector<double>& thermo) override;

  // 中间 Force::compute 后：汇总新力 -> 后半 kick -> rebuild v -> thermo。
  void compute2(
    double time_step, const std::vector<Group>& groups,
    Box& box, Atom& atom, GPU_Vector<double>& thermo) override;

  // 3*N_free + 6*N_nonlinear_body，不自动扣除全体系动量自由度。
  int thermal_degrees_of_freedom() const override;
  const RigidBodySystem& rigid_bodies() const { return rigid_; }

private:
  enum class Stage { uninitialized, full_step, half_step };
  int molecule_grouping_method_;
  int thermal_dof_ = 0;
  Stage stage_ = Stage::uninitialized;
  double step_time_step_ = 0.0; // 同一 run 内固定步长；两个半步必须一致
  RigidBodySystem rigid_;

  void check_step(double time_step, const Atom& atom, const GPU_Vector<double>& thermo) const;

  // 调用已有 kick_rotation / integrate_free_rotation，不重写 Miller。
  // step1 半 kick + 五阶段自由旋转；step2 只有半 kick。
  void integrate_rotation(bool is_step1, double time_step);
};
