#pragma once
#include "ensemble.cuh"
#include "rigid_body.cuh"
#include "rigid_nh.cuh"
#include "rigid_thermo.cuh"
#include <memory>

// 三维全周期正交盒、各向同性、恒定 T/P；Miller + LAMMPS rigid NH 链。
class Ensemble_Rigid_NPT : public Ensemble
{
public:
  // coupling 为步数，tau=dt*coupling；pressure 使用内部单位 eV/Angstrom^3。
  Ensemble_Rigid_NPT(
    int molecule_grouping_method, double target_temperature, double temperature_coupling,
    double target_pressure, double pressure_coupling, double time_step);
  void initialize(Atom& atom, const Box& box, const std::vector<Group>& groups);

  // 初次 compute1 延迟到初始 Force::compute 之后准备压力和活塞半步。
  // kick -> scale -> affine translation/rotation drift -> advance three chains.
  void compute1(
    double time_step, const std::vector<Group>& groups, Box& box, Atom& atom,
    GPU_Vector<double>& thermo) override;
  // scale -> kick -> rebuild -> instantaneous constrained pressure -> piston half step.
  void compute2(
    double time_step, const std::vector<Group>& groups, Box& box, Atom& atom,
    GPU_Vector<double>& thermo) override;

  int thermal_degrees_of_freedom() const override { return thermo_state_.degrees_of_freedom(); }
  const RigidBodySystem& rigid_bodies() const { return rigid_; }
  const RigidNH& extended_system() const;
  // 当前完整时间层的瞬时约束 virial，6*B SoA；不含恒温/活塞摩擦。
  const GPU_Vector<double>& constraint_virial_per_body() const
  {
    return thermo_state_.constraint_virial_per_body();
  }

private:
  enum class Stage { uninitialized, needs_setup, full_step, half_step };
  int molecule_grouping_method_;
  const double target_temperature_, target_pressure_, temperature_coupling_, pressure_coupling_;
  const double time_step_;
  Stage stage_ = Stage::uninitialized;
  RigidBodySystem rigid_;
  RigidThermo thermo_state_;
  std::unique_ptr<RigidNH> nh_;

  void check_step(double dt, const Box& box, const Atom& atom, const GPU_Vector<double>& thermo) const;
  double update_thermo(
    const std::vector<Group>& groups, const Box& box, const Atom& atom, GPU_Vector<double>& thermo);
};
