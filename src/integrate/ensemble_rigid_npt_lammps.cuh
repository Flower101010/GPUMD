#pragma once
#include "ensemble_rigid_npt.cuh"

// Opt-in FixRigidNHSmall ordering, including remap -> set_xv -> remap and
// discrete impulse virial. This intentionally does not preserve bead distances
// at the force/output time layer. Common validation/state/NH chains are reused.
class Ensemble_Rigid_NPT_LAMMPS final : public Ensemble_Rigid_NPT
{
public:
  using Ensemble_Rigid_NPT::Ensemble_Rigid_NPT;
  void initialize(Atom&, const Box&, const std::vector<Group>&) override;
  void compute1(double, const std::vector<Group>&, Box&, Atom&, GPU_Vector<double>&) override;
  void compute2(double, const std::vector<Group>&, Box&, Atom&, GPU_Vector<double>&) override;
  const GPU_Vector<double>& constraint_virial_per_body() const override { return impulse_virial_; }

private:
  GPU_Vector<double> saved_positions_, initial_velocities_, impulse_virial_;
  rigid_math::Vec3 box_center_{};
  double member_scale_ = 1.0;
  void capture_positions(const Atom&, const Box&);
  void tally_impulse(double h, double weight, bool accumulate, double position_scale,
                     const Box&, const Atom&, const GPU_Vector<double>& before);
  double update_lammps_thermo(const std::vector<Group>&, const Box&, const Atom&, GPU_Vector<double>&);
};
