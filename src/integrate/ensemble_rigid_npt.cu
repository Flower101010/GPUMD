#include "ensemble_rigid_npt.cuh"
#include "integrate/rigid_body_integrator.cuh"
#include "model/rigid_body_input.cuh"
#include "utilities/common.cuh"
#include <cmath>
#include <stdexcept>

namespace {
void check_box(const Box& box)
{
  if (box.pbc_x != 1 || box.pbc_y != 1 || box.pbc_z != 1 ||
      box.cpu_h[1] != 0 || box.cpu_h[2] != 0 || box.cpu_h[3] != 0 ||
      box.cpu_h[5] != 0 || box.cpu_h[6] != 0 || box.cpu_h[7] != 0 ||
      !std::isfinite(box.get_volume()) || box.get_volume() <= 0 ||
      box.cpu_h[0] <= 0 || box.cpu_h[4] <= 0 || box.cpu_h[8] <= 0)
    throw std::invalid_argument("Rigid NPT requires a fully periodic orthogonal box with positive lengths.");
}
}

Ensemble_Rigid_NPT::Ensemble_Rigid_NPT(
  int grouping, double t, double tc, double p, double pc, double dt)
  : molecule_grouping_method_(grouping), target_temperature_(t), target_pressure_(p),
    temperature_coupling_(tc), pressure_coupling_(pc), time_step_(dt)
{
  if (grouping < 0 || !std::isfinite(t) || t <= 0 || !std::isfinite(p) ||
      !std::isfinite(tc) || tc < 1 || !std::isfinite(pc) || pc < 1 ||
      !std::isfinite(dt) || dt <= 0)
    throw std::invalid_argument("Rigid NPT requires valid grouping, finite T/P, positive T/dt and couplings >= 1.");
  type = 13;
  temperature = t;
  temperature_coupling = tc;
  target_pressure[0] = p;
  num_target_pressure_components = 1;
}

void Ensemble_Rigid_NPT::initialize(Atom& atom, const Box& box, const std::vector<Group>& groups)
{
  if (stage_ == Stage::half_step)
    throw std::logic_error("Cannot reinitialize rigid NPT between integration stages.");
  stage_ = Stage::uninitialized;
  nh_.reset();
  rigid_.clear();
  thermo_state_.clear();
  check_box(box);
  if (atom.number_of_atoms <= 0 || atom.unwrapped_position.size() ||
      fixed_group >= 0 || move_group >= 0 || deform_x || deform_y || deform_z ||
      deform_xy || deform_xz || deform_yz)
    throw std::invalid_argument("Rigid NPT requires a nonempty system without fix, move, deform or unwrapped storage.");
  const auto labels = rigid_body_labels_from_group(groups, molecule_grouping_method_, atom.number_of_atoms);
  try {
    rigid_.initialize(atom, box, labels);
    rigid_.rebuild_positions(atom, box);
    rigid_.rebuild_velocities(atom);
    thermo_state_.initialize(rigid_);
    nh_.reset(new RigidNH({
      thermo_state_.translational_degrees_of_freedom(), thermo_state_.rotational_degrees_of_freedom(),
      K_B * target_temperature_, time_step_ * temperature_coupling_,
      time_step_ * pressure_coupling_, target_pressure_}));
    stage_ = Stage::needs_setup;
  } catch (...) {
    rigid_.clear();
    thermo_state_.clear();
    throw;
  }
}

const RigidNH& Ensemble_Rigid_NPT::extended_system() const
{
  if (!nh_) throw std::logic_error("Rigid NPT must be initialized.");
  return *nh_;
}

void Ensemble_Rigid_NPT::check_step(
  double dt, const Box& box, const Atom& atom, const GPU_Vector<double>& thermo) const
{
  check_box(box);
  if (!std::isfinite(dt) || dt <= 0 || dt != time_step_ || temperature != target_temperature_ ||
      target_pressure[0] != target_pressure_)
    throw std::invalid_argument("Rigid NPT requires a fixed time step and constant target temperature/pressure per run.");
  if (fixed_group >= 0 || move_group >= 0 || deform_x || deform_y || deform_z ||
      deform_xy || deform_xz || deform_yz || atom.unwrapped_position.size())
    throw std::invalid_argument("Rigid NPT does not support fix, move, deform or unwrapped storage.");
  const auto view = rigid_.device_view();
  const std::size_t n = view.number_of_atoms;
  if (atom.number_of_atoms != view.number_of_atoms || atom.mass.size() != n ||
      atom.position_per_atom.size() != 3 * n || atom.velocity_per_atom.size() != 3 * n ||
      atom.force_per_atom.size() != 3 * n || atom.potential_per_atom.size() != n ||
      atom.virial_per_atom.size() != 9 * n || thermo.size() < 8)
    throw std::invalid_argument("Rigid NPT requires matching Atom arrays and eight thermo entries.");
}

double Ensemble_Rigid_NPT::update_thermo(
  const std::vector<Group>& groups, const Box& box, const Atom& atom, GPU_Vector<double>& thermo)
{
  find_thermo(false, box.get_volume(), groups, atom.mass, atom.potential_per_atom,
    atom.velocity_per_atom, atom.virial_per_atom, thermo);
  // 使用当前力/姿态/动量的机械约束力，排除恒温器和活塞摩擦的冲量。
  // trace 给出分子压力：3PV=2K_translation+tr(W_model)-sum_members(r_rel dot f)。
  thermo_state_.sample_constraint_virial(rigid_, atom);
  thermo_state_.add_constraint_pressure(box.get_volume(), thermo);
  double values[5];
  thermo.copy_to_host(values, 5);
  const double pressure = (values[2] + values[3] + values[4]) / 3;
  if (!std::isfinite(pressure)) throw std::overflow_error("Nonfinite rigid NPT pressure.");
  return pressure;
}

void Ensemble_Rigid_NPT::compute1(
  double dt, const std::vector<Group>& groups, Box& box, Atom& atom, GPU_Vector<double>& thermo)
{
  if (stage_ != Stage::needs_setup && stage_ != Stage::full_step)
    throw std::logic_error("Rigid NPT compute1 requires initialization and a completed previous step.");
  check_step(dt, box, atom, thermo);
  rigid_.remap_centers(box);
  rigid_.compute_forces_and_torques(atom);
  if (stage_ == Stage::needs_setup) {
    const auto kinetic = thermo_state_.sample_kinetic_energy(rigid_, atom);
    const double pressure = update_thermo(groups, box, atom, thermo);
    nh_->advance_piston(dt / 2, pressure, box.get_volume(), kinetic.twice_translation, kinetic.twice_rotation);
  }
  // FixRigidNHSmall::initial_integrate: kick 后使用旧链与活塞速率缩放。
  const auto scales = nh_->momentum_scales(dt / 2);
  const auto drift = rigid_integrator::isotropic_drift_factors(dt, nh_->piston().rate);
  rigid_integrator::kick_translation(dt / 2, atom, rigid_);
  rigid_integrator::kick_rotation(dt / 2, rigid_);
  rigid_integrator::scale_momenta(scales.translation, scales.rotation, atom, rigid_);
  // 先缩盒/COM，再 drift(phi*v)，精确实现 R'=exp(nu*dt)R+expm1(nu*dt)/nu*v。
  // 内部几何不随盒子缩放；体姿态仍用现有 Miller drift。
  rigid_.rescale_box(atom, box, drift.box_scale);
  rigid_integrator::drift_translation(drift.translation_time, atom, rigid_);
  rigid_integrator::drift_rotation(dt, rigid_);
  rigid_.remap_centers(box);
  rigid_.rebuild_positions(atom, box);
  rigid_.rebuild_velocities(atom);
  const auto kinetic = thermo_state_.sample_kinetic_energy(rigid_, atom);
  nh_->advance_thermostats(dt, kinetic.twice_translation, kinetic.twice_rotation);
  nh_->advance_piston_thermostat(dt);
  stage_ = Stage::half_step;
}

void Ensemble_Rigid_NPT::compute2(
  double dt, const std::vector<Group>& groups, Box& box, Atom& atom, GPU_Vector<double>& thermo)
{
  if (stage_ != Stage::half_step)
    throw std::logic_error("Rigid NPT compute2 requires compute1 and newly evaluated bead forces.");
  check_step(dt, box, atom, thermo);
  rigid_.compute_forces_and_torques(atom);
  // FixRigidNHSmall::final_integrate: 新链速率，先缩放再 kick。
  const auto scales = nh_->momentum_scales(dt / 2);
  rigid_integrator::scale_momenta(scales.translation, scales.rotation, atom, rigid_);
  rigid_integrator::kick_translation(dt / 2, atom, rigid_);
  rigid_integrator::kick_rotation(dt / 2, rigid_);
  rigid_.rebuild_velocities(atom);
  const auto kinetic = thermo_state_.sample_kinetic_energy(rigid_, atom);
  const double pressure = update_thermo(groups, box, atom, thermo);
  nh_->advance_piston(dt / 2, pressure, box.get_volume(), kinetic.twice_translation, kinetic.twice_rotation);
  stage_ = Stage::full_step;
}
