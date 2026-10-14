#include "ensemble_rigid_nvt.cuh"
#include "integrate/rigid_body_integrator.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/rigid_body_input.cuh"
#include <stdexcept>

Ensemble_Rigid_NVT::Ensemble_Rigid_NVT(int molecule_grouping_method)
  : molecule_grouping_method_(molecule_grouping_method)
{
  if (molecule_grouping_method < 0)
    throw std::invalid_argument("Rigid NVT requires a nonnegative molecule grouping method.");
  type = 0;
  temperature = 0.0;
}

void Ensemble_Rigid_NVT::initialize(Atom& atom, const Box& box, const std::vector<Group>& groups)
{
  if (stage_ == Stage::half_step)
    throw std::logic_error("Cannot reinitialize rigid NVT between its two integration stages.");
  stage_ = Stage::uninitialized;
  step_time_step_ = 0.0;
  rigid_.clear();
  thermo_state_.clear();
  if (atom.number_of_atoms <= 0)
    throw std::invalid_argument("Rigid NVT requires a nonempty system for temperature output.");
  if (
    fixed_group >= 0 || move_group >= 0 || deform_x || deform_y || deform_z || deform_xy ||
    deform_xz || deform_yz)
    throw std::invalid_argument(
      "Rigid NVT does not support fixed/moving groups or box deformation.");
  const auto labels =
    rigid_body_labels_from_group(groups, molecule_grouping_method_, atom.number_of_atoms);
  try {
    rigid_.initialize(atom, box, labels);
    if (rigid_.number_of_bodies() && atom.unwrapped_position.size())
      throw std::invalid_argument(
        "Rigid NVT does not support unwrapped trajectories for rigid members.");
    rigid_.rebuild_positions(atom, box);
    rigid_.rebuild_velocities(atom);
    thermo_state_.initialize(rigid_);
    stage_ = Stage::full_step;
  } catch (...) {
    rigid_.clear();
    thermo_state_.clear();
    throw;
  }
}

int Ensemble_Rigid_NVT::thermal_degrees_of_freedom() const
{
  return thermo_state_.degrees_of_freedom();
}

void Ensemble_Rigid_NVT::check_step(
  double dt, const Atom& atom, const GPU_Vector<double>& thermo) const
{
  if (!std::isfinite(dt) || dt <= 0.0 || (step_time_step_ != 0.0 && dt != step_time_step_))
    throw std::invalid_argument("Rigid NVT requires one finite positive fixed time step per run.");
  if (
    fixed_group >= 0 || move_group >= 0 || deform_x || deform_y || deform_z || deform_xy ||
    deform_xz || deform_yz)
    throw std::invalid_argument(
      "Rigid NVT does not support fixed/moving groups or box deformation.");
  const auto view = rigid_.device_view();
  const std::size_t n = view.number_of_atoms;
  if (
    atom.number_of_atoms != view.number_of_atoms || atom.mass.size() != n ||
    atom.position_per_atom.size() != 3 * n || atom.velocity_per_atom.size() != 3 * n ||
    atom.force_per_atom.size() != 3 * n || atom.potential_per_atom.size() != n ||
    atom.virial_per_atom.size() != 9 * n || thermo.size() < 8)
    throw std::invalid_argument(
      "Rigid NVT requires matching Atom arrays and eight thermo entries.");
  if (view.number_of_bodies && atom.unwrapped_position.size())
    throw std::invalid_argument(
      "Rigid NVT does not support unwrapped trajectories for rigid members.");
}

void Ensemble_Rigid_NVT::compute1(
  double dt, const std::vector<Group>& groups, Box& box, Atom& atom, GPU_Vector<double>& thermo)
{
  if (stage_ != Stage::full_step)
    throw std::logic_error(
      "Rigid NVT compute1 requires initialization and a completed previous step.");
  check_step(dt, atom, thermo);
  // 先验证固定盒；此处 remap 对已包装质心只改变可能的舍入误差。
  rigid_.remap_centers(box);
  rigid_.compute_forces_and_torques(atom);
  thermo_state_.begin_step(rigid_);
  rigid_integrator::kick_translation(0.5 * dt, atom, rigid_);
  rigid_integrator::drift_translation(dt, atom, rigid_);
  rigid_integrator::kick_rotation(0.5 * dt, rigid_);
  rigid_integrator::drift_rotation(dt, rigid_);
  rigid_.remap_centers(box);
  rigid_.rebuild_positions(atom, box);
  thermo_state_.record_first_half(0.5 * dt, rigid_, atom);
  rigid_.rebuild_velocities(atom);
  step_time_step_ = dt;
  stage_ = Stage::half_step;
}

void Ensemble_Rigid_NVT::compute2(
  double dt, const std::vector<Group>& groups, Box& box, Atom& atom, GPU_Vector<double>& thermo)
{
  if (stage_ != Stage::half_step)
    throw std::logic_error("Rigid NVT compute2 requires compute1 and newly evaluated bead forces.");
  check_step(dt, atom, thermo);
  rigid_.remap_centers(box);
  rigid_.compute_forces_and_torques(atom);
  rigid_integrator::kick_translation(0.5 * dt, atom, rigid_);
  rigid_integrator::kick_rotation(0.5 * dt, rigid_);
  thermo_state_.record_second_half(0.5 * dt, rigid_, atom);
  rigid_.rebuild_velocities(atom);
  find_thermo(
    false,
    box.get_volume(),
    groups,
    atom.mass,
    atom.potential_per_atom,
    atom.velocity_per_atom,
    atom.virial_per_atom,
    thermo);
  thermo_state_.add_constraint_pressure(box.get_volume(), thermo);
  stage_ = Stage::full_step;
}
