#include "ensemble_rigid_nvt.cuh"
#include "integrate/rigid_body_integrator.cuh"
#include "integrate/nose_hoover_chain.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/rigid_body_input.cuh"
#include "utilities/common.cuh"
#include <stdexcept>

Ensemble_Rigid_NVT::Ensemble_Rigid_NVT(
  int molecule_grouping_method, double target_temperature, double coupling, double dt)
  : molecule_grouping_method_(molecule_grouping_method), thermostat_time_step_(dt)
{
  if (molecule_grouping_method < 0)
    throw std::invalid_argument("Rigid NVT requires a nonnegative molecule grouping method.");
  if (!std::isfinite(target_temperature) || target_temperature <= 0.0 ||
      !std::isfinite(coupling) || coupling < 1.0 || !std::isfinite(dt) || dt <= 0.0)
    throw std::invalid_argument("Rigid NVT requires positive temperature/time step and coupling >= 1.");
  type = 2;
  temperature = target_temperature;
  temperature_coupling = coupling;
}

void Ensemble_Rigid_NVT::initialize(Atom& atom, const Box& box, const std::vector<Group>& groups)
{
  if (stage_ == Stage::half_step)
    throw std::logic_error("Cannot reinitialize rigid NVT between its two integration stages.");
  stage_ = Stage::uninitialized;
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
    const double tau = thermostat_time_step_ * temperature_coupling;
    const double chain_mass = K_B * temperature * tau * tau;
    const double first_mass = thermal_degrees_of_freedom() * chain_mass;
    if (!std::isfinite(chain_mass) || chain_mass <= 0.0 ||
        !std::isfinite(first_mass) || first_mass <= 0.0)
      throw std::invalid_argument("Rigid NVT thermostat masses must be finite and positive.");
    for (int i = 0; i < NOSE_HOOVER_CHAIN_LENGTH; ++i) {
      pos_nhc1[i] = 0.0;
      vel_nhc1[i] = i % 2 == 0 ? 1.0 : -1.0;
      mas_nhc1[i] = i == 0 ? first_mass : chain_mass;
    }
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
  if (!std::isfinite(dt) || dt <= 0.0 || dt != thermostat_time_step_)
    throw std::invalid_argument("Rigid NVT requires one finite positive fixed time step per run.");
  if (!std::isfinite(temperature) || temperature <= 0.0)
    throw std::invalid_argument("Rigid NVT requires a finite positive target temperature.");
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

void Ensemble_Rigid_NVT::thermostat_half_step(
  double duration, const std::vector<Group>& groups, const Box& box, Atom& atom,
  GPU_Vector<double>& thermo)
{
  // bead v 已与刚体状态一致，其动能恰为自由粒子 + 质心 + 转动动能。
  find_thermo(
    true, box.get_volume(), groups, atom.mass, atom.potential_per_atom,
    atom.velocity_per_atom, atom.virial_per_atom, thermo);
  double instantaneous_temperature;
  thermo.copy_to_host(&instantaneous_temperature, 1);
  const double dof = thermal_degrees_of_freedom();
  const double twice_kinetic_energy = instantaneous_temperature * dof * K_B;
  const double factor = nose_hoover_chain::propagate(
    NOSE_HOOVER_CHAIN_LENGTH, pos_nhc1, vel_nhc1, mas_nhc1,
    twice_kinetic_energy, K_B * temperature, dof, duration);
  rigid_integrator::scale_momenta(factor, atom, rigid_);
  rigid_.rebuild_velocities(atom);
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
  // 恒温缩放必须在约束冲量统计窗口外；重建后的 bead v 才是物理 kick 的起点。
  thermostat_half_step(0.5 * dt, groups, box, atom, thermo);
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
  thermostat_half_step(0.5 * dt, groups, box, atom, thermo);
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
