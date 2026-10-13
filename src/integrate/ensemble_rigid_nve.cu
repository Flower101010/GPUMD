#include "ensemble_rigid_nve.cuh"
#include "integrate/rigid_body_integrator.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/rigid_body_input.cuh"
#include <limits>
#include <stdexcept>

namespace
{
__global__ void integrate_body_rotation(
  bool is_step1, double dt, RigidBodySystem::DeviceView view)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= view.number_of_bodies)
    return;
  auto& state = view.state[b];
  rigid_math::kick_rotation(state.quat, state.conjqm, view.load[b].torque, 0.5 * dt);
  if (is_step1)
    rigid_math::integrate_free_rotation(state.quat, state.conjqm, view.geometry[b].inertia, dt);
}
} // namespace

Ensemble_Rigid_NVE::Ensemble_Rigid_NVE(int molecule_grouping_method)
  : molecule_grouping_method_(molecule_grouping_method)
{
  if (molecule_grouping_method < 0)
    throw std::invalid_argument("Rigid NVE requires a nonnegative molecule grouping method.");
  type = 0;
  temperature = 0.0;
}

void Ensemble_Rigid_NVE::initialize(
  Atom& atom, const Box& box, const std::vector<Group>& groups)
{
  if (stage_ == Stage::half_step)
    throw std::logic_error("Cannot reinitialize rigid NVE between its two integration stages.");
  stage_ = Stage::uninitialized;
  thermal_dof_ = 0;
  step_time_step_ = 0.0;
  rigid_.clear();
  if (atom.number_of_atoms <= 0)
    throw std::invalid_argument("Rigid NVE requires a nonempty system for temperature output.");
  if (fixed_group >= 0 || move_group >= 0 || deform_x || deform_y || deform_z ||
      deform_xy || deform_xz || deform_yz)
    throw std::invalid_argument("Rigid NVE does not support fixed/moving groups or box deformation.");
  const auto labels = rigid_body_labels_from_group(groups, molecule_grouping_method_, atom.number_of_atoms);
  try {
    rigid_.initialize(atom, box, labels);
    if (rigid_.number_of_bodies() && atom.unwrapped_position.size())
      throw std::invalid_argument("Rigid NVE does not support unwrapped trajectories for rigid members.");
    const long long dof = 3LL * (atom.number_of_atoms - rigid_.number_of_members()) +
                          6LL * rigid_.number_of_bodies();
    if (dof > std::numeric_limits<int>::max())
      throw std::overflow_error("Too many rigid NVE thermal degrees of freedom.");
    rigid_.rebuild_positions(atom, box);
    rigid_.rebuild_velocities(atom);
    thermal_dof_ = static_cast<int>(dof);
    stage_ = Stage::full_step;
  } catch (...) {
    rigid_.clear();
    throw;
  }
}

int Ensemble_Rigid_NVE::thermal_degrees_of_freedom() const { return thermal_dof_; }

void Ensemble_Rigid_NVE::check_step(
  double dt, const Atom& atom, const GPU_Vector<double>& thermo) const
{
  if (!std::isfinite(dt) || dt <= 0.0 || (step_time_step_ != 0.0 && dt != step_time_step_))
    throw std::invalid_argument("Rigid NVE requires one finite positive fixed time step per run.");
  if (fixed_group >= 0 || move_group >= 0 || deform_x || deform_y || deform_z ||
      deform_xy || deform_xz || deform_yz)
    throw std::invalid_argument("Rigid NVE does not support fixed/moving groups or box deformation.");
  const auto view = rigid_.device_view();
  const std::size_t n = view.number_of_atoms;
  if (atom.number_of_atoms != view.number_of_atoms || atom.mass.size() != n ||
      atom.position_per_atom.size() != 3 * n || atom.velocity_per_atom.size() != 3 * n ||
      atom.force_per_atom.size() != 3 * n || atom.potential_per_atom.size() != n ||
      atom.virial_per_atom.size() != 9 * n || thermo.size() < 8)
    throw std::invalid_argument("Rigid NVE requires matching Atom arrays and eight thermo entries.");
  if (view.number_of_bodies && atom.unwrapped_position.size())
    throw std::invalid_argument("Rigid NVE does not support unwrapped trajectories for rigid members.");
}

void Ensemble_Rigid_NVE::integrate_rotation(bool is_step1, double dt)
{
  const auto view = rigid_.device_view();
  if (view.number_of_bodies == 0)
    return;
  integrate_body_rotation<<<(view.number_of_bodies - 1) / 128 + 1, 128>>>(is_step1, dt, view);
  GPU_CHECK_KERNEL
}

void Ensemble_Rigid_NVE::compute1(
  double dt, const std::vector<Group>& groups,
  Box& box, Atom& atom, GPU_Vector<double>& thermo)
{
  if (stage_ != Stage::full_step)
    throw std::logic_error("Rigid NVE compute1 requires initialization and a completed previous step.");
  check_step(dt, atom, thermo);
  // 先验证固定盒；此处 remap 对已包装质心只改变可能的舍入误差。
  rigid_.remap_centers(box);
  rigid_.compute_forces_and_torques(atom);
  rigid_integrator::velocity_verlet(true, dt, atom, rigid_);
  integrate_rotation(true, dt);
  rigid_.remap_centers(box);
  rigid_.rebuild_positions(atom, box);
  rigid_.rebuild_velocities(atom);
  step_time_step_ = dt;
  stage_ = Stage::half_step;
}

void Ensemble_Rigid_NVE::compute2(
  double dt, const std::vector<Group>& groups,
  Box& box, Atom& atom, GPU_Vector<double>& thermo)
{
  if (stage_ != Stage::half_step)
    throw std::logic_error("Rigid NVE compute2 requires compute1 and newly evaluated bead forces.");
  check_step(dt, atom, thermo);
  rigid_.remap_centers(box);
  rigid_.compute_forces_and_torques(atom);
  rigid_integrator::velocity_verlet(false, dt, atom, rigid_);
  integrate_rotation(false, dt);
  rigid_.rebuild_velocities(atom);
  find_thermo(false, box.get_volume(), groups, atom.mass, atom.potential_per_atom,
              atom.velocity_per_atom, atom.virial_per_atom, thermo);
  stage_ = Stage::full_step;
}
