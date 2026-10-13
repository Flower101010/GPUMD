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
  bool is_step1, double dt, RigidBodySystem::DeviceView view, Quaternion* previous_quats)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= view.number_of_bodies)
    return;
  auto& state = view.state[b];
  if (is_step1)
    previous_quats[b] = state.quat;
  rigid_math::kick_rotation(state.quat, state.conjqm, view.load[b].torque, 0.5 * dt);
  if (is_step1)
    rigid_math::integrate_free_rotation(state.quat, state.conjqm, view.geometry[b].inertia, dt);
}

// LAMMPS FixRigid::set_xv/set_v 的离散冲量定义：
// fc=m*(v_after-v_before)/(dt/2)-f；每阶段计入 1/2 r⊗fc。
// 对自由刚体每体 sum(fc)=0，因此总 virial 可用相对质心坐标，避免跨盒历史。
// 必须在 rebuild_velocities 覆盖旧 bead 速度之前调用。
__global__ void tally_body_constraint_virial(
  bool is_step1, double dt, RigidBodySystem::DeviceView view,
  const Quaternion* previous_quats, const double* mass, const double* force,
  const double* velocity_before, double* virial)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= view.number_of_bodies)
    return;
  const auto& g = view.geometry[b];
  const auto& s = view.state[b];
  const Quaternion qp = rigid_math::multiply(rigid_math::conjugate(s.quat), s.conjqm);
  const Vec3 omega = rigid_math::rotate(
    s.quat, {0.5 * qp.x / g.inertia.x, 0.5 * qp.y / g.inertia.y, 0.5 * qp.z / g.inertia.z});
  const Quaternion q_position = is_step1 ? previous_quats[b] : s.quat;
  const int n = view.number_of_atoms;
  double w[6] = {0, 0, 0, 0, 0, 0};
  for (int j = g.first_member; j < g.first_member + g.natoms; ++j) {
    const int i = view.member_atoms[j];
    const Vec3 r_after = rigid_math::rotate(s.quat, view.member_displace[j]);
    const Vec3 v_after = rigid_math::add(s.vcm, rigid_math::cross(omega, r_after));
    const Vec3 fc{
      mass[i] * (v_after.x - velocity_before[i]) / (0.5 * dt) - force[i],
      mass[i] * (v_after.y - velocity_before[i + n]) / (0.5 * dt) - force[i + n],
      mass[i] * (v_after.z - velocity_before[i + 2 * n]) / (0.5 * dt) - force[i + 2 * n]};
    const Vec3 r = rigid_math::rotate(q_position, view.member_displace[j]);
    w[0] += 0.5 * r.x * fc.x;
    w[1] += 0.5 * r.y * fc.y;
    w[2] += 0.5 * r.z * fc.z;
    w[3] += 0.5 * r.x * fc.y;
    w[4] += 0.5 * r.x * fc.z;
    w[5] += 0.5 * r.y * fc.z;
  }
  for (int c = 0; c < 6; ++c) {
    const int k = c * view.number_of_bodies + b;
    if (is_step1)
      virial[k] = w[c]; // 每步重新开始，不继承上一步或初始估算
    else
      virial[k] += w[c];
  }
}

__global__ void add_constraint_pressure(int bodies, const double* virial, double volume, double* thermo)
{
  const int c = threadIdx.x;
  if (c >= 6)
    return;
  double sum = 0.0;
  for (int b = 0; b < bodies; ++b)
    sum += virial[c * bodies + b];
  thermo[2 + c] += sum / volume;
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
  previous_quats_.clear();
  constraint_virial_.clear();
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
    const int bodies = rigid_.number_of_bodies();
    previous_quats_.resize(bodies);
    constraint_virial_.resize(6 * static_cast<std::size_t>(bodies));
    if (bodies)
      constraint_virial_.fill(0.0);
    thermal_dof_ = static_cast<int>(dof);
    stage_ = Stage::full_step;
  } catch (...) {
    rigid_.clear();
    previous_quats_.clear();
    constraint_virial_.clear();
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
  integrate_body_rotation<<<(view.number_of_bodies - 1) / 128 + 1, 128>>>(
    is_step1, dt, view, previous_quats_.data());
  GPU_CHECK_KERNEL
}

void Ensemble_Rigid_NVE::tally_constraint_virial(bool is_step1, double dt, const Atom& atom)
{
  const auto view = rigid_.device_view();
  if (view.number_of_bodies == 0)
    return;
  tally_body_constraint_virial<<<(view.number_of_bodies - 1) / 128 + 1, 128>>>(
    is_step1, dt, view, previous_quats_.data(), atom.mass.data(), atom.force_per_atom.data(),
    atom.velocity_per_atom.data(), constraint_virial_.data());
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
  tally_constraint_virial(true, dt, atom);
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
  tally_constraint_virial(false, dt, atom);
  rigid_.rebuild_velocities(atom);
  find_thermo(false, box.get_volume(), groups, atom.mass, atom.potential_per_atom,
              atom.velocity_per_atom, atom.virial_per_atom, thermo);
  if (rigid_.number_of_bodies()) {
    add_constraint_pressure<<<1, 32>>>(
      rigid_.number_of_bodies(), constraint_virial_.data(), box.get_volume(), thermo.data());
    GPU_CHECK_KERNEL
  }
  stage_ = Stage::full_step;
}
