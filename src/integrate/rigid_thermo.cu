#include "integrate/rigid_thermo.cuh"
#include "model/atom.cuh"
#include <limits>
#include <stdexcept>

namespace
{
// 小体系采用一个 block；每个线程跨步累加，固定次序归约，无 double 原子操作。计算的是两倍平动和转动动能
__global__ void reduce_kinetic_energy(
  RigidBodySystem::ConstDeviceView view, const double* mass, const double* velocity,
  double* result)
{
  __shared__ double translation[128];
  __shared__ double rotation[128];
  const int t = threadIdx.x;
  double kt = 0, kr = 0;
  for (std::size_t b = t; b < static_cast<std::size_t>(view.number_of_bodies); b += blockDim.x) {
    const auto& s = view.state[b];
    const auto& g = view.geometry[b];
    kt += g.mass * rigid_math::norm_squared(s.vcm);
    const Vec3 momentum = rigid_math::conjqm_to_body_angular_momentum(s.quat, s.conjqm);
    kr += rigid_math::dot(momentum, rigid_math::apply_inverse_inertia(momentum, g.inertia));
  }
  const std::size_t n = view.number_of_atoms;
  for (std::size_t i = t; i < n; i += blockDim.x) {
    if (view.atom_to_body[i] != -1)
      continue;
    const Vec3 v{velocity[i], velocity[i + n], velocity[i + 2 * n]};
    kt += mass[i] * rigid_math::norm_squared(v);
  }
  translation[t] = kt;
  rotation[t] = kr;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride /= 2) {
    if (t < stride) {
      translation[t] += translation[t + stride];
      rotation[t] += rotation[t + stride];
    }
    __syncthreads();
  }
  if (t == 0) {
    result[0] = translation[0];
    result[1] = rotation[0];
  }
}

__global__ void capture_body_positions(RigidBodySystem::ConstDeviceView view, Quaternion* previous)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < view.number_of_bodies)
    previous[b] = view.state[b].quat;
}

// LAMMPS FixRigid::set_xv/set_v 的离散冲量定义：
// fc=m*(v_after-v_before)/(dt/2)-f；每阶段计入 1/2 r⊗fc。
// 对自由刚体每体 sum(fc)=0，因此总 virial 可用相对质心坐标，避免跨盒历史。
// 必须在 rebuild_velocities 覆盖旧 bead 速度之前调用。
__global__ void tally_body_constraint_virial(
  bool is_step1,
  double h,
  RigidBodySystem::ConstDeviceView view,
  const Quaternion* previous_quats,
  const double* mass,
  const double* force,
  const double* velocity_before,
  double* virial)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= view.number_of_bodies)
    return;
  const auto& g = view.geometry[b];
  const auto& s = view.state[b];
  const Vec3 momentum = rigid_math::conjqm_to_body_angular_momentum(s.quat, s.conjqm);
  const Vec3 omega = rigid_math::rotate(
    s.quat, rigid_math::apply_inverse_inertia(momentum, g.inertia));
  const Quaternion q_position = is_step1 ? previous_quats[b] : s.quat;
  const std::size_t n = view.number_of_atoms;
  double w[6] = {0, 0, 0, 0, 0, 0};
  for (int j = g.first_member; j < g.first_member + g.natoms; ++j) {
    const int i = view.member_atoms[j];
    const Vec3 r_after = rigid_math::rotate(s.quat, view.member_displace[j]);
    const Vec3 v_after = rigid_math::add(s.vcm, rigid_math::cross(omega, r_after));
    const Vec3 fc{
      mass[i] * (v_after.x - velocity_before[i]) / h - force[i],
      mass[i] * (v_after.y - velocity_before[i + n]) / h - force[i + n],
      mass[i] * (v_after.z - velocity_before[i + 2 * n]) / h - force[i + 2 * n]};
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

__global__ void
add_constraint_pressure(int bodies, const double* virial, double volume, double* thermo)
{
  const int c = threadIdx.x;
  if (c >= 6)
    return;
  double sum = 0.0;
  for (int b = 0; b < bodies; ++b)
    sum += virial[c * bodies + b];
  thermo[2 + c] += sum / volume;
}

__global__ void instantaneous_constraint_virial(
  RigidBodySystem::ConstDeviceView view, const double* mass, const double* force, double* virial)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= view.number_of_bodies) return;
  using namespace rigid_math;
  const auto& g = view.geometry[b];
  const auto& s = view.state[b];
  const auto& load = view.load[b];
  const Vec3 angular_momentum = conjqm_to_body_angular_momentum(s.quat, s.conjqm);
  const Vec3 omega_body = apply_inverse_inertia(angular_momentum, g.inertia);
  const Vec3 torque = subtract(inverse_rotate(s.quat, load.torque), cross(omega_body, angular_momentum));
  const Vec3 alpha = rotate(s.quat, apply_inverse_inertia(torque, g.inertia));
  const Vec3 omega = rotate(s.quat, omega_body);
  const Vec3 acceleration_cm = scale(load.fcm, 1 / g.mass);
  const std::size_t n = view.number_of_atoms;
  double w[6]{};
  for (int j = g.first_member; j < g.first_member + g.natoms; ++j) {
    const int i = view.member_atoms[j];
    const Vec3 r = rotate(s.quat, view.member_displace[j]);
    const Vec3 acceleration = add(acceleration_cm, add(cross(alpha, r), cross(omega, cross(omega, r))));
    const Vec3 fc = subtract(scale(acceleration, mass[i]), {force[i], force[i + n], force[i + 2 * n]});
    w[0] += r.x * fc.x; w[1] += r.y * fc.y; w[2] += r.z * fc.z;
    w[3] += 0.5 * (r.x * fc.y + r.y * fc.x);
    w[4] += 0.5 * (r.x * fc.z + r.z * fc.x);
    w[5] += 0.5 * (r.y * fc.z + r.z * fc.y);
  }
  for (int c = 0; c < 6; ++c) virial[c * view.number_of_bodies + b] = w[c];
}
} // namespace

void RigidThermo::clear()
{
  previous_quats_.clear();
  constraint_virial_.clear();
  kinetic_sample_.clear();
  atoms_ = bodies_ = members_ = degrees_of_freedom_ = rotational_degrees_of_freedom_ = 0;
  half_step_ = 0;
  stage_ = Stage::uninitialized;
}

void RigidThermo::initialize(const RigidBodySystem& rigid)
{
  clear();
  const auto view = rigid.device_view();
  const long long translation = 3LL * (view.number_of_atoms - view.number_of_members + view.number_of_bodies);
  const long long rotation = 3LL * view.number_of_bodies - rigid.number_of_dimers();
  const long long dof = translation + rotation;
  if (dof > std::numeric_limits<int>::max())
    throw std::overflow_error("Too many rigid thermal degrees of freedom.");
  previous_quats_.resize(view.number_of_bodies);
  constraint_virial_.resize(6 * static_cast<std::size_t>(view.number_of_bodies));
  constraint_virial_.fill(0.0);
  kinetic_sample_.resize(2);
  atoms_ = view.number_of_atoms;
  bodies_ = view.number_of_bodies;
  members_ = view.number_of_members;
  degrees_of_freedom_ = static_cast<int>(dof);
  rotational_degrees_of_freedom_ = static_cast<int>(rotation);
  stage_ = Stage::ready;
}

RigidBodySystem::ConstDeviceView RigidThermo::checked_view(const RigidBodySystem& rigid) const
{
  if (stage_ == Stage::uninitialized)
    throw std::logic_error("Rigid thermo must be initialized.");
  const auto view = rigid.device_view();
  if (
    view.number_of_atoms != atoms_ || view.number_of_bodies != bodies_ ||
    view.number_of_members != members_)
    throw std::invalid_argument("Rigid topology changed; reinitialize thermo.");
  return view;
}

RigidThermo::KineticSample RigidThermo::sample_kinetic_energy(
  const RigidBodySystem& rigid, const Atom& atom)
{
  const auto view = checked_view(rigid);
  const std::size_t n = view.number_of_atoms;
  if (atom.number_of_atoms != view.number_of_atoms || atom.mass.size() != n ||
      atom.velocity_per_atom.size() != 3 * n)
    throw std::invalid_argument("Rigid kinetic sampling requires matching masses and velocities.");
  reduce_kinetic_energy<<<1, 128>>>(view, atom.mass.data(), atom.velocity_per_atom.data(), kinetic_sample_.data());
  GPU_CHECK_KERNEL
  double sample[2];
  kinetic_sample_.copy_to_host(sample);
  return {sample[0], sample[1]};
}

void RigidThermo::begin_step(const RigidBodySystem& rigid)
{
  const auto view = checked_view(rigid);
  if (stage_ != Stage::ready && stage_ != Stage::complete)
    throw std::logic_error("Finish both constraint samples before starting another step.");
  if (bodies_) {
    capture_body_positions<<<(bodies_ - 1) / 128 + 1, 128>>>(view, previous_quats_.data());
    GPU_CHECK_KERNEL
  }
  half_step_ = 0;
  stage_ = Stage::step_started;
}

void RigidThermo::check_sample(double h, const RigidBodySystem& rigid, const Atom& atom) const
{
  checked_view(rigid);
  if (!std::isfinite(h) || h <= 0)
    throw std::invalid_argument("Constraint virial requires a finite positive half step.");
  const std::size_t n = atoms_;
  if (
    atom.number_of_atoms != atoms_ || atom.mass.size() != n ||
    atom.force_per_atom.size() != 3 * n || atom.velocity_per_atom.size() != 3 * n)
    throw std::invalid_argument("Constraint virial requires matching Atom arrays.");
}

void RigidThermo::record_first_half(double h, const RigidBodySystem& rigid, const Atom& atom)
{
  if (stage_ != Stage::step_started)
    throw std::logic_error("Capture the old pose before the first constraint sample.");
  check_sample(h, rigid, atom);
  if (bodies_) {
    tally_body_constraint_virial<<<(bodies_ - 1) / 128 + 1, 128>>>(
      true,
      h,
      rigid.device_view(),
      previous_quats_.data(),
      atom.mass.data(),
      atom.force_per_atom.data(),
      atom.velocity_per_atom.data(),
      constraint_virial_.data());
    GPU_CHECK_KERNEL
  }
  half_step_ = h;
  stage_ = Stage::first_half_recorded;
}

void RigidThermo::record_second_half(double h, const RigidBodySystem& rigid, const Atom& atom)
{
  if (stage_ != Stage::first_half_recorded)
    throw std::logic_error("Record the first constraint sample before the second.");
  check_sample(h, rigid, atom);
  if (h != half_step_)
    throw std::invalid_argument("Both constraint samples must use the same half step.");
  if (bodies_) {
    tally_body_constraint_virial<<<(bodies_ - 1) / 128 + 1, 128>>>(
      false,
      h,
      rigid.device_view(),
      previous_quats_.data(),
      atom.mass.data(),
      atom.force_per_atom.data(),
      atom.velocity_per_atom.data(),
      constraint_virial_.data());
    GPU_CHECK_KERNEL
  }
  stage_ = Stage::complete;
}

void RigidThermo::add_constraint_pressure(double volume, GPU_Vector<double>& thermo) const
{
  if (stage_ != Stage::complete)
    throw std::logic_error("Constraint pressure requires both completed half steps.");
  if (!std::isfinite(volume) || volume <= 0 || thermo.size() < 8)
    throw std::invalid_argument(
      "Constraint pressure requires positive volume and eight thermo entries.");
  if (!bodies_)
    return;
  ::add_constraint_pressure<<<1, 32>>>(bodies_, constraint_virial_.data(), volume, thermo.data());
  GPU_CHECK_KERNEL
}

void RigidThermo::sample_constraint_virial(const RigidBodySystem& rigid, const Atom& atom)
{
  const auto view = checked_view(rigid);
  if (stage_ != Stage::ready && stage_ != Stage::complete)
    throw std::logic_error("Cannot replace an unfinished constraint virial window.");
  const std::size_t n = atoms_;
  if (atom.number_of_atoms != atoms_ || atom.mass.size() != n || atom.force_per_atom.size() != 3 * n)
    throw std::invalid_argument("Instantaneous constraint virial requires matching masses and forces.");
  if (bodies_) {
    instantaneous_constraint_virial<<<(bodies_ - 1) / 128 + 1, 128>>>(
      view, atom.mass.data(), atom.force_per_atom.data(), constraint_virial_.data());
    GPU_CHECK_KERNEL
  }
  stage_ = Stage::complete;
}
