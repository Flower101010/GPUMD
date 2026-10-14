#include "integrate/rigid_body_integrator.cuh"
#include "model/atom.cuh"
#include <limits>
#include <stdexcept>

namespace rigid_integrator
{
namespace
{
__global__ void scale_momenta_kernel(
  double factor, RigidBodySystem::DeviceView view, double* velocity)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < view.number_of_bodies) {
    auto& state = view.state[i];
    state.vcm = rigid_math::scale(state.vcm, factor);
    state.conjqm.w *= factor;
    state.conjqm.x *= factor;
    state.conjqm.y *= factor;
    state.conjqm.z *= factor;
    return;
  }
  const int a = i - view.number_of_bodies;
  if (a >= view.number_of_atoms || view.atom_to_body[a] != -1)
    return;
  const std::size_t n = view.number_of_atoms, index = a;
  velocity[index] *= factor;
  velocity[index + n] *= factor;
  velocity[index + 2 * n] *= factor;
}

// 编译期选择操作，避免重复 kernel 或在公开接口中用 bool 混合 kick/drift。
template <bool Kick, bool Drift>
__global__ void integrate_translation(
  double kick_time,
  double drift_time,
  RigidBodySystem::DeviceView view,
  const double* mass,
  const double* force,
  double* position,
  double* velocity)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < view.number_of_bodies) {
    auto& state = view.state[i];
    if constexpr (Kick)
      kick_translation(kick_time, view.geometry[i].mass, view.load[i].fcm, state.vcm);
    if constexpr (Drift)
      drift_translation(drift_time, state.xcm, state.vcm);
    return;
  }
  const int a = i - view.number_of_bodies;
  if (a >= view.number_of_atoms || view.atom_to_body[a] != -1)
    return;
  const std::size_t n = view.number_of_atoms, index = a;
  Vec3 v{velocity[index], velocity[index + n], velocity[index + 2 * n]};
  if constexpr (Kick) {
    const Vec3 f{force[index], force[index + n], force[index + 2 * n]};
    kick_translation(kick_time, mass[index], f, v);
    velocity[index] = v.x;
    velocity[index + n] = v.y;
    velocity[index + 2 * n] = v.z;
  }
  if constexpr (Drift) {
    Vec3 x{position[index], position[index + n], position[index + 2 * n]};
    drift_translation(drift_time, x, v);
    position[index] = x.x;
    position[index + n] = x.y;
    position[index + 2 * n] = x.z;
  }
}

template <bool Kick, bool Drift>
void launch_translation(double kick_time, double drift_time, Atom& atom, RigidBodySystem& rigid)
{
  if (!std::isfinite(kick_time) || !std::isfinite(drift_time))
    throw std::invalid_argument("Rigid translation requires a finite operation time.");
  const auto view = rigid.device_view();
  const std::size_t n = view.number_of_atoms;
  if (atom.number_of_atoms != view.number_of_atoms || atom.velocity_per_atom.size() != 3 * n)
    throw std::invalid_argument("Rigid translation requires matching Atom velocities.");
  if constexpr (Kick) {
    if (atom.mass.size() != n || atom.force_per_atom.size() != 3 * n)
      throw std::invalid_argument("Rigid kick requires matching Atom masses and forces.");
  }
  if constexpr (Drift) {
    if (atom.position_per_atom.size() != 3 * n)
      throw std::invalid_argument("Rigid drift requires matching Atom positions.");
  }
  if (view.number_of_atoms > std::numeric_limits<int>::max() - view.number_of_bodies - 127)
    throw std::overflow_error("Too many translation entries for the GPU launch.");
  const int entries = view.number_of_atoms + view.number_of_bodies;
  if (entries == 0)
    return;
  integrate_translation<Kick, Drift><<<(entries - 1) / 128 + 1, 128>>>(
    kick_time,
    drift_time,
    view,
    atom.mass.data(),
    atom.force_per_atom.data(),
    atom.position_per_atom.data(),
    atom.velocity_per_atom.data());
  GPU_CHECK_KERNEL
}

template <bool Kick>
__global__ void integrate_rotation(double h, RigidBodySystem::DeviceView view)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= view.number_of_bodies)
    return;
  auto& state = view.state[b];
  if constexpr (Kick)
    rigid_math::kick_rotation(state.quat, state.conjqm, view.load[b].torque, h);
  else
    rigid_math::integrate_free_rotation(state.quat, state.conjqm, view.geometry[b].inertia, h);
}

template <bool Kick>
void launch_rotation(double h, RigidBodySystem& rigid)
{
  if (!std::isfinite(h))
    throw std::invalid_argument("Rigid rotation requires a finite operation time.");
  const auto view = rigid.device_view();
  if (view.number_of_bodies == 0)
    return;
  integrate_rotation<Kick><<<(view.number_of_bodies - 1) / 128 + 1, 128>>>(h, view);
  GPU_CHECK_KERNEL
}
} // namespace

void scale_momenta(double factor, Atom& atom, RigidBodySystem& rigid)
{
  if (!std::isfinite(factor) || factor <= 0.0)
    throw std::invalid_argument("Rigid momentum scaling requires a finite positive factor.");
  const auto view = rigid.device_view();
  const std::size_t n = view.number_of_atoms;
  if (atom.number_of_atoms != view.number_of_atoms || atom.velocity_per_atom.size() != 3 * n)
    throw std::invalid_argument("Rigid momentum scaling requires matching Atom velocities.");
  if (view.number_of_atoms > std::numeric_limits<int>::max() - view.number_of_bodies - 127)
    throw std::overflow_error("Too many momentum entries for the GPU launch.");
  const int entries = view.number_of_atoms + view.number_of_bodies;
  if (entries == 0)
    return;
  scale_momenta_kernel<<<(entries - 1) / 128 + 1, 128>>>(
    factor, view, atom.velocity_per_atom.data());
  GPU_CHECK_KERNEL
}

void kick_translation(double h, Atom& atom, RigidBodySystem& rigid)
{
  launch_translation<true, false>(h, 0, atom, rigid);
}
void drift_translation(double h, Atom& atom, RigidBodySystem& rigid)
{
  launch_translation<false, true>(0, h, atom, rigid);
}
void kick_rotation(double h, RigidBodySystem& rigid) { launch_rotation<true>(h, rigid); }
void drift_rotation(double h, RigidBodySystem& rigid) { launch_rotation<false>(h, rigid); }
void velocity_verlet(bool is_step1, double dt, Atom& atom, RigidBodySystem& rigid)
{
  if (is_step1)
    launch_translation<true, true>(0.5 * dt, dt, atom, rigid);
  else
    launch_translation<true, false>(0.5 * dt, 0, atom, rigid);
}
} // namespace rigid_integrator
