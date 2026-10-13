#include "integrate/rigid_body_integrator.cuh"
#include "model/atom.cuh"
#include <limits>
#include <stdexcept>

namespace rigid_integrator
{
namespace
{
__global__ void integrate_translation(
  bool is_step1,
  double dt,
  RigidBodySystem::DeviceView view,
  const double* mass,
  const double* force,
  double* position,
  double* velocity)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < view.number_of_bodies) {
    velocity_verlet(is_step1, dt, view.geometry[i], view.load[i], view.state[i]);
  } else {
    const int a = i - view.number_of_bodies;
    const int n = view.number_of_atoms;
    if (a >= n || view.atom_to_body[a] != -1)
      return;
    Vec3 x{position[a], position[a + n], position[a + 2 * n]};
    Vec3 v{velocity[a], velocity[a + n], velocity[a + 2 * n]};
    const Vec3 f{force[a], force[a + n], force[a + 2 * n]};
    velocity_verlet(is_step1, dt, mass[a], f, x, v);
    if (is_step1) {
      position[a] = x.x;
      position[a + n] = x.y;
      position[a + 2 * n] = x.z;
    }
    velocity[a] = v.x;
    velocity[a + n] = v.y;
    velocity[a + 2 * n] = v.z;
  }
}
} // namespace

void velocity_verlet(bool is_step1, double dt, Atom& atom, RigidBodySystem& rigid)
{
  if (!std::isfinite(dt))
    throw std::invalid_argument("Rigid translation requires a finite time step.");
  const auto view = rigid.device_view(); // also requires completed initialization
  const int n = view.number_of_atoms;
  const std::size_t vector_size = 3 * static_cast<std::size_t>(n);
  if (atom.number_of_atoms != n || atom.mass.size() != static_cast<std::size_t>(n) ||
      atom.position_per_atom.size() != vector_size ||
      atom.velocity_per_atom.size() != vector_size || atom.force_per_atom.size() != vector_size)
    throw std::invalid_argument("Rigid translation requires matching Atom arrays.");
  // Leave room for the padded final block in the kernel's signed int index.
  if (n > std::numeric_limits<int>::max() - view.number_of_bodies - 127)
    throw std::overflow_error("Too many translation entries for the GPU launch.");
  const int entries = n + view.number_of_bodies;
  if (entries == 0)
    return;
  integrate_translation<<<(entries - 1) / 128 + 1, 128>>>(
    is_step1, dt, view, atom.mass.data(), atom.force_per_atom.data(),
    atom.position_per_atom.data(), atom.velocity_per_atom.data());
  GPU_CHECK_KERNEL
}
} // namespace rigid_integrator
