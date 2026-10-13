#include "integrate/ensemble_rigid_nve.cuh"
#include "integrate/rigid_body_integrator.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace rigid_math;
static_assert(std::is_base_of<Ensemble, Ensemble_Rigid_NVE>::value, "Ensemble interface");
static_assert(!std::is_abstract<Ensemble_Rigid_NVE>::value, "Both stages declared");

static void near(double a, double b)
{
  assert(std::abs(a - b) < 2e-11 * (1 + std::abs(b)));
}
static void near(Vec3 a, Vec3 b) { near(a.x, b.x); near(a.y, b.y); near(a.z, b.z); }
static void near(Quaternion a, Quaternion b)
{
  near(a.w, b.w); near(a.x, b.x); near(a.y, b.y); near(a.z, b.z);
}
static void upload(GPU_Vector<double>& a, const std::vector<double>& x)
{
  a.resize(x.size());
  if (!x.empty()) a.copy_from_host(x.data());
}
static std::vector<double> download(const GPU_Vector<double>& a)
{
  std::vector<double> x(a.size());
  if (!x.empty()) a.copy_to_host(x.data());
  return x;
}
static std::vector<double> soa(const std::vector<Vec3>& x)
{
  std::vector<double> a(3 * x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    a[i] = x[i].x; a[i + x.size()] = x[i].y; a[i + 2 * x.size()] = x[i].z;
  }
  return a;
}
static Vec3 read(const std::vector<double>& a, int i, int n)
{
  return {a[i], a[i + n], a[i + 2 * n]};
}
template<class Exception, class F> static void rejects(F f)
{
  bool caught = false;
  try { f(); } catch (const Exception&) { caught = true; }
  assert(caught);
}

// Analytic constant acceleration, interleaved membership, distinct stage-two
// force, and an ordinary singleton whose COM velocity must remain nonzero.
static void gpu_case(int n, bool with_body)
{
  Atom atom;
  atom.number_of_atoms = n;
  Box box{};
  box.cpu_h[0] = box.cpu_h[4] = box.cpu_h[8] = 1000;
  box.cpu_h[9] = box.cpu_h[13] = box.cpu_h[17] = 0.001;
  std::vector<int> labels(n, -1);
  std::vector<double> mass(n, 18);
  std::vector<Vec3> x(n), v(n, {0.3, -0.2, 0.4});
  const Vec3 acceleration{0.2, -0.4, 0.6}, next_acceleration{-0.6, 0.8, -0.2};
  for (int i = 0; i < n; ++i) x[i] = {5.0 + i, 8.0, 7.0};
  if (with_body) {
    const int b = n == 3 ? 1 : 2;
    const int c = n == 3 ? 2 : 3;
    labels[0] = labels[b] = labels[c] = 9;
    mass[0] = 8; mass[b] = 12; mass[c] = 16;
    x[0] = {2, 2, 2}; x[b] = {3.2, 2, 2}; x[c] = {2.2, 3.1, 2};
  }
  std::vector<Vec3> force(n);
  for (int i = 0; i < n; ++i) force[i] = scale(acceleration, mass[i]);
  upload(atom.mass, mass); upload(atom.position_per_atom, soa(x));
  upload(atom.velocity_per_atom, soa(v)); upload(atom.force_per_atom, soa(force));
  RigidBodySystem rigid;
  rigid.initialize(atom, box, labels);
  rigid.compute_forces_and_torques(atom);
  RigidBodySystem::HostSnapshot before, half, after;
  rigid.copy_to_host(before);
  const double dt = 0.125;
  rigid_integrator::velocity_verlet(true, dt, atom, rigid);
  rigid.copy_to_host(half);
  const auto xhalf = download(atom.position_per_atom);
  const auto vhalf = download(atom.velocity_per_atom);
  for (int i = 0; i < n; ++i) {
    if (labels[i] < 0) {
      near(read(xhalf, i, n), add(x[i], add(scale(v[i], dt), scale(acceleration, 0.5*dt*dt))));
      near(read(vhalf, i, n), add(v[i], scale(acceleration, 0.5*dt)));
    } else {
      near(read(xhalf, i, n), x[i]); near(read(vhalf, i, n), v[i]);
    }
  }
  for (std::size_t b = 0; b < before.bodies.size(); ++b) {
    const auto& s = before.bodies[b].state;
    near(half.bodies[b].state.xcm,
         add(s.xcm, add(scale(s.vcm, dt), scale(acceleration, 0.5*dt*dt))));
    near(half.bodies[b].state.vcm, add(s.vcm, scale(acceleration, 0.5*dt)));
    near(half.bodies[b].state.quat, s.quat);
    near(half.bodies[b].state.conjqm, s.conjqm);
  }
  for (int i = 0; i < n; ++i) force[i] = scale(next_acceleration, mass[i]);
  upload(atom.force_per_atom, soa(force));
  rigid.compute_forces_and_torques(atom);
  rigid_integrator::velocity_verlet(false, dt, atom, rigid);
  rigid.copy_to_host(after);
  const auto xafter = download(atom.position_per_atom);
  const auto vafter = download(atom.velocity_per_atom);
  assert(xafter == xhalf); // stage two must not drift either representation
  for (int i = 0; i < n; ++i)
    near(read(vafter, i, n), labels[i] < 0 ?
         add(v[i], scale(add(acceleration, next_acceleration), 0.5*dt)) : v[i]);
  for (std::size_t b = 0; b < before.bodies.size(); ++b) {
    near(after.bodies[b].state.xcm, half.bodies[b].state.xcm);
    near(after.bodies[b].state.vcm,
         add(before.bodies[b].state.vcm, scale(add(acceleration, next_acceleration), 0.5*dt)));
    near(after.bodies[b].state.quat, before.bodies[b].state.quat);
    near(after.bodies[b].state.conjqm, before.bodies[b].state.conjqm);
  }
  rejects<std::invalid_argument>([&] {
    rigid_integrator::velocity_verlet(true, std::numeric_limits<double>::infinity(), atom, rigid);
  });
  atom.force_per_atom.resize(1);
  rejects<std::invalid_argument>([&] { rigid_integrator::velocity_verlet(true, dt, atom, rigid); });
}

// An independently specified harmonic oscillator tests full-step reversibility
// and the determinant-one canonical map, rather than merely repeating a kick.
static void host_case()
{
  const double mass = 2.5, spring = 3.0, dt = 0.07;
  auto step = [&](Vec3& x, Vec3& v, double h) {
    rigid_integrator::velocity_verlet(true, h, mass, scale(x, -spring), x, v);
    rigid_integrator::velocity_verlet(false, h, mass, scale(x, -spring), x, v);
  };
  Vec3 x{0.7, -0.2, 0.9}, v{-0.6, 0.1, 0.4};
  const Vec3 x0 = x, v0 = v;
  step(x, v, dt); step(x, v, -dt);
  near(x, x0); near(v, v0);
  Vec3 a{1, 0, 0}, av{0, 0, 0}, b{0, 0, 0}, bv{1/mass, 0, 0};
  step(a, av, dt); step(b, bv, dt);
  near(a.x * mass * bv.x - b.x * mass * av.x, 1.0);
}

int main()
{
  host_case();
  int devices = 0;
  if (gpuGetDeviceCount(&devices) != gpuSuccess || !devices) {
    std::cout << "SKIP: no accessible GPU\n"; return 0;
  }
  gpu_case(4, true);   // single free bead interleaved with a triangle
  gpu_case(3, true);   // all rigid
  gpu_case(257, false); // all free, multiple blocks
  gpu_case(0, false); // empty initialized system, no launch
  Atom atom;
  RigidBodySystem uninitialized;
  rejects<std::logic_error>([&] { rigid_integrator::velocity_verlet(true, 0.1, atom, uninitialized); });
  std::cout << "Rigid/free translation checks passed.\n";
}
