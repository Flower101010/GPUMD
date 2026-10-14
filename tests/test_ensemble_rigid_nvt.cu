#include "integrate/ensemble_nhc.cuh"
#include "integrate/ensemble_rigid_nve.cuh"
#include "integrate/ensemble_rigid_nvt.cuh"
#include "integrate/nose_hoover_chain.cuh"
#include "utilities/common.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
constexpr double dt = 0.002;
constexpr double target = 300;
constexpr double coupling = 500;

void near(double a, double b)
{
  assert(std::isfinite(a) && std::isfinite(b));
  assert(std::abs(a - b) < 2e-10 * (1 + std::abs(b)));
}
void near(const std::vector<double>& a, const std::vector<double>& b)
{
  assert(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i)
    near(a[i], b[i]);
}
void upload(GPU_Vector<double>& a, const std::vector<double>& values)
{
  a.resize(values.size());
  a.copy_from_host(values.data());
}
std::vector<double> download(const GPU_Vector<double>& a)
{
  std::vector<double> result(a.size());
  a.copy_to_host(result.data());
  return result;
}
double twice_kinetic(const Atom& atom)
{
  const auto m = download(atom.mass), v = download(atom.velocity_per_atom);
  double result = 0;
  for (std::size_t i = 0; i < v.size(); ++i)
    result += m[i % m.size()] * v[i] * v[i];
  return result;
}
void scale_bead_velocities(Atom& atom, double factor)
{
  auto v = download(atom.velocity_per_atom);
  for (auto& value : v)
    value *= factor;
  upload(atom.velocity_per_atom, v);
}
struct Fixture {
  Atom atom;
  Box box{};
  std::vector<Group> groups = std::vector<Group>(1);
  GPU_Vector<double> thermo{8};
  explicit Fixture(bool mixed)
  {
    atom.number_of_atoms = 4;
    box.cpu_h[0] = box.cpu_h[4] = box.cpu_h[8] = 100;
    box.cpu_h[9] = box.cpu_h[13] = box.cpu_h[17] = 0.01;
    box.pbc_x = box.pbc_y = box.pbc_z = 0;
    groups[0].cpu_label = mixed ? std::vector<int>{7, 3, 7, 7}
                                : std::vector<int>{0, 1, 2, 3};
    upload(atom.mass, {8, 18, 12, 16});
    upload(atom.position_per_atom, {4.5, 6.6, 5.7, 5.1, 5, 5.4, 5.1, 5.8, 5, 5.3, 5, 5});
    // A translating and rotating triangle, plus a free particle.
    const std::vector<rigid_math::Vec3> x = {
      {4.5, 5, 5}, {6.6, 5.4, 5.3}, {5.7, 5.1, 5}, {5.1, 5.8, 5}};
    std::vector<double> v(12);
    for (int i = 0; i < 4; ++i) {
      const auto velocity = rigid_math::add(
        {0.03, -0.02, 0.04}, rigid_math::cross(
          {0.08, 0.11, 0.09}, rigid_math::subtract(x[i], {5, 5, 5})));
      v[i] = velocity.x;
      v[i + 4] = velocity.y;
      v[i + 8] = velocity.z;
    }
    v[1] = -0.06; v[5] = 0.05; v[9] = 0.02;
    upload(atom.velocity_per_atom, v);
    upload(atom.force_per_atom, std::vector<double>(12, 0));
    upload(atom.potential_per_atom, std::vector<double>(4, 0));
    upload(atom.virial_per_atom, std::vector<double>(36, 0));
  }
  void forces()
  {
    const auto x = download(atom.position_per_atom);
    std::vector<double> force(12), potential(4, 0);
    // Conservative one-body harmonic potential; it exerts a nonzero body torque.
    for (int i = 0; i < 12; ++i) {
      const double displacement = x[i] - 5;
      force[i] = -0.2 * displacement;
      potential[i % 4] += 0.1 * displacement * displacement;
    }
    upload(atom.force_per_atom, force);
    upload(atom.potential_per_atom, potential);
  }
};

void test_free_particles_match_existing_nhc()
{
  Fixture a(false), b(false);
  Ensemble_Rigid_NVT rigid(0, target, coupling, dt);
  double move_velocity[3] = {0, 0, 0};
  Ensemble_NHC ordinary(2, -1, move_velocity, 4, target, coupling, dt);
  rigid.initialize(a.atom, a.box, a.groups);
  assert(rigid.thermal_degrees_of_freedom() == 12);
  for (int n = 0; n < 100; ++n) {
    if (n == 50) { rigid.temperature = ordinary.temperature = 350; }
    a.forces(); b.forces();
    rigid.compute1(dt, a.groups, a.box, a.atom, a.thermo);
    ordinary.compute1(dt, b.groups, b.box, b.atom, b.thermo);
    near(download(a.atom.position_per_atom), download(b.atom.position_per_atom));
    near(download(a.atom.velocity_per_atom), download(b.atom.velocity_per_atom));
    a.forces(); b.forces();
    rigid.compute2(dt, a.groups, a.box, a.atom, a.thermo);
    ordinary.compute2(dt, b.groups, b.box, b.atom, b.thermo);
    near(download(a.atom.velocity_per_atom), download(b.atom.velocity_per_atom));
    for (int i = 0; i < NOSE_HOOVER_CHAIN_LENGTH; ++i) {
      near(rigid.pos_nhc1[i], ordinary.pos_nhc1[i]);
      near(rigid.vel_nhc1[i], ordinary.vel_nhc1[i]);
    }
    near(download(a.thermo)[0], twice_kinetic(a.atom) / (12 * K_B));
  }
}

void test_mixed_splitting_and_constraint_window()
{
  Fixture actual(true), reference(true);
  Ensemble_Rigid_NVT nvt(0, target, coupling, dt);
  nvt.initialize(actual.atom, actual.box, actual.groups);
  assert(nvt.thermal_degrees_of_freedom() == 9);
  double eta[4]{}, pi[4] = {0.015, -0.01, 0.02, -0.005}, q[4];
  for (int i = 0; i < 4; ++i) {
    nvt.vel_nhc1[i] = pi[i];
    q[i] = nvt.mas_nhc1[i];
    near(q[i], K_B * target * (dt * coupling) * (dt * coupling) * (i == 0 ? 9 : 1));
  }
  const double first_scale = nose_hoover_chain::propagate(
    4, eta, pi, q, twice_kinetic(actual.atom), K_B * target, 9, dt / 2);
  // Independent reference: scale all bead velocities on the host BEFORE
  // initializing NVE, then run its existing physical step unchanged.
  scale_bead_velocities(reference.atom, first_scale);
  Ensemble_Rigid_NVE nve(0);
  nve.initialize(reference.atom, reference.box, reference.groups);
  actual.forces(); reference.forces();
  nvt.compute1(dt, actual.groups, actual.box, actual.atom, actual.thermo);
  nve.compute1(dt, reference.groups, reference.box, reference.atom, reference.thermo);
  near(download(actual.atom.position_per_atom), download(reference.atom.position_per_atom));
  near(download(actual.atom.velocity_per_atom), download(reference.atom.velocity_per_atom));
  near(download(nvt.constraint_virial_per_body()), download(nve.constraint_virial_per_body()));
  actual.forces(); reference.forces();
  nve.compute2(dt, reference.groups, reference.box, reference.atom, reference.thermo);
  const double last_scale = nose_hoover_chain::propagate(
    4, eta, pi, q, twice_kinetic(reference.atom), K_B * target, 9, dt / 2);
  scale_bead_velocities(reference.atom, last_scale);
  nvt.compute2(dt, actual.groups, actual.box, actual.atom, actual.thermo);
  near(download(actual.atom.velocity_per_atom), download(reference.atom.velocity_per_atom));
  near(download(nvt.constraint_virial_per_body()), download(nve.constraint_virial_per_body()));
  for (int i = 0; i < 4; ++i) {
    near(nvt.pos_nhc1[i], eta[i]);
    near(nvt.vel_nhc1[i], pi[i]);
  }
  const auto thermo = download(actual.thermo);
  near(thermo[0], twice_kinetic(actual.atom) / (9 * K_B));
  // Output pressure uses final thermostatted velocities and physical-only virial.
  const auto v = download(reference.atom.velocity_per_atom), m = download(reference.atom.mass);
  const auto w = download(nve.constraint_virial_per_body());
  const int row[6] = {0, 1, 2, 0, 0, 1}, col[6] = {0, 1, 2, 1, 2, 2};
  for (int c = 0; c < 6; ++c) {
    double pressure = w[c];
    for (int i = 0; i < 4; ++i)
      pressure += m[i] * v[i + 4 * row[c]] * v[i + 4 * col[c]];
    near(thermo[2 + c], pressure / actual.box.get_volume());
  }
  // Each run resets the chain; a changed time step is rejected before integration.
  nvt.initialize(actual.atom, actual.box, actual.groups);
  for (int i = 0; i < 4; ++i) {
    near(nvt.pos_nhc1[i], 0);
    near(nvt.vel_nhc1[i], i % 2 == 0 ? 1 : -1);
  }
  bool rejected = false;
  try { nvt.compute1(2 * dt, actual.groups, actual.box, actual.atom, actual.thermo); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}
} // namespace

int main()
{
  int devices = 0;
  if (gpuGetDeviceCount(&devices) != gpuSuccess || !devices) {
    std::cout << "SKIP: no accessible GPU\n";
    return 0;
  }
  test_free_particles_match_existing_nhc();
  test_mixed_splitting_and_constraint_window();
  std::cout << "Rigid NVT scaling and thermostat splitting checks passed.\n";
}
