#include "integrate/ensemble_rigid_npt.cuh"
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
    box.pbc_x = box.pbc_y = box.pbc_z = 1;
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


void check_geometry(const Fixture& f)
{
  const auto x = download(f.atom.position_per_atom), v = download(f.atom.velocity_per_atom);
  const double initial[3] = {1.45, 1.0, 0.85};
  const int a[3] = {0, 0, 2}, b[3] = {2, 3, 3};
  for (int pair = 0; pair < 3; ++pair) {
    double r2 = 0, rdv = 0;
    for (int c = 0; c < 3; ++c) {
      const double length = f.box.cpu_h[4*c];
      double r = x[a[pair]+4*c] - x[b[pair]+4*c];
      r -= length * std::round(r / length);
      r2 += r*r;
      rdv += r * (v[a[pair]+4*c] - v[b[pair]+4*c]);
    }
    near(r2, initial[pair]);
    near(rdv, 0);
  }
}

// Independent host affine trajectory for force-free particles. The scalar chain
// module itself is separately checked against original LAMMPS function bodies.
void test_free_particle_flow()
{
  Fixture f(false);
  Ensemble_Rigid_NPT npt(0, target, coupling, 0.0001, 1000, dt);
  npt.initialize(f.atom, f.box, f.groups);
  assert(npt.thermal_degrees_of_freedom() == 12);
  auto x = download(f.atom.position_per_atom), v = download(f.atom.velocity_per_atom);
  const auto m = download(f.atom.mass);
  RigidNH reference({12, 0, K_B*target, coupling*dt, 1000*dt, 0.0001});
  double length = 100;
  auto kinetic = [&] {
    double k = 0;
    for (int i = 0; i < 12; ++i) k += m[i%4]*v[i]*v[i];
    return k;
  };
  // initialize does not read force, pressure or advance the piston.
  near(npt.extended_system().piston().rate, 0);
  reference.advance_piston(dt/2, kinetic()/(3*length*length*length), length*length*length, kinetic(), 0);
  for (int step = 0; step < 100; ++step) {
    const auto first = reference.momentum_scales(dt/2);
    const double nu = reference.piston().rate;
    const double scale = std::exp(nu*dt);
    const double phi = nu == 0 ? dt : std::expm1(nu*dt)/nu;
    length *= scale;
    for (int i = 0; i < 12; ++i) {
      v[i] *= first.translation;
      x[i] = scale*x[i] + phi*v[i];
      x[i] -= length*std::floor(x[i]/length);
    }
    reference.advance_thermostats(dt, kinetic(), 0);
    reference.advance_piston_thermostat(dt);
    npt.compute1(dt, f.groups, f.box, f.atom, f.thermo);
    near(f.box.cpu_h[0], length);
    near(f.box.cpu_h[4], length);
    near(f.box.cpu_h[8], length);
    near(download(f.atom.position_per_atom), x);
    near(download(f.atom.velocity_per_atom), v);
    const auto last = reference.momentum_scales(dt/2);
    for (auto& value : v) value *= last.translation;
    const double k = kinetic(), volume = length*length*length;
    reference.advance_piston(dt/2, k/(3*volume), volume, k, 0);
    npt.compute2(dt, f.groups, f.box, f.atom, f.thermo);
    near(download(f.atom.velocity_per_atom), v);
    near(npt.extended_system().piston().rate, reference.piston().rate);
    const auto t = download(f.thermo);
    near(t[0], k/(12*K_B));
    near((t[2]+t[3]+t[4])/3, k/(3*volume));
  }
}

void test_mixed_constraints_and_pressure()
{
  Fixture f(true);
  Ensemble_Rigid_NPT npt(0, target, coupling, -0.00001, 2000, dt);
  npt.initialize(f.atom, f.box, f.groups);
  assert(npt.thermal_degrees_of_freedom() == 9);
  assert(npt.extended_system().parameters().translation_dof == 6);
  assert(npt.extended_system().parameters().rotation_dof == 3);
  for (int step = 0; step < 100; ++step) {
    f.forces();
    npt.compute1(dt, f.groups, f.box, f.atom, f.thermo);
    check_geometry(f);
    f.forces();
    const auto model_virial = download(f.atom.virial_per_atom);
    const auto force = download(f.atom.force_per_atom);
    npt.compute2(dt, f.groups, f.box, f.atom, f.thermo);
    check_geometry(f);
    near(download(f.atom.force_per_atom), force);
    near(download(f.atom.virial_per_atom), model_virial);
    RigidBodySystem::HostSnapshot snapshot;
    npt.rigid_bodies().copy_to_host(snapshot);
    const auto& body = snapshot.bodies[0];
    const auto mass = download(f.atom.mass), velocity = download(f.atom.velocity_per_atom);
    double kt = body.geometry.mass*rigid_math::norm_squared(body.state.vcm);
    for (int c = 0; c < 3; ++c) kt += mass[1]*velocity[1+4*c]*velocity[1+4*c];
    double internal_virial = 0;
    for (int j = 0; j < body.geometry.natoms; ++j) {
      const int i = snapshot.member_atoms[j];
      const auto r = rigid_math::rotate(body.state.quat, snapshot.member_displace[j]);
      internal_virial += r.x*force[i]+r.y*force[i+4]+r.z*force[i+8];
    }
    const auto t = download(f.thermo);
    near(t[0], twice_kinetic(f.atom)/(9*K_B));
    // For arbitrary rotation and torque, internal rotational kinetic energy
    // cancels from the scalar pressure. No thermostat impulse enters this test.
    near((t[2]+t[3]+t[4])*f.box.get_volume(), kt-internal_virial);
    for (double value : t) assert(std::isfinite(value));
  }
  bool rejected = false;
  try { npt.compute2(dt, f.groups, f.box, f.atom, f.thermo); }
  catch (const std::logic_error&) { rejected = true; }
  assert(rejected);
  rejected = false;
  try { npt.compute1(2*dt, f.groups, f.box, f.atom, f.thermo); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
  npt.initialize(f.atom, f.box, f.groups);
  near(npt.extended_system().piston().rate, 0);
  near(npt.extended_system().translation_chain()[0].rate, 0);
}

void test_box_and_stage_rejections()
{
  Fixture f(true);
  Ensemble_Rigid_NPT npt(0, target, coupling, 0, 2000, dt);
  bool rejected = false;
  try { npt.compute1(dt, f.groups, f.box, f.atom, f.thermo); }
  catch (const std::logic_error&) { rejected = true; }
  assert(rejected);
  for (int mode = 0; mode < 3; ++mode) {
    Box bad = f.box;
    if (mode == 0) bad.pbc_z = 0;
    if (mode == 1) bad.cpu_h[1] = 1;
    if (mode == 2) bad.cpu_h[0] = -100;
    rejected = false;
    try { npt.initialize(f.atom, bad, f.groups); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
}
} // namespace

int main()
{
  int devices = 0;
  if (gpuGetDeviceCount(&devices) != gpuSuccess || !devices) {
    std::cout << "SKIP: no accessible GPU\n";
    return 0;
  }
  test_free_particle_flow();
  test_mixed_constraints_and_pressure();
  test_box_and_stage_rejections();
  std::cout << "Rigid NPT affine flow, chains, mixed pressure and constraint checks passed.\n";
}
