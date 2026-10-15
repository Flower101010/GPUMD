#include "integrate/ensemble_rigid_nvt.cuh"
#include "fixtures/rigid_nh_reference.h"
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


void compare_chain(const RigidThermostat::Chain& chain, const std::vector<double>& mass,
  const std::vector<double>& position, const std::vector<double>& rate)
{
  assert(chain.size() == mass.size());
  for (std::size_t i = 0; i < chain.size(); ++i) {
    near(chain[i].mass, mass[i]);
    near(chain[i].coordinate, position[i]);
    near(chain[i].rate, rate[i]);
  }
}

// Host physical step with the literal LAMMPS kick/scale/drift/chain/scale/kick
// ordering. The chain body is the independent, extracted LAMMPS implementation.
void test_lammps_splitting(int mode)
{
  Fixture actual(mode != 0);
  if (mode == 2) actual.groups[0].cpu_label = {7, 7, 7, 7};
  Ensemble_Rigid_NVT nvt(0, target, coupling, dt);
  nvt.initialize(actual.atom, actual.box, actual.groups);
  RigidBodySystem::HostSnapshot host;
  nvt.rigid_bodies().copy_to_host(host);
  auto x = download(actual.atom.position_per_atom), v = download(actual.atom.velocity_per_atom);
  const auto mass = download(actual.atom.mass);
  const int ft = mode == 0 ? 12 : (mode == 1 ? 6 : 3), fr = mode == 0 ? 0 : 3;
  assert(nvt.thermal_degrees_of_freedom() == ft + fr);
  // Source rotation equations require nonzero nf_r. For the free-only extension
  // use an unused independent reference rotation chain, comparing translation only.
  lammps_reference::Reference ref({ft, fr == 0 ? 3 : fr, K_B*target, dt*coupling, 1, 0});
  std::vector<bool> member(4, false);
  for (int i : host.member_atoms) member[i] = true;
  auto forces = [&] {
    std::vector<double> f(12);
    for (int i = 0; i < 12; ++i) f[i] = -0.2*(x[i]-5);
    return f;
  };
  auto rebuild = [&] {
    for (const auto& body : host.bodies) {
      const auto& g = body.geometry;
      const auto& s = body.state;
      const auto L = rigid_math::conjqm_to_angular_momentum(s.quat, s.conjqm);
      const auto omega = rigid_math::angular_momentum_to_omega(s.quat, L, g.inertia);
      for (int j = g.first_member; j < g.first_member+g.natoms; ++j) {
        const int i = host.member_atoms[j];
        const auto r = rigid_math::rotate(s.quat, host.member_displace[j]);
        const auto p = rigid_math::add(s.xcm, r);
        const auto u = rigid_math::add(s.vcm, rigid_math::cross(omega, r));
        x[i] = p.x; x[i+4] = p.y; x[i+8] = p.z;
        v[i] = u.x; v[i+4] = u.y; v[i+8] = u.z;
      }
    }
  };
  auto kick = [&](const std::vector<double>& f) {
    for (int i = 0; i < 4; ++i) if (!member[i])
      for (int c = 0; c < 3; ++c) v[i+4*c] += dt/2*f[i+4*c]/mass[i];
    for (auto& body : host.bodies) {
      Vec3 force{}, torque{};
      auto& s = body.state;
      const auto& g = body.geometry;
      for (int j = g.first_member; j < g.first_member+g.natoms; ++j) {
        const int i = host.member_atoms[j];
        const Vec3 fi{f[i], f[i+4], f[i+8]};
        force = rigid_math::add(force, fi);
        torque = rigid_math::add(torque,
          rigid_math::cross(rigid_math::rotate(s.quat, host.member_displace[j]), fi));
      }
      s.vcm = rigid_math::add(s.vcm, rigid_math::scale(force, dt/(2*g.mass)));
      const auto torque_body = rigid_math::inverse_rotate(s.quat, torque);
      const auto fquat = rigid_math::multiply(s.quat, {0, torque_body.x, torque_body.y, torque_body.z});
      s.conjqm.w += dt*fquat.w; s.conjqm.x += dt*fquat.x;
      s.conjqm.y += dt*fquat.y; s.conjqm.z += dt*fquat.z;
    }
  };
  auto scale = [&] {
    const double st = std::exp(-dt/2*ref.eta_dot_t[0]);
    const double sr = std::exp(-dt/2*ref.eta_dot_r[0]);
    for (int i = 0; i < 4; ++i) if (!member[i])
      for (int c = 0; c < 3; ++c) v[i+4*c] *= st;
    for (auto& body : host.bodies) {
      body.state.vcm = rigid_math::scale(body.state.vcm, st);
      auto& p = body.state.conjqm;
      p.w *= sr; p.x *= sr; p.y *= sr; p.z *= sr;
    }
  };
  for (int step = 0; step < 100; ++step) {
    nvt.temperature = target + step; // Exercise source mass updates under a ramp.
    const auto f = forces();
    actual.forces();
    kick(f); scale();
    for (int i = 0; i < 4; ++i) if (!member[i])
      for (int c = 0; c < 3; ++c) x[i+4*c] += dt*v[i+4*c];
    for (auto& body : host.bodies) {
      auto& s = body.state;
      s.xcm = rigid_math::add(s.xcm, rigid_math::scale(s.vcm, dt));
      rigid_math::integrate_free_rotation(s.quat, s.conjqm, body.geometry.inertia, dt);
    }
    rebuild();
    double kt = 0, kr = 0;
    for (int i = 0; i < 4; ++i) if (!member[i])
      for (int c = 0; c < 3; ++c) kt += mass[i]*v[i+4*c]*v[i+4*c];
    for (const auto& body : host.bodies) {
      kt += body.geometry.mass*rigid_math::norm_squared(body.state.vcm);
      const auto L = rigid_math::conjqm_to_angular_momentum(body.state.quat, body.state.conjqm);
      const auto omega = rigid_math::angular_momentum_to_omega(body.state.quat, L, body.geometry.inertia);
      kr += rigid_math::dot(L, omega);
    }
    ref.set_duration(dt);
    ref.t_target = K_B*nvt.temperature;
    ref.akin_t = kt; ref.akin_r = fr == 0 ? 3*ref.t_target : kr;
    ref.nhc_temp_integrate();
    nvt.compute1(dt, actual.groups, actual.box, actual.atom, actual.thermo);
    near(download(actual.atom.position_per_atom), x);
    near(download(actual.atom.velocity_per_atom), v);
    compare_chain(nvt.thermostat().translation_chain(), ref.q_t, ref.eta_t, ref.eta_dot_t);
    if (fr) compare_chain(nvt.thermostat().rotation_chain(), ref.q_r, ref.eta_r, ref.eta_dot_r);
    else assert(nvt.thermostat().rotation_chain().empty());
    actual.forces();
    const auto new_force = forces();
    scale(); kick(new_force); rebuild();
    nvt.compute2(dt, actual.groups, actual.box, actual.atom, actual.thermo);
    near(download(actual.atom.velocity_per_atom), v);
    const auto t = download(actual.thermo);
    near(t[0], twice_kinetic(actual.atom)/((ft+fr)*K_B));
    // Continuous mechanical pressure trace, independently using relative arms.
    double internal = 0;
    for (const auto& body : host.bodies)
      for (int j = body.geometry.first_member; j < body.geometry.first_member+body.geometry.natoms; ++j) {
        const int i = host.member_atoms[j];
        const auto r = rigid_math::rotate(body.state.quat, host.member_displace[j]);
        internal += r.x*new_force[i]+r.y*new_force[i+4]+r.z*new_force[i+8];
      }
    double final_kt = 0;
    for (int i = 0; i < 4; ++i) if (!member[i])
      for (int c = 0; c < 3; ++c) final_kt += mass[i]*v[i+4*c]*v[i+4*c];
    for (const auto& body : host.bodies)
      final_kt += body.geometry.mass*rigid_math::norm_squared(body.state.vcm);
    near((t[2]+t[3]+t[4])*actual.box.get_volume(), final_kt-internal);
    if (fr && step == 0)
      assert(std::abs(ref.eta_dot_t[0]-ref.eta_dot_r[0]) > 1e-5);
  }
  // Reset is per-run, and changed dt is rejected before any dynamics.
  nvt.initialize(actual.atom, actual.box, actual.groups);
  for (const auto* chain : {&nvt.thermostat().translation_chain(), &nvt.thermostat().rotation_chain()})
    for (const auto& link : *chain) { near(link.coordinate, 0); near(link.rate, 0); }
  bool rejected = false;
  try { nvt.compute1(2*dt, actual.groups, actual.box, actual.atom, actual.thermo); }
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
  for (int mode : {0, 1, 2}) test_lammps_splitting(mode);
  std::cout << "Rigid NVT independent chains, LAMMPS splitting, ramps and pressure checks passed.\n";
}
