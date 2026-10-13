#include "integrate/ensemble_rigid_nve.cuh"
#include "integrate/ensemble_nve.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace rigid_math;
static void near(double a, double b, double tolerance = 2e-10)
{
  if (std::abs(a-b) > tolerance * (1 + std::abs(b)))
    std::cerr << "Mismatch: " << a << " versus " << b << '\n';
  assert(std::abs(a-b) <= tolerance * (1 + std::abs(b)));
}
static void near(Vec3 a, Vec3 b) { near(a.x,b.x); near(a.y,b.y); near(a.z,b.z); }
static std::vector<double> soa(const std::vector<Vec3>& v)
{
  std::vector<double> a(3*v.size());
  for (std::size_t i=0; i<v.size(); ++i) {
    a[i]=v[i].x; a[i+v.size()]=v[i].y; a[i+2*v.size()]=v[i].z;
  }
  return a;
}
static Vec3 read(const std::vector<double>& a, int i, int n)
{
  return {a[i],a[i+n],a[i+2*n]};
}
static void upload(GPU_Vector<double>& a, const std::vector<double>& v)
{
  a.resize(v.size());
  if (!v.empty()) a.copy_from_host(v.data());
}
static std::vector<double> download(const GPU_Vector<double>& a)
{
  std::vector<double> v(a.size());
  if (!v.empty()) a.copy_to_host(v.data());
  return v;
}
template<class Exception, class F> static void rejects(F f)
{
  bool caught=false;
  try { f(); } catch (const Exception&) { caught=true; }
  assert(caught);
}
static double kinetic(const std::vector<double>& mass, const std::vector<double>& v)
{
  double k=0;
  for (std::size_t i=0; i<mass.size(); ++i) k += 0.5*mass[i]*norm_squared(read(v,i,mass.size()));
  return k;
}
static Vec3 momentum(const std::vector<double>& mass, const std::vector<double>& v)
{
  Vec3 p{0,0,0};
  for (std::size_t i=0; i<mass.size(); ++i) p=add(p,scale(read(v,i,mass.size()),mass[i]));
  return p;
}

struct Fixture {
  Atom atom;
  Box box{};
  std::vector<Group> groups = std::vector<Group>(2);
  GPU_Vector<double> thermo;
  std::vector<double> mass{8,18,12,16};
  Fixture() {
    atom.number_of_atoms=4;
    box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=100;
    box.cpu_h[9]=box.cpu_h[13]=box.cpu_h[17]=0.01;
    box.pbc_x=box.pbc_y=box.pbc_z=0;
    groups[0].cpu_label={0,0,0,0}; // deliberately not the molecule grouping
    groups[1].cpu_label={7,3,7,7};
    const std::vector<Vec3> x{{4.5,5,5},{6.6,5.4,5.3},{5.7,5.1,5},{5.1,5.8,5}};
    std::vector<Vec3> v;
    for (const auto& r:x) v.push_back(add(Vec3{0.03,-0.02,0.04},cross(Vec3{0.08,0.11,0.09},subtract(r,Vec3{5,5,5}))));
    v[1]={-0.06,0.05,0.02};
    upload(atom.mass,mass); upload(atom.position_per_atom,soa(x));
    upload(atom.velocity_per_atom,soa(v)); upload(atom.force_per_atom,std::vector<double>(12,0));
    upload(atom.potential_per_atom,std::vector<double>(4,0));
    upload(atom.virial_per_atom,std::vector<double>(36,0));
    thermo.resize(8);
  }
  // Independent conservative fixture: one free particle and one triangle bead
  // connected by U=k/2 |r_free-r_bead|^2. This exerts nonzero body torque.
  double force(bool interacting) {
    std::vector<Vec3> f(4, {0,0,0});
    double u=0;
    if (interacting) {
      const auto x=download(atom.position_per_atom);
      const Vec3 d=subtract(read(x,1,4),read(x,0,4));
      const double spring=0.8;
      f[0]=scale(d,spring); f[1]=scale(d,-spring);
      u=0.5*spring*norm_squared(d);
    }
    upload(atom.force_per_atom,soa(f));
    upload(atom.potential_per_atom,{0.5*u,0.5*u,0,0});
    return u;
  }
};

static double simulate(double dt, bool interacting)
{
  Fixture f;
  Ensemble_Rigid_NVE ensemble(1);
  ensemble.initialize(f.atom,f.box,f.groups);
  assert(ensemble.thermal_degrees_of_freedom()==9);
  assert(ensemble.rigid_bodies().number_of_bodies()==1);
  const auto x0=download(f.atom.position_per_atom);
  const auto v0=download(f.atom.velocity_per_atom);
  const Vec3 p0=momentum(f.mass,v0);
  const double e0=kinetic(f.mass,v0)+f.force(interacting);
  double max_error=0;
  const int steps=std::lround(4.0/dt);
  for (int i=0;i<steps;++i) {
    ensemble.compute1(dt,f.groups,f.box,f.atom,f.thermo);
    const double u=f.force(interacting);
    ensemble.compute2(dt,f.groups,f.box,f.atom,f.thermo);
    const auto v=download(f.atom.velocity_per_atom);
    const auto x=download(f.atom.position_per_atom);
    const auto t=download(f.thermo);
    const double k=kinetic(f.mass,v);
    near(t[0],2*k/(9*K_B)); near(t[1],u);
    near(momentum(f.mass,v),p0);
    for (int a:{0,2,3}) for (int b:{0,2,3})
      near(norm_squared(subtract(read(x,a,4),read(x,b,4))),
           norm_squared(subtract(read(x0,a,4),read(x0,b,4))));
    max_error=std::max(max_error,std::abs(k+u-e0));
    if (i%10==0) {
      RigidBodySystem::HostSnapshot snapshot;
      ensemble.rigid_bodies().copy_to_host(snapshot);
      const auto& body=snapshot.bodies[0];
      const auto& s=body.state;
      near(norm_squared(s.quat),1); near(dot(s.quat,s.conjqm),0);
      const Vec3 l=conjqm_to_angular_momentum(s.quat,s.conjqm);
      const Vec3 omega=angular_momentum_to_omega(s.quat,l,body.geometry.inertia);
      const double body_k=0.5*body.geometry.mass*norm_squared(s.vcm)+0.5*dot(l,omega);
      near(k,body_k+0.5*f.mass[1]*norm_squared(read(v,1,4)));
    }
  }
  // Reverse all physical velocities, reinitialize, and integrate back with +dt.
  auto reversed=download(f.atom.velocity_per_atom);
  for (double& v:reversed) v=-v;
  upload(f.atom.velocity_per_atom,reversed);
  ensemble.initialize(f.atom,f.box,f.groups);
  f.force(interacting);
  for (int i=0;i<steps;++i) {
    ensemble.compute1(dt,f.groups,f.box,f.atom,f.thermo);
    f.force(interacting);
    ensemble.compute2(dt,f.groups,f.box,f.atom,f.thermo);
  }
  const auto xr=download(f.atom.position_per_atom), vr=download(f.atom.velocity_per_atom);
  for (std::size_t i=0;i<xr.size();++i) { near(xr[i],x0[i],2e-9); near(vr[i],-v0[i],2e-9); }
  return max_error;
}

static void guards_and_free()
{
  Fixture f;
  rejects<std::invalid_argument>([] { Ensemble_Rigid_NVE invalid(-1); });
  Ensemble_Rigid_NVE ensemble(1);
  rejects<std::logic_error>([&] { ensemble.compute1(0.1,f.groups,f.box,f.atom,f.thermo); });
  ensemble.initialize(f.atom,f.box,f.groups);
  rejects<std::logic_error>([&] { ensemble.compute2(0.1,f.groups,f.box,f.atom,f.thermo); });
  rejects<std::invalid_argument>([&] { ensemble.compute1(0,f.groups,f.box,f.atom,f.thermo); });
  Box changed=f.box; changed.cpu_h[0]+=1;
  const auto before=download(f.atom.position_per_atom);
  rejects<std::invalid_argument>([&] { ensemble.compute1(0.1,f.groups,changed,f.atom,f.thermo); });
  assert(download(f.atom.position_per_atom)==before);
  f.force(false);
  ensemble.compute1(0.1,f.groups,f.box,f.atom,f.thermo);
  rejects<std::logic_error>([&] { ensemble.compute1(0.1,f.groups,f.box,f.atom,f.thermo); });
  rejects<std::logic_error>([&] { ensemble.initialize(f.atom,f.box,f.groups); });
  rejects<std::invalid_argument>([&] { ensemble.compute2(0.2,f.groups,f.box,f.atom,f.thermo); });
  f.force(false); ensemble.compute2(0.1,f.groups,f.box,f.atom,f.thermo);
  rejects<std::invalid_argument>([&] { ensemble.compute1(0.2,f.groups,f.box,f.atom,f.thermo); });
  ensemble.fixed_group=0;
  rejects<std::invalid_argument>([&] { ensemble.initialize(f.atom,f.box,f.groups); });
  assert(!ensemble.rigid_bodies().initialized()); assert(ensemble.thermal_degrees_of_freedom()==0);
  ensemble.fixed_group=-1;
  f.groups[1].cpu_label={0,1,2,3};
  ensemble.initialize(f.atom,f.box,f.groups);
  assert(ensemble.thermal_degrees_of_freedom()==12);
  const auto x=download(f.atom.position_per_atom),v=download(f.atom.velocity_per_atom);
  f.force(false); ensemble.compute1(0.1,f.groups,f.box,f.atom,f.thermo);
  f.force(false); ensemble.compute2(0.1,f.groups,f.box,f.atom,f.thermo);
  const auto xa=download(f.atom.position_per_atom);
  for (std::size_t i=0;i<x.size();++i) near(xa[i],x[i]+0.1*v[i]);
  Ensemble_NVE legacy(0);
  legacy.temperature=0;
  legacy.find_thermo(false,f.box.get_volume(),f.groups,f.atom.mass,f.atom.potential_per_atom,
                     f.atom.velocity_per_atom,f.atom.virial_per_atom,f.thermo);
  near(download(f.thermo)[0],2*kinetic(f.mass,v)/(12*K_B));
}

static void all_rigid_periodic()
{
  Fixture f;
  f.atom.number_of_atoms=3;
  f.mass={8,12,16};
  f.groups[1].cpu_label={4,4,4};
  f.box.pbc_x=f.box.pbc_y=f.box.pbc_z=1;
  const std::vector<Vec3> x{{99.8,4,5},{0.8,4,5},{99.8,5,5}};
  const std::vector<Vec3> v(3,{0.6,0.2,0.1});
  upload(f.atom.mass,f.mass); upload(f.atom.position_per_atom,soa(x));
  upload(f.atom.velocity_per_atom,soa(v));
  upload(f.atom.force_per_atom,std::vector<double>(9,0));
  upload(f.atom.potential_per_atom,std::vector<double>(3,0));
  upload(f.atom.virial_per_atom,std::vector<double>(27,0));
  Ensemble_Rigid_NVE ensemble(1);
  ensemble.initialize(f.atom,f.box,f.groups);
  assert(ensemble.thermal_degrees_of_freedom()==6);
  for (int i=0;i<20;++i) {
    ensemble.compute1(0.25,f.groups,f.box,f.atom,f.thermo);
    ensemble.compute2(0.25,f.groups,f.box,f.atom,f.thermo);
    const auto positions=download(f.atom.position_per_atom);
    for (double r:positions) assert(r>=0 && r<100);
    auto displacement=[&](int a,int b) {
      Vec3 d=subtract(read(positions,a,3),read(positions,b,3));
      d.x-=100*std::nearbyint(d.x/100);
      d.y-=100*std::nearbyint(d.y/100);
      d.z-=100*std::nearbyint(d.z/100);
      return norm_squared(d);
    };
    near(displacement(0,1),1); near(displacement(0,2),1); near(displacement(1,2),2);
    near(download(f.thermo)[0],2*kinetic(f.mass,download(f.atom.velocity_per_atom))/(6*K_B));
  }
}

int main()
{
  int devices=0;
  if (gpuGetDeviceCount(&devices)!=gpuSuccess || !devices) {
    std::cout << "SKIP: no accessible GPU\n"; return 0;
  }
  guards_and_free();
  all_rigid_periodic();
  const double free_error=simulate(0.02,false);
  const double coarse=simulate(0.04,true),medium=simulate(0.02,true),fine=simulate(0.01,true);
  std::cout << "Energy max errors: free=" << free_error << ", coupled="
            << coarse << ", " << medium << ", " << fine << '\n';
  assert(coarse>0 && medium>0 && fine>0);
  assert(coarse/medium>3 && coarse/medium<5);
  assert(medium/fine>3 && medium/fine<5);
  std::cout << "Rigid NVE stage checks passed.\n";
}
