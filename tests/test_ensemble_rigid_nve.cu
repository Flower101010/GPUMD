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

// Independent LAMMPS-style reference using actual bead x/v snapshots rather
// than body quaternions or the production kernel's reconstructed velocities.
static std::vector<double> constraint_reference(
  const std::vector<double>& x, const std::vector<double>& before,
  const std::vector<double>& after, const std::vector<double>& force,
  const std::vector<double>& mass, const std::vector<int>& members, double dt)
{
  std::vector<double> w(6,0);
  Vec3 total{0,0,0};
  const int n=mass.size();
  for (int i:members) {
    const Vec3 r=read(x,i,n);
    const Vec3 fc=subtract(scale(subtract(read(after,i,n),read(before,i,n)),mass[i]/(0.5*dt)),read(force,i,n));
    total=add(total,fc);
    w[0]+=0.5*r.x*fc.x; w[1]+=0.5*r.y*fc.y; w[2]+=0.5*r.z*fc.z;
    w[3]+=0.5*r.x*fc.y; w[4]+=0.5*r.x*fc.z; w[5]+=0.5*r.y*fc.z;
  }
  near(total,{0,0,0}); // permits replacing coherent absolute x by COM-relative r
  return w;
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
    std::vector<double> virial(36,0);
    double u=0;
    if (interacting) {
      const auto x=download(atom.position_per_atom);
      const Vec3 d=subtract(read(x,1,4),read(x,0,4));
      const double spring=0.8;
      f[0]=scale(d,spring); f[1]=scale(d,-spring);
      u=0.5*spring*norm_squared(d);
      const double a[3]={d.x,d.y,d.z};
      const int row[9]={0,1,2,0,0,1,1,2,2},col[9]={0,1,2,1,2,2,0,0,1};
      for (int c=0;c<9;++c)
        virial[c*4]=virial[c*4+1]=-0.5*spring*a[row[c]]*a[col[c]];
    }
    upload(atom.force_per_atom,soa(f));
    upload(atom.potential_per_atom,{0.5*u,0.5*u,0,0});
    upload(atom.virial_per_atom,virial);
    return u;
  }
};

static double simulate(double dt, bool interacting, double* pressure_error=nullptr)
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
  double max_pressure_error=0;
  const int steps=std::lround(4.0/dt);
  for (int i=0;i<steps;++i) {
    const auto x_before=download(f.atom.position_per_atom),v_before=download(f.atom.velocity_per_atom);
    const auto old_force=download(f.atom.force_per_atom);
    ensemble.compute1(dt,f.groups,f.box,f.atom,f.thermo);
    const auto x_half=download(f.atom.position_per_atom),v_half=download(f.atom.velocity_per_atom);
    auto expected_w=constraint_reference(x_before,v_before,v_half,old_force,f.mass,{0,2,3},dt);
    const auto first_w=download(ensemble.constraint_virial_per_body());
    for (int c=0;c<6;++c) near(first_w[c],expected_w[c]);
    const double u=f.force(interacting);
    const auto new_force=download(f.atom.force_per_atom);
    const auto model_virial=download(f.atom.virial_per_atom);
    ensemble.compute2(dt,f.groups,f.box,f.atom,f.thermo);
    const auto v=download(f.atom.velocity_per_atom);
    const auto x=download(f.atom.position_per_atom);
    const auto t=download(f.thermo);
    const double k=kinetic(f.mass,v);
    assert(download(f.atom.virial_per_atom)==model_virial);
    assert(download(f.atom.force_per_atom)==new_force);
    const auto second_w=constraint_reference(x_half,v_half,v,new_force,f.mass,{0,2,3},dt);
    const auto full_w=download(ensemble.constraint_virial_per_body());
    const int row[6]={0,1,2,0,0,1},col[6]={0,1,2,1,2,2};
    for (int c=0;c<6;++c) {
      expected_w[c]+=second_w[c];
      near(full_w[c],expected_w[c]); // both halves, old position and no prior-step accumulation
      double numerator=expected_w[c];
      for (int a=0;a<4;++a)
        numerator+=model_virial[c*4+a]+f.mass[a]*v[a+row[c]*4]*v[a+col[c]*4];
      near(t[2+c]*f.box.get_volume(),numerator);
    }
    if (!interacting) {
      Vec3 p_body{0,0,0};
      for (int a:{0,2,3}) p_body=add(p_body,scale(read(v,a,4),f.mass[a]));
      const double translational_twice_k=norm_squared(p_body)/36+f.mass[1]*norm_squared(read(v,1,4));
      max_pressure_error=std::max(max_pressure_error,
        std::abs((t[2]+t[3]+t[4])*f.box.get_volume()-translational_twice_k));
    }
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
  if (pressure_error) *pressure_error=max_pressure_error;
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
  assert(ensemble.constraint_virial_per_body().size()==0);
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
    for (double w:download(ensemble.constraint_virial_per_body())) near(w,0);
  }
}

// Two rotating triangles plus a singleton: compare equivalent coherent and
// independently wrapped inputs in a tilted cell. Includes both-stage tensor
// comparison and the reduction across more than one body.
static void periodic_virial_invariance()
{
  Fixture plain,periodic;
  const int n=7;
  const std::vector<double> mass{8,10,12,18,14,16,20};
  const std::vector<Vec3> x{{9.8,2,3},{2,9.8,4},{10.8,2,3},{4,4,4},
                          {3,9.8,4},{9.8,3,3},{2,10.8,4}};
  std::vector<Vec3> v(n,{0.03,0.02,0.01});
  for (int a:{0,2,5}) v[a]=add(Vec3{0.6,0.2,0.1},cross(Vec3{0.08,0.11,0.09},subtract(x[a],x[0])));
  for (int a:{1,4,6}) v[a]=add(Vec3{-0.2,0.4,0.3},cross(Vec3{0.12,-0.03,0.09},subtract(x[a],x[1])));
  auto wrapped=[](Vec3 r) {
    double z=r.z/10, y=(r.y-0.5*z)/10, a=(r.x-2*y-z)/10;
    a-=std::floor(a); y-=std::floor(y); z-=std::floor(z);
    return Vec3{10*a+2*y+z,10*y+0.5*z,10*z};
  };
  for (Fixture* f:{&plain,&periodic}) {
    f->atom.number_of_atoms=n; f->mass=mass;
    f->groups[1].cpu_label={7,11,7,3,11,7,11};
    f->box.cpu_h[0]=f->box.cpu_h[4]=f->box.cpu_h[8]=10;
    f->box.cpu_h[1]=2; f->box.cpu_h[2]=1; f->box.cpu_h[5]=0.5;
    std::vector<Vec3> positions=x;
    if (f==&periodic) {
      f->box.pbc_x=f->box.pbc_y=f->box.pbc_z=1;
      for (Vec3& r:positions) r=wrapped(r);
    }
    upload(f->atom.mass,mass); upload(f->atom.position_per_atom,soa(positions));
    upload(f->atom.velocity_per_atom,soa(v));
    upload(f->atom.force_per_atom,std::vector<double>(3*n,0));
    upload(f->atom.potential_per_atom,std::vector<double>(n,0));
    upload(f->atom.virial_per_atom,std::vector<double>(9*n,0));
  }
  Ensemble_Rigid_NVE a(1),b(1);
  a.initialize(plain.atom,plain.box,plain.groups);
  b.initialize(periodic.atom,periodic.box,periodic.groups);
  assert(a.constraint_virial_per_body().size()==12);
  const double dt=0.05;
  const std::vector<std::vector<int>> members{{0,2,5},{1,4,6}};
  const std::vector<double> force(3*n,0);
  for (int step=0;step<20;++step) {
    const auto x0=download(plain.atom.position_per_atom),v0=download(plain.atom.velocity_per_atom);
    a.compute1(dt,plain.groups,plain.box,plain.atom,plain.thermo);
    b.compute1(dt,periodic.groups,periodic.box,periodic.atom,periodic.thermo);
    const auto x1=download(plain.atom.position_per_atom),v1=download(plain.atom.velocity_per_atom);
    auto first_a=download(a.constraint_virial_per_body()),first_b=download(b.constraint_virial_per_body());
    for (int i=0;i<12;++i) near(first_a[i],first_b[i]);
    a.compute2(dt,plain.groups,plain.box,plain.atom,plain.thermo);
    b.compute2(dt,periodic.groups,periodic.box,periodic.atom,periodic.thermo);
    const auto v2=download(plain.atom.velocity_per_atom);
    const auto wa=download(a.constraint_virial_per_body()),wb=download(b.constraint_virial_per_body());
    const auto ta=download(plain.thermo),tb=download(periodic.thermo);
    double sum[6]={0,0,0,0,0,0};
    for (int body=0;body<2;++body) {
      const auto w1=constraint_reference(x0,v0,v1,force,mass,members[body],dt);
      const auto w2=constraint_reference(x1,v1,v2,force,mass,members[body],dt);
      for (int c=0;c<6;++c) {
        near(first_a[c*2+body],w1[c]);
        near(wa[c*2+body],w1[c]+w2[c]);
        near(wa[c*2+body],wb[c*2+body]);
        sum[c]+=w1[c]+w2[c];
      }
    }
    const int row[6]={0,1,2,0,0,1},col[6]={0,1,2,1,2,2};
    for (int c=0;c<6;++c) {
      near(ta[c+2],tb[c+2]);
      for (int i=0;i<n;++i) sum[c]+=mass[i]*v2[i+row[c]*n]*v2[i+col[c]*n];
      near(ta[c+2]*plain.box.get_volume(),sum[c]);
    }
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
  periodic_virial_invariance();
  double p_coarse,p_medium,p_fine;
  simulate(0.08,false,&p_coarse);
  simulate(0.04,false,&p_medium);
  const double free_error=simulate(0.02,false,&p_fine);
  const double coarse=simulate(0.04,true),medium=simulate(0.02,true),fine=simulate(0.01,true);
  std::cout << "Energy max errors: free=" << free_error << ", coupled="
            << coarse << ", " << medium << ", " << fine << '\n';
  assert(coarse>0 && medium>0 && fine>0);
  assert(coarse/medium>3 && coarse/medium<5);
  assert(medium/fine>3 && medium/fine<5);
  std::cout << "Ideal rigid/free pressure trace errors (3PV): "
            << p_coarse << ", " << p_medium << ", " << p_fine << '\n';
  assert(p_coarse/p_medium>3 && p_coarse/p_medium<5);
  assert(p_medium/p_fine>3 && p_medium/p_fine<5);
  std::cout << "Rigid NVE stage checks passed.\n";
}
