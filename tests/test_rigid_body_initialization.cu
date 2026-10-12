#include "integrate/rigid_body.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace rigid_math;

static void near(double a, double b, double tol = 2e-10)
{
  assert(std::abs(a-b) <= tol*(1+std::abs(b)));
}
static void near(Vec3 a, Vec3 b)
{
  near(a.x,b.x); near(a.y,b.y); near(a.z,b.z);
}
static Vec3 h_times(const Box& box, Vec3 v)
{
  return {box.cpu_h[0]*v.x+box.cpu_h[1]*v.y+box.cpu_h[2]*v.z,
          box.cpu_h[3]*v.x+box.cpu_h[4]*v.y+box.cpu_h[5]*v.z,
          box.cpu_h[6]*v.x+box.cpu_h[7]*v.y+box.cpu_h[8]*v.z};
}
// 测试盒均为上三角；用解析逆得到期望的盒内质心。
static Vec3 wrapped_center(const Box& box, Vec3 center)
{
  double z = center.z / box.cpu_h[8];
  double y = (center.y - box.cpu_h[5]*z) / box.cpu_h[4];
  double x = (center.x - box.cpu_h[1]*y - box.cpu_h[2]*z) / box.cpu_h[0];
  if (box.pbc_x) x -= std::floor(x);
  if (box.pbc_y) y -= std::floor(y);
  if (box.pbc_z) z -= std::floor(z);
  return h_times(box,{x,y,z});
}
static Box orthogonal()
{
  Box box{};
  box.cpu_h[0] = box.cpu_h[4] = box.cpu_h[8] = 10;
  return box;
}
static std::vector<double> soa(const std::vector<Vec3>& v)
{
  std::vector<double> result(3*v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = v[i].x; result[i+v.size()] = v[i].y; result[i+2*v.size()] = v[i].z;
  }
  return result;
}
static void populate(Atom& atom, const std::vector<double>& mass,
                     const std::vector<Vec3>& position, const std::vector<Vec3>& velocity)
{
  atom.number_of_atoms = static_cast<int>(mass.size());
  atom.mass.resize(mass.size());
  atom.position_per_atom.resize(3*mass.size());
  atom.velocity_per_atom.resize(3*mass.size());
  if (!mass.empty()) {
    atom.mass.copy_from_host(mass.data());
    atom.position_per_atom.copy_from_host(soa(position).data());
    atom.velocity_per_atom.copy_from_host(soa(velocity).data());
  }
  // 故意把 CPU 缓存设为错误值：初始化必须读取当前 DEVICE 数据。
  atom.cpu_mass.assign(mass.size(), -1);
  atom.cpu_velocity_per_atom.assign(3*mass.size(), 999);
}
static Vec3 center_of(const std::vector<Vec3>& x, const std::vector<double>& mass)
{
  Vec3 weighted{0,0,0}; double M = 0;
  for (std::size_t i = 0; i < x.size(); ++i) { weighted = add(weighted,scale(x[i],mass[i])); M += mass[i]; }
  return scale(weighted,1/M);
}
static void check_body(const RigidBodySystem::HostSnapshot& snapshot, int b,
                       const Box& box, const std::vector<Vec3>& x,
                       const std::vector<double>& mass, Vec3 V, Vec3 omega)
{
  const auto& body = snapshot.bodies[b];
  const auto& g = body.geometry;
  const auto& s = body.state;
  const Vec3 center = center_of(x,mass);
  double M = 0; for (double m : mass) M += m;
  near(g.mass,M); near(s.vcm,V);
  near(s.xcm,wrapped_center(box,center));
  near(dot(s.quat,s.quat),1); near(dot(s.quat,s.conjqm),0);
  double tensor[3][3] = {};
  Vec3 expected_L{0,0,0}, weighted{0,0,0};
  for (int k = 0; k < g.natoms; ++k) {
    Vec3 r = subtract(x[k],center);
    Vec3 d = snapshot.member_displace[g.first_member+k];
    near(rotate(s.quat,d),r);
    weighted = add(weighted,scale(d,mass[k]));
    expected_L = add(expected_L,cross(r,scale(cross(omega,r),mass[k])));
    tensor[0][0] += mass[k]*(d.y*d.y+d.z*d.z);
    tensor[1][1] += mass[k]*(d.x*d.x+d.z*d.z);
    tensor[2][2] += mass[k]*(d.x*d.x+d.y*d.y);
    tensor[0][1] -= mass[k]*d.x*d.y;
    tensor[0][2] -= mass[k]*d.x*d.z;
    tensor[1][2] -= mass[k]*d.y*d.z;
  }
  near(weighted,{0,0,0});
  near(Vec3{tensor[0][0],tensor[1][1],tensor[2][2]},g.inertia);
  near(tensor[0][1],0); near(tensor[0][2],0); near(tensor[1][2],0);
  near(conjqm_to_angular_momentum(s.quat,s.conjqm),expected_L);
  near(angular_momentum_to_omega(s.quat,expected_L,g.inertia),omega);
  near(body.load.fcm,{0,0,0}); near(body.load.torque,{0,0,0});
  assert(g.inertia.x <= g.inertia.y && g.inertia.y <= g.inertia.z);
}
static std::vector<Vec3> velocities_for(const std::vector<Vec3>& x,
  const std::vector<double>& mass, Vec3 V, Vec3 omega)
{
  const Vec3 center = center_of(x,mass);
  std::vector<Vec3> result;
  for (Vec3 p : x) {
    Vec3 r = subtract(p,center);
    // 加入伸缩速度，它不改变总线动量/角动量，应被初始化的刚体投影排除。
    result.push_back(add(V,add(cross(omega,r),scale(r,0.3))));
  }
  return result;
}

int main()
{
  int device_count = 0;
  if (gpuGetDeviceCount(&device_count) != gpuSuccess || device_count == 0) {
    std::cout << "SKIP: no accessible GPU\n";
    return 0;
  }
  RigidBodySystem rigid;
  Atom atom;
  Box box = orthogonal();
  const Vec3 V{0.2,-0.4,0.7}, omega{0.3,-0.2,0.5};
  const std::vector<Vec3> triangle{{39.8,2,3},{40.2,2,3},{39.8,2.4,3}};
  const auto velocity = velocities_for(triangle,{1,1,1},V,omega);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  populate(atom,{2,1,1,1,-1},{{10.1,-0.2,31},{9.8,2,3},{0.2,2,3},{9.8,2.4,3},{nan,nan,nan}},
           {{0.1,0.2,0.3},velocity[0],velocity[1],velocity[2],{nan,nan,nan}});
  const auto original_positions = soa(std::vector<Vec3>{{10.1,-0.2,31},{9.8,2,3},{0.2,2,3},{9.8,2.4,3},{nan,nan,nan}});
  rigid.initialize(atom,box,{10,20,20,20,-1});
  assert(rigid.initialized() && rigid.number_of_bodies() == 2 && rigid.number_of_members() == 4);
  RigidBodySystem::HostSnapshot snapshot;
  rigid.copy_to_host(snapshot);
  assert(snapshot.member_atoms == std::vector<int>({0,1,2,3}));
  const auto& point = snapshot.bodies[0];
  near(point.geometry.mass,2); assert(point.geometry.rotational_dof == 0);
  near(point.state.xcm,{0.1,9.8,1});
  near(point.state.vcm,{0.1,0.2,0.3}); near(dot(point.state.conjqm,point.state.conjqm),0);
  near(point.state.quat.w,1);
  check_body(snapshot,1,box,triangle,{1,1,1},V,omega);
  near(snapshot.bodies[1].geometry.inertia,{0.16/3,0.16,0.64/3});
  std::vector<int> reverse(5);
  const RigidBodySystem& const_rigid = rigid;
  const auto view = const_rigid.device_view();
  CHECK(gpuMemcpy(reverse.data(),view.atom_to_body,5*sizeof(int),gpuMemcpyDeviceToHost));
  assert(reverse == std::vector<int>({0,1,1,1,-1}));
  std::vector<double> after(15);
  atom.position_per_atom.copy_to_host(after.data());
  for (int i = 0; i < 15; ++i)
    if (std::isfinite(original_positions[i])) near(after[i],original_positions[i]);

  // 倾斜盒 + 部分周期性 + 不等质量 + 任意空间主轴。
  box = orthogonal();
  box.cpu_h[1] = 2; box.cpu_h[2] = 1; box.cpu_h[4] = 9; box.cpu_h[5] = 0.5; box.cpu_h[8] = 8;
  box.pbc_z = 0;
  const std::vector<Vec3> frac{{0.98,0.98,-0.2},{1.02,0.98,-0.2},{0.98,1.02,-0.19}};
  std::vector<Vec3> actual, wrapped;
  for (Vec3 f : frac) {
    actual.push_back(h_times(box,f));
    f.x -= std::floor(f.x); f.y -= std::floor(f.y);
    wrapped.push_back(h_times(box,f));
  }
  populate(atom,{1,2,3},wrapped,velocities_for(actual,{1,2,3},V,omega));
  atom.position_image.clear();
  rigid.initialize(atom,box,{7,7,7});
  rigid.copy_to_host(snapshot);
  check_body(snapshot,0,box,actual,{1,2,3},V,omega);

  // 显式几何不受半盒限制，也不依赖 Atom 的位置/image。
  box = orthogonal();
  const std::vector<Vec3> large{{0,0,0},{6,0,0},{0,6,1},{1,1,3}};
  populate(atom,{1,1,1,1},large,velocities_for(large,{1,1,1,1},V,omega));
  atom.position_per_atom.clear();
  rigid.initialize(atom,box,{3,3,3,3},soa(large));
  rigid.copy_to_host(snapshot);
  check_body(snapshot,0,box,large,{1,1,1,1},V,omega);

  // 对称四面体的三个主惯量均为 8：重根主轴选择须可复现。
  const std::vector<Vec3> tetra{{1,1,1},{1,-1,-1},{-1,1,-1},{-1,-1,1}};
  populate(atom,{1,1,1,1},tetra,velocities_for(tetra,{1,1,1,1},V,omega));
  rigid.initialize(atom,box,{11,11,11,11},soa(tetra));
  rigid.copy_to_host(snapshot);
  check_body(snapshot,0,box,tetra,{1,1,1,1},V,omega);
  near(snapshot.bodies[0].geometry.inertia,{8,8,8});
  const Quaternion saved = snapshot.bodies[0].state.quat;
  rigid.initialize(atom,box,{11,11,11,11},soa(tetra));
  rigid.copy_to_host(snapshot);
  near(dot(saved,snapshot.bodies[0].state.quat),1);

  // 多个朝向覆盖矩阵转四元数的不同分支，并验证主轴方向/力矩坐标约定。
  const Vec3 axis = scale(Vec3{1,2,3},1/std::sqrt(14.0));
  for (int sample = 0; sample < 24; ++sample) {
    const double angle = sample * 0.29;
    std::vector<Vec3> rotated;
    for (Vec3 r : large) {
      // 独立 Rodrigues 公式构造输入，避免用待验证的 quat 生成相同输出。
      rotated.push_back(add(Vec3{2,3,4},
        add(scale(r,std::cos(angle)),add(scale(cross(axis,r),std::sin(angle)),
                                      scale(axis,dot(axis,r)*(1-std::cos(angle)))))));
    }
    populate(atom,{1,2,3,4},rotated,velocities_for(rotated,{1,2,3,4},V,omega));
    rigid.initialize(atom,box,{100,100,100,100},soa(rotated));
    rigid.copy_to_host(snapshot);
    check_body(snapshot,0,box,rotated,{1,2,3,4},V,omega);
  }

  auto fails = [&](const std::vector<int>& labels) {
    bool caught = false;
    try { rigid.initialize(atom,box,labels); } catch (const std::invalid_argument&) { caught = true; }
    assert(caught && !rigid.initialized() && rigid.number_of_bodies() == 0);
  };
  populate(atom,{1,1,1},{{1,1,1},{2,1,1},{3,1,1}},{{0,0,0},{0,0,0},{0,0,0}});
  fails({0,0,0}); // 线性
  populate(atom,{1,1,1},{{1,1,1},{1,1,1},{1,1,1}},{{0,0,0},{0,0,0},{0,0,0}});
  fails({0,0,0}); // 重合
  populate(atom,{1,1,1},{{1,1,1},{4,1,1},{7,2,1}},{{0,0,0},{0,0,0},{0,0,0}});
  fails({0,0,0}); // 参考展开后的跨度 > 半盒
  fails({-2,0,0}); fails({0});
  populate(atom,{0,1,1},{{1,1,1},{2,1,1},{1,2,1}},{{0,0,0},{0,0,0},{0,0,0}});
  fails({0,0,0});
  populate(atom,{1},{{1,1,1}},{{nan,0,0}}); fails({0});
  populate(atom,{1},{{1,1,1}},{{0,0,0}});
  box.cpu_h[8] = 0; fails({0}); box = orthogonal();
  populate(atom,{1},{{10.2,1,1}},{{0,0,0}});
  // 刚体初始化与 Atom 的跨盒历史无关，连该缓冲区尺寸也不依赖。
  atom.position_image.resize(1, std::numeric_limits<int>::max());
  rigid.initialize(atom,box,{0});
  rigid.copy_to_host(snapshot);
  near(snapshot.bodies[0].state.xcm,{0.2,1,1});
  int unchanged_image = 0;
  atom.position_image.copy_to_host(&unchanged_image);
  assert(unchanged_image == std::numeric_limits<int>::max());
  atom.position_image.clear();
  rigid.initialize(atom,box,{-1});
  rigid.copy_to_host(snapshot);
  assert(rigid.initialized() && snapshot.bodies.empty() && snapshot.member_atoms.empty());
  populate(atom,{}, {}, {});
  rigid.initialize(atom,box,{});
  rigid.copy_to_host(snapshot);
  assert(rigid.initialized() && rigid.number_of_members() == 0);
  rigid.clear(); assert(!rigid.initialized());
  std::cout << "Rigid body initialization checks passed.\n";
}
