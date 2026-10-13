#include "integrate/rigid_body.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace rigid_math;

static void near(double a, double b)
{
  if (std::abs(a - b) > 3e-10 * (1 + std::abs(b)))
    std::cerr << "Mismatch: " << a << " versus " << b << '\n';
  assert(std::abs(a - b) <= 3e-10 * (1 + std::abs(b)));
}
static void near(Vec3 a, Vec3 b)
{
  near(a.x, b.x);
  near(a.y, b.y);
  near(a.z, b.z);
}
static std::vector<double> soa(const std::vector<Vec3>& values)
{
  std::vector<double> result(3 * values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    result[i] = values[i].x;
    result[i + values.size()] = values[i].y;
    result[i + 2 * values.size()] = values[i].z;
  }
  return result;
}
static Vec3 read_vec(const std::vector<double>& values, int i, int N)
{
  return {values[i], values[i + N], values[i + 2 * N]};
}
static Vec3 h_times(const Box& box, Vec3 f)
{
  return {
    box.cpu_h[0] * f.x + box.cpu_h[1] * f.y + box.cpu_h[2] * f.z,
    box.cpu_h[3] * f.x + box.cpu_h[4] * f.y + box.cpu_h[5] * f.z,
    box.cpu_h[6] * f.x + box.cpu_h[7] * f.y + box.cpu_h[8] * f.z};
}
// 本测试盒均为上三角；用解析逆独立计算期望周期包装。
static Vec3 wrapped(const Box& box, Vec3 p)
{
  double z = p.z / box.cpu_h[8];
  double y = (p.y - box.cpu_h[5] * z) / box.cpu_h[4];
  double x = (p.x - box.cpu_h[1] * y - box.cpu_h[2] * z) / box.cpu_h[0];
  if (box.pbc_x)
    x -= std::floor(x);
  if (box.pbc_y)
    y -= std::floor(y);
  if (box.pbc_z)
    z -= std::floor(z);
  return h_times(box, {x, y, z});
}
static void near_position(const Box& box, Vec3 actual, Vec3 expected)
{
  // 边界处 0 和 L-epsilon 周期等价；同时检查坐标确实在主盒附近。
  auto fractional = [&](Vec3 p) {
    const double z = p.z / box.cpu_h[8];
    const double y = (p.y - box.cpu_h[5] * z) / box.cpu_h[4];
    const double x = (p.x - box.cpu_h[1] * y - box.cpu_h[2] * z) / box.cpu_h[0];
    return Vec3{x, y, z};
  };
  const Vec3 f = fractional(actual);
  if (box.pbc_x)
    assert(f.x >= -1e-12 && f.x <= 1 + 1e-12);
  if (box.pbc_y)
    assert(f.y >= -1e-12 && f.y <= 1 + 1e-12);
  if (box.pbc_z)
    assert(f.z >= -1e-12 && f.z <= 1 + 1e-12);
  Vec3 error = fractional(subtract(actual, expected));
  if (box.pbc_x)
    error.x -= std::nearbyint(error.x);
  if (box.pbc_y)
    error.y -= std::nearbyint(error.y);
  if (box.pbc_z)
    error.z -= std::nearbyint(error.z);
  near(h_times(box, error), {0, 0, 0});
}
static void upload(GPU_Vector<double>& target, const std::vector<double>& values)
{
  target.resize(values.size());
  if (!values.empty())
    target.copy_from_host(values.data());
}
static std::vector<double> download(const GPU_Vector<double>& values)
{
  std::vector<double> result(values.size());
  if (!result.empty())
    values.copy_to_host(result.data());
  return result;
}
static Vec3 turn90(Vec3 v) { return {-v.y, v.x, v.z}; }
static void near(Quaternion a, Quaternion b)
{
  near(a.w, b.w);
  near(a.x, b.x);
  near(a.y, b.y);
  near(a.z, b.z);
}

template <class Exception, class Function>
static void rejects(Function f)
{
  bool caught = false;
  try {
    f();
  } catch (const Exception&) {
    caught = true;
  }
  assert(caught);
}

static void exercise(const Box& box)
{
  const int N = 8;
  Atom atom;
  atom.number_of_atoms = N;
  const std::vector<int> labels{9, -1, 2, 9, 2, 9, 2, 2};
  const std::vector<double> mass{1, 2, 1, 2, 2, 3, 3, 4};
  const std::vector<Vec3> x{
    {9.8, 2, 3},
    {777, 888, 999},
    {2, 2, 2},
    {10.2, 2, 3},
    {2.3, 2, 2},
    {9.8, 2.4, 3},
    {2, 2.3, 2},
    {2, 2, 2.3}};
  std::vector<Vec3> velocity(N, {0.1, 0.2, 0.3});
  velocity[1] = {-7, 8, -9};
  std::vector<Vec3> forces{
    {1, 2, 3},
    {1e90, 1e90, 1e90},
    {-1, 3, 2},
    {2, -4, 1},
    {4, 1, -2},
    {-3, 1, 5},
    {1, -2, 4},
    {3, 4, -1}};
  upload(atom.mass, mass);
  upload(atom.position_per_atom, soa(x));
  upload(atom.velocity_per_atom, soa(velocity));
  upload(atom.force_per_atom, soa(forces));
  RigidBodySystem rigid;
  rigid.initialize(atom, box, labels, soa(x));
  RigidBodySystem::HostSnapshot snapshot;
  rigid.copy_to_host(snapshot);
  assert(snapshot.bodies.size() == 2);
  assert(snapshot.member_atoms == std::vector<int>({2, 4, 6, 7, 0, 3, 5}));

  std::vector<RigidBodySystem::BodyState> states;
  std::vector<Vec3> lever(N), expected_velocity(N), expected_position(N);
  expected_velocity[1] = velocity[1];
  expected_position[1] = x[1];
  const double h = std::sqrt(0.5);
  const Quaternion space_turn{h, 0, 0, h};
  for (int b = 0; b < 2; ++b) {
    const auto& g = snapshot.bodies[b].geometry;
    auto state = snapshot.bodies[b].state;
    Vec3 center{0, 0, 0};
    for (int j = g.first_member; j < g.first_member + g.natoms; ++j) {
      const int i = snapshot.member_atoms[j];
      center = add(center, scale(x[i], mass[i] / g.mass));
    }
    state.quat = multiply(space_turn, state.quat);
    // 质心跨越正/负多个周期；第三方向在部分周期盒中应保留。
    state.xcm = h_times(box, b == 0 ? Vec3{2.95, -3.02, 4.01} : Vec3{-3.01, 4.02, -2.05});
    state.vcm = b == 0 ? Vec3{1, 2, -3} : Vec3{-2, 1, 4};
    const Vec3 omega = b == 0 ? Vec3{0.7, -0.4, 0.2} : Vec3{-0.5, 0.3, 0.9};
    Vec3 L{0, 0, 0};
    for (int j = g.first_member; j < g.first_member + g.natoms; ++j) {
      const int i = snapshot.member_atoms[j];
      lever[i] = turn90(subtract(x[i], center));
      expected_velocity[i] = add(state.vcm, cross(omega, lever[i]));
      expected_position[i] = wrapped(box, add(wrapped(box, state.xcm), lever[i]));
      L = add(L, cross(lever[i], scale(cross(omega, lever[i]), mass[i])));
    }
    state.conjqm = angular_momentum_to_conjqm(state.quat, L);
    states.push_back(state);
  }
  CHECK(gpuMemcpy(
    rigid.device_view().state,
    states.data(),
    states.size() * sizeof(states[0]),
    gpuMemcpyHostToDevice));

  // 力矩只能依赖固定几何和姿态，不能用包装后 bead 坐标相减。
  // 故意污染成员坐标；合力和力矩仍应来自上述已知空间力臂。
  std::vector<Vec3> poisoned(N, {-1234, 5678, -9999});
  poisoned[1] = x[1];
  upload(atom.position_per_atom, soa(poisoned));
  rigid.compute_forces_and_torques(atom);
  rigid.compute_forces_and_torques(atom); // 第二次必须覆盖，不能累加旧载荷。
  rigid.copy_to_host(snapshot);
  for (int b = 0; b < 2; ++b) {
    Vec3 F{0, 0, 0}, tau{0, 0, 0};
    const auto& g = snapshot.bodies[b].geometry;
    for (int j = g.first_member; j < g.first_member + g.natoms; ++j) {
      const int i = snapshot.member_atoms[j];
      F = add(F, forces[i]);
      tau = add(tau, cross(lever[i], forces[i]));
    }
    near(snapshot.bodies[b].load.fcm, F);
    near(snapshot.bodies[b].load.torque, tau);
  }
  assert(download(atom.force_per_atom) == soa(forces));

  rigid.remap_centers(box);
  rigid.remap_centers(box); // 幂等；默认 stream 顺序执行。
  rigid.rebuild_positions(atom, box);
  rigid.rebuild_velocities(atom);
  const auto result_x = download(atom.position_per_atom);
  const auto result_v = download(atom.velocity_per_atom);
  for (int i = 0; i < N; ++i) {
    if (i == 1)
      near(read_vec(result_x, i, N), expected_position[i]);
    else
      near_position(box, read_vec(result_x, i, N), expected_position[i]);
    near(read_vec(result_v, i, N), expected_velocity[i]);
  }
  rigid.copy_to_host(snapshot);
  for (int b = 0; b < 2; ++b) {
    const auto& actual = snapshot.bodies[b].state;
    near_position(box, actual.xcm, wrapped(box, states[b].xcm));
    near(actual.vcm, states[b].vcm);
    near(actual.quat, states[b].quat);
    near(actual.conjqm, states[b].conjqm);
    const auto& g = snapshot.bodies[b].geometry;
    Vec3 momentum{0, 0, 0}, weighted{0, 0, 0};
    for (int j = g.first_member; j < g.first_member + g.natoms; ++j) {
      const int i = snapshot.member_atoms[j];
      momentum = add(momentum, scale(read_vec(result_v, i, N), mass[i]));
      weighted = add(weighted, scale(lever[i], mass[i]));
    }
    near(momentum, scale(actual.vcm, g.mass));
    near(weighted, {0, 0, 0});
  }

  // 质量比例均匀加速度：绕质心力矩必须为零，而非保留上一时刻力矩。
  const Vec3 acceleration{0.2, -0.3, 0.4};
  for (int i = 0; i < N; ++i)
    forces[i] = scale(acceleration, mass[i]);
  upload(atom.force_per_atom, soa(forces));
  rigid.compute_forces_and_torques(atom);
  rigid.copy_to_host(snapshot);
  for (const auto& body : snapshot.bodies) {
    near(body.load.fcm, scale(acceleration, body.geometry.mass));
    near(body.load.torque, {0, 0, 0});
  }

  // 所有包装/重建都不写 image。连续展开输出则应明确拒绝。
  atom.position_image.resize(1, 17);
  rigid.rebuild_positions(atom, box);
  int image = 0;
  atom.position_image.copy_to_host(&image);
  assert(image == 17);
  atom.unwrapped_position.resize(3 * N);
  rejects<std::invalid_argument>([&] { rigid.rebuild_positions(atom, box); });
  atom.unwrapped_position.clear();
  Box changed = box;
  changed.cpu_h[0] += 0.1;
  rejects<std::invalid_argument>([&] { rigid.remap_centers(changed); });
  rejects<std::invalid_argument>([&] { rigid.rebuild_positions(atom, changed); });
  changed = box;
  changed.pbc_x = 1 - box.pbc_x;
  rejects<std::invalid_argument>([&] { rigid.remap_centers(changed); });
  atom.number_of_atoms = N + 1;
  rejects<std::invalid_argument>([&] { rigid.compute_forces_and_torques(atom); });
  rejects<std::invalid_argument>([&] { rigid.rebuild_velocities(atom); });
  atom.number_of_atoms = N;
  atom.force_per_atom.clear();
  rejects<std::invalid_argument>([&] { rigid.compute_forces_and_torques(atom); });
  atom.velocity_per_atom.clear();
  rejects<std::invalid_argument>([&] { rigid.rebuild_velocities(atom); });
  atom.position_per_atom.clear();
  rejects<std::invalid_argument>([&] { rigid.rebuild_positions(atom, box); });
  assert(rigid.initialized()); // 错误调用不销毁有效的刚体状态。
}

static void many_bodies()
{
  const int B = 257, N = 4 * B;
  Box box{};
  box.cpu_h[0] = box.cpu_h[4] = box.cpu_h[8] = 10;
  Atom atom;
  atom.number_of_atoms = N;
  std::vector<int> labels(N, -1);
  std::vector<Vec3> x(N), velocity(N), force(N);
  std::vector<double> mass(N, 1);
  for (int b = 0; b < B; ++b) {
    const double a = b + 1;
    x[4 * b] = {a, 1, 1};
    x[4 * b + 1] = {a + 0.2, 1, 1};
    x[4 * b + 2] = {a, 1.2, 1};
    x[4 * b + 3] = {999, -888, 777};
    for (int k = 0; k < 3; ++k) {
      labels[4 * b + k] = 10 * b + 7;
      velocity[4 * b + k] = {b * 0.01, 2, 3};
    }
    velocity[4 * b + 3] = {-9, 8, -7};
    force[4 * b] = {1, 0, 0};
    force[4 * b + 1] = {0, 1, 0};
    force[4 * b + 2] = {0, 0, 1};
    force[4 * b + 3] = {1e90, 1e90, 1e90};
  }
  upload(atom.mass, mass);
  upload(atom.position_per_atom, soa(x));
  upload(atom.velocity_per_atom, soa(velocity));
  upload(atom.force_per_atom, soa(force));
  RigidBodySystem rigid;
  rigid.initialize(atom, box, labels, soa(x));
  rigid.compute_forces_and_torques(atom);
  rigid.remap_centers(box);
  rigid.rebuild_positions(atom, box);
  rigid.rebuild_velocities(atom);
  RigidBodySystem::HostSnapshot snapshot;
  rigid.copy_to_host(snapshot);
  assert(snapshot.bodies.size() == B && snapshot.member_atoms.size() == 3 * B);
  const auto result_x = download(atom.position_per_atom);
  const auto result_v = download(atom.velocity_per_atom);
  for (int b = 0; b < B; ++b) {
    assert(snapshot.bodies[b].geometry.id == 10 * b + 7);
    near(snapshot.bodies[b].load.fcm, {1, 1, 1});
    near(snapshot.bodies[b].load.torque, {0.4 / 3, 0.2 / 3, 0.2});
    for (int k = 0; k < 4; ++k) {
      const int i = 4 * b + k;
      if (k == 3)
        near(read_vec(result_x, i, N), x[i]);
      else
        near_position(box, read_vec(result_x, i, N), wrapped(box, x[i]));
      near(read_vec(result_v, i, N), velocity[i]);
    }
  }
}

int main()
{
  int count = 0;
  if (gpuGetDeviceCount(&count) != gpuSuccess || count == 0) {
    std::cout << "SKIP: no accessible GPU\n";
    return 0;
  }
  Box box{};
  box.cpu_h[0] = box.cpu_h[4] = box.cpu_h[8] = 10;
  exercise(box);
  box.cpu_h[1] = 2;
  box.cpu_h[2] = 1;
  box.cpu_h[4] = 9;
  box.cpu_h[5] = 0.5;
  box.cpu_h[8] = 8;
  box.pbc_z = 0;
  exercise(box);
  box.pbc_x = box.pbc_y = 0;
  exercise(box); // 非周期体系也必须保持姿态与局部几何。

  many_bodies();

  Atom atom;
  RigidBodySystem rigid;
  rejects<std::logic_error>([&] { rigid.compute_forces_and_torques(atom); });
  rejects<std::logic_error>([&] { rigid.remap_centers(box); });
  rejects<std::logic_error>([&] { rigid.rebuild_positions(atom, box); });
  rejects<std::logic_error>([&] { rigid.rebuild_velocities(atom); });
  rigid.initialize(atom, box, {});
  rigid.compute_forces_and_torques(atom);
  rigid.remap_centers(box);
  rigid.rebuild_positions(atom, box);
  rigid.rebuild_velocities(atom);
  atom.number_of_atoms = 1;
  upload(atom.mass, {1});
  upload(atom.position_per_atom, {1, 2, 3});
  upload(atom.velocity_per_atom, {4, 5, 6});
  rigid.initialize(atom, box, {-1});
  rigid.compute_forces_and_torques(atom);
  rigid.remap_centers(box);
  rigid.rebuild_positions(atom, box);
  rigid.rebuild_velocities(atom);
  assert(download(atom.position_per_atom) == std::vector<double>({1, 2, 3}));
  assert(download(atom.velocity_per_atom) == std::vector<double>({4, 5, 6}));
  std::cout << "Rigid body GPU runtime checks passed.\n";
}
