#include "integrate/rigid_body_integrator.cuh"
#include "integrate/rigid_thermo.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace rigid_math;
static void near(double a, double b) { assert(std::abs(a - b) < 3e-11 * (1 + std::abs(b))); }
static void near(Vec3 a, Vec3 b)
{
  near(a.x, b.x);
  near(a.y, b.y);
  near(a.z, b.z);
}
static void near(Quaternion a, Quaternion b)
{
  near(a.w, b.w);
  near(a.x, b.x);
  near(a.y, b.y);
  near(a.z, b.z);
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
static RigidBodySystem::Body body(const RigidBodySystem& rigid)
{
  RigidBodySystem::HostSnapshot snapshot;
  rigid.copy_to_host(snapshot);
  return snapshot.bodies.at(0);
}
template <class Exception, class F>
static void rejects(F f)
{
  bool caught = false;
  try {
    f();
  } catch (const Exception&) {
    caught = true;
  }
  assert(caught);
}

int main()
{
  // h 是完整操作时长，不能在 kick 内再次除以 2。
  Vec3 x{1, 2, 3}, v{0.2, -0.1, 0.4};
  rigid_integrator::kick_translation(0.3, 2, {4, -2, 6}, v);
  near(v, {0.8, -0.4, 1.3});
  rigid_integrator::drift_translation(0.7, x, v);
  near(x, {1.56, 1.72, 3.91});

  int devices = 0;
  if (gpuGetDeviceCount(&devices) != gpuSuccess || !devices) {
    std::cout << "SKIP: no accessible GPU\n";
    return 0;
  }
  Box box{};
  box.cpu_h[0] = box.cpu_h[4] = box.cpu_h[8] = 100;
  Atom atom;
  atom.number_of_atoms = 4;
  upload(atom.mass, {1, 2, 3, 4});
  const std::vector<double> positions{1, 2, 1, 7, 1, 1, 2, 8, 1, 1, 1, 9};
  const std::vector<double> velocities{0, 0, 0, 0.2, 0, 0, 0, -0.1, 0, 0, 0, 0.4};
  const std::vector<double> forces{1, 0, 0, 8, 0, 2, 0, -4, 0, 0, 3, 12};
  upload(atom.position_per_atom, positions);
  upload(atom.velocity_per_atom, velocities);
  upload(atom.force_per_atom, forces);
  RigidBodySystem rigid;
  rigid.initialize(atom, box, {0, 0, 0, -1});
  rigid.compute_forces_and_torques(atom);
  const auto initial = body(rigid);
  rigid_integrator::kick_translation(0.2, atom, rigid);
  const auto kicked = body(rigid);
  near(kicked.state.vcm, {0.2 / 6, 0.4 / 6, 0.6 / 6});
  near(kicked.state.xcm, initial.state.xcm);
  near(kicked.state.quat, initial.state.quat);
  near(kicked.state.conjqm, initial.state.conjqm);
  assert(download(atom.position_per_atom) == positions);
  auto result = download(atom.velocity_per_atom);
  near(Vec3{result[3], result[7], result[11]}, {0.6, -0.3, 1.0});
  for (int i : {0, 1, 2, 4, 5, 6, 8, 9, 10})
    near(result[i], velocities[i]);
  atom.force_per_atom.clear(); // drift 不应依赖 force 缓冲区。
  rigid_integrator::drift_translation(0.35, atom, rigid);
  const auto drifted = body(rigid);
  near(drifted.state.xcm, add(initial.state.xcm, scale(kicked.state.vcm, 0.35)));
  near(drifted.state.vcm, kicked.state.vcm);
  result = download(atom.position_per_atom);
  near(Vec3{result[3], result[7], result[11]}, {7.21, 7.895, 9.35});
  for (int i : {0, 1, 2, 4, 5, 6, 8, 9, 10})
    near(result[i], positions[i]);
  rejects<std::invalid_argument>([&] { rigid_integrator::kick_translation(0.1, atom, rigid); });
  rejects<std::invalid_argument>([&] {
    rigid_integrator::drift_translation(std::numeric_limits<double>::infinity(), atom, rigid);
  });

  // 独立转矩冲量：q 不变，空间 L 增加 tau*h。
  rigid_integrator::kick_rotation(0.17, rigid);
  auto rotated = body(rigid);
  near(rotated.state.quat, initial.state.quat);
  near(
    conjqm_to_angular_momentum(rotated.state.quat, rotated.state.conjqm),
    scale(initial.load.torque, 0.17));
  near(rotated.state.xcm, drifted.state.xcm);
  near(rotated.state.vcm, drifted.state.vcm);

  // 主轴自由转动解析解与负时间反演，不依赖转矩或 Atom forces。
  auto state = rotated.state;
  const double omega = 0.4, h = 0.23;
  const Vec3 L = rotate(state.quat, {rotated.geometry.inertia.x * omega, 0, 0});
  state.conjqm = angular_momentum_to_conjqm(state.quat, L);
  CHECK(gpuMemcpy(rigid.device_view().state, &state, sizeof(state), gpuMemcpyHostToDevice));
  const Quaternion expected =
    multiply(state.quat, {std::cos(omega * h / 2), std::sin(omega * h / 2), 0, 0});
  rigid_integrator::drift_rotation(h, rigid);
  rotated = body(rigid);
  near(rotated.state.quat, expected);
  near(dot(rotated.state.quat, rotated.state.conjqm), 0);
  near(conjqm_to_angular_momentum(rotated.state.quat, rotated.state.conjqm), L);
  rigid_integrator::drift_rotation(-h, rigid);
  rotated = body(rigid);
  near(rotated.state.quat, state.quat);
  near(rotated.state.conjqm, state.conjqm);
  rejects<std::invalid_argument>(
    [&] { rigid_integrator::kick_rotation(std::numeric_limits<double>::quiet_NaN(), rigid); });

  // 恒温缩放只作用于独立动量；bead 由调用者重建，避免被重复缩放。
  rigid.rebuild_velocities(atom);
  const auto before_scale = download(atom.velocity_per_atom);
  const auto before_body = body(rigid);
  rigid_integrator::scale_momenta(1.7, atom, rigid);
  const auto scaled_body = body(rigid);
  near(scaled_body.state.xcm, before_body.state.xcm);
  near(scaled_body.state.quat, before_body.state.quat);
  near(scaled_body.state.vcm, scale(before_body.state.vcm, 1.7));
  near(
    conjqm_to_angular_momentum(scaled_body.state.quat, scaled_body.state.conjqm),
    scale(conjqm_to_angular_momentum(before_body.state.quat, before_body.state.conjqm), 1.7));
  near(dot(scaled_body.state.quat, scaled_body.state.conjqm), 0);
  result = download(atom.velocity_per_atom);
  for (int i : {0, 1, 2, 4, 5, 6, 8, 9, 10})
    near(result[i], before_scale[i]);
  rigid.rebuild_velocities(atom);
  result = download(atom.velocity_per_atom);
  for (std::size_t i = 0; i < result.size(); ++i)
    near(result[i], 1.7 * before_scale[i]);
  rigid_integrator::scale_momenta(1.0 / 1.7, atom, rigid);
  rigid.rebuild_velocities(atom);
  rejects<std::invalid_argument>([&] { rigid_integrator::scale_momenta(0, atom, rigid); });
  rejects<std::invalid_argument>([&] {
    rigid_integrator::scale_momenta(std::numeric_limits<double>::quiet_NaN(), atom, rigid);
  });

  // 共用 thermo 明确拒绝乱序/未完成样本，错误调用不推进统计阶段。
  state.conjqm = {0, 0, 0, 0};
  CHECK(gpuMemcpy(rigid.device_view().state, &state, sizeof(state), gpuMemcpyHostToDevice));
  upload(atom.force_per_atom, std::vector<double>(12, 0));
  rigid.compute_forces_and_torques(atom);
  rigid.rebuild_velocities(atom);
  RigidThermo stats;
  GPU_Vector<double> thermo(8, 0.0);
  rejects<std::logic_error>([&] { stats.begin_step(rigid); });
  stats.initialize(rigid);
  assert(stats.degrees_of_freedom() == 9);
  rejects<std::logic_error>([&] { stats.add_constraint_pressure(1, thermo); });
  rejects<std::logic_error>([&] { stats.record_first_half(0.1, rigid, atom); });
  stats.begin_step(rigid);
  rejects<std::logic_error>([&] { stats.begin_step(rigid); });
  rejects<std::logic_error>([&] { stats.record_second_half(0.1, rigid, atom); });
  rejects<std::invalid_argument>([&] { stats.record_first_half(0, rigid, atom); });
  stats.record_first_half(0.1, rigid, atom);
  rejects<std::logic_error>([&] { stats.add_constraint_pressure(1, thermo); });
  rejects<std::invalid_argument>([&] { stats.record_second_half(0.2, rigid, atom); });
  stats.record_second_half(0.1, rigid, atom);
  rejects<std::invalid_argument>([&] { stats.add_constraint_pressure(0, thermo); });
  stats.add_constraint_pressure(1, thermo);
  for (double entry : download(thermo))
    near(entry, 0);
  for (double entry : download(stats.constraint_virial_per_body()))
    near(entry, 0);
  stats.begin_step(rigid);
  stats.clear();
  assert(stats.degrees_of_freedom() == 0);
  rejects<std::logic_error>([&] { stats.begin_step(rigid); });

  Atom empty;
  RigidBodySystem empty_rigid;
  empty_rigid.initialize(empty, box, {});
  rigid_integrator::kick_translation(0.1, empty, empty_rigid);
  rigid_integrator::drift_translation(0.1, empty, empty_rigid);
  rigid_integrator::kick_rotation(0.1, empty_rigid);
  rigid_integrator::drift_rotation(0.1, empty_rigid);
  rigid_integrator::scale_momenta(1.2, empty, empty_rigid);
  std::cout << "Rigid operators and thermo window checks passed.\n";
}
