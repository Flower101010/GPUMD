#include "model/atom.cuh"
#include "model/box.cuh"
#include "rigid_body.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace
{
using namespace rigid_math;
constexpr double inertia_tolerance = 1e-10;

void require(bool condition, const char* message)
{
  if (!condition)
    throw std::invalid_argument(message);
}

bool finite(Vec3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }

Vec3 read_vec(const std::vector<double>& values, int i, int N)
{
  const std::size_t index = static_cast<std::size_t>(i), count = static_cast<std::size_t>(N);
  return {values[index], values[index + count], values[index + 2 * count]};
}

void write_vec(std::vector<double>& values, int i, int N, Vec3 v)
{
  const std::size_t index = static_cast<std::size_t>(i), count = static_cast<std::size_t>(N);
  values[index] = v.x;
  values[index + count] = v.y;
  values[index + 2 * count] = v.z;
}

// 初始化自己从 H 求逆，不依赖调用方是否已刷新 Box 的缓存逆矩阵。
struct Cell {
  double h[9], inv[9];
  bool periodic[3];
  explicit Cell(const Box& box)
  {
    require(
      (box.pbc_x == 0 || box.pbc_x == 1) && (box.pbc_y == 0 || box.pbc_y == 1) &&
        (box.pbc_z == 0 || box.pbc_z == 1),
      "Invalid periodic boundary flags.");
    periodic[0] = box.pbc_x;
    periodic[1] = box.pbc_y;
    periodic[2] = box.pbc_z;
    double largest = 0;
    for (int i = 0; i < 9; ++i) {
      h[i] = box.cpu_h[i];
      require(std::isfinite(h[i]), "Non-finite box matrix.");
      largest = std::max(largest, std::abs(h[i]));
    }
    require(largest > 0, "Singular box matrix.");
    double a[9];
    for (int i = 0; i < 9; ++i)
      a[i] = h[i] / largest;
    const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) + a[1] * (a[5] * a[6] - a[3] * a[8]) +
                       a[2] * (a[3] * a[7] - a[4] * a[6]);
    require(
      std::abs(det) > 64 * std::numeric_limits<double>::epsilon(),
      "Singular or ill-conditioned box matrix.");
    const double cofactors[9] = {
      a[4] * a[8] - a[5] * a[7],
      a[2] * a[7] - a[1] * a[8],
      a[1] * a[5] - a[2] * a[4],
      a[5] * a[6] - a[3] * a[8],
      a[0] * a[8] - a[2] * a[6],
      a[2] * a[3] - a[0] * a[5],
      a[3] * a[7] - a[4] * a[6],
      a[1] * a[6] - a[0] * a[7],
      a[0] * a[4] - a[1] * a[3]};
    for (int i = 0; i < 9; ++i) {
      inv[i] = (cofactors[i] / det) / largest;
      require(std::isfinite(inv[i]), "Non-finite inverse box matrix.");
    }
  }
  __host__ __device__ Vec3 fractional(Vec3 v) const
  {
    return {
      inv[0] * v.x + inv[1] * v.y + inv[2] * v.z,
      inv[3] * v.x + inv[4] * v.y + inv[5] * v.z,
      inv[6] * v.x + inv[7] * v.y + inv[8] * v.z};
  }
  __host__ __device__ Vec3 cartesian(Vec3 v) const
  {
    return {
      h[0] * v.x + h[1] * v.y + h[2] * v.z,
      h[3] * v.x + h[4] * v.y + h[5] * v.z,
      h[6] * v.x + h[7] * v.y + h[8] * v.z};
  }
  __host__ __device__ Vec3 wrapped(Vec3 v) const
  {
    Vec3 f = fractional(v);
    if (periodic[0])
      f.x -= floor(f.x);
    if (periodic[1])
      f.y -= floor(f.y);
    if (periodic[2])
      f.z -= floor(f.z);
    return cartesian(f);
  }
  void wrap(Vec3& v) const
  {
    require(finite(fractional(v)), "Non-finite fractional coordinate.");
    v = wrapped(v);
    require(finite(v), "Non-finite wrapped coordinate.");
  }
};

// 对称 3x3 Jacobi 特征分解。归一化张量后旋转，固定选主元规则保证可复现。
void principal_axes(const double tensor[3][3], Vec3& inertia, Vec3 axes[3])
{
  double largest = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      require(std::isfinite(tensor[i][j]), "Non-finite inertia tensor.");
      largest = std::max(largest, std::abs(tensor[i][j]));
    }
  require(largest > 0, "Coincident multi-bead bodies are not supported.");
  double a[3][3], e[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      a[i][j] = tensor[i][j] / largest;
  bool converged = false;
  for (int sweep = 0; sweep < 64; ++sweep) {
    int p = 0, q = 1;
    if (std::abs(a[0][2]) > std::abs(a[p][q])) {
      p = 0;
      q = 2;
    }
    if (std::abs(a[1][2]) > std::abs(a[p][q])) {
      p = 1;
      q = 2;
    }
    if (std::abs(a[p][q]) <= 32 * std::numeric_limits<double>::epsilon()) {
      converged = true;
      break;
    }
    const double tau = (a[q][q] - a[p][p]) / (2 * a[p][q]);
    const double t = std::copysign(1.0, tau) / (std::abs(tau) + std::hypot(1.0, tau));
    const double c = 1 / std::hypot(1.0, t), s = t * c;
    a[p][p] -= t * a[p][q];
    a[q][q] += t * a[p][q];
    a[p][q] = a[q][p] = 0;
    for (int k = 0; k < 3; ++k) {
      if (k != p && k != q) {
        const double ap = a[k][p], aq = a[k][q];
        a[k][p] = a[p][k] = c * ap - s * aq;
        a[k][q] = a[q][k] = s * ap + c * aq;
      }
      const double ep = e[k][p], eq = e[k][q];
      e[k][p] = c * ep - s * eq;
      e[k][q] = s * ep + c * eq;
    }
  }
  require(converged, "Rigid inertia eigendecomposition did not converge.");
  int order[3] = {0, 1, 2};
  std::stable_sort(order, order + 3, [&](int i, int j) { return a[i][i] < a[j][j]; });
  double moments[3];
  const double max_eigenvalue = a[order[2]][order[2]];
  for (int i = 0; i < 3; ++i) {
    // 初版明确拒绝线性/近线性多 bead；不以跳过零惯量代替正确积分。
    require(
      a[order[i]][order[i]] > inertia_tolerance * max_eigenvalue,
      "Linear or near-degenerate multi-bead bodies are not supported.");
    moments[i] = a[order[i]][order[i]] * largest;
    require(std::isfinite(moments[i]) && moments[i] > 0, "Invalid principal inertia.");
    axes[i] = {e[0][order[i]], e[1][order[i]], e[2][order[i]]};
    const double components[3] = {axes[i].x, axes[i].y, axes[i].z};
    int pivot = 0;
    for (int k = 1; k < 3; ++k)
      if (std::abs(components[k]) > std::abs(components[pivot]))
        pivot = k;
    if (components[pivot] < 0)
      axes[i] = scale(axes[i], -1);
  }
  if (dot(cross(axes[0], axes[1]), axes[2]) < 0)
    axes[2] = scale(axes[2], -1);
  inertia = {moments[0], moments[1], moments[2]};
}

Quaternion axes_to_quaternion(const Vec3 axes[3])
{
  // 矩阵的列是空间主轴；Hamilton 四元数表示 body -> space。
  const double m[3][3] = {
    {axes[0].x, axes[1].x, axes[2].x},
    {axes[0].y, axes[1].y, axes[2].y},
    {axes[0].z, axes[1].z, axes[2].z}};
  Quaternion q;
  const double trace = m[0][0] + m[1][1] + m[2][2];
  if (trace > 0) {
    const double s = 2 * std::sqrt(trace + 1);
    q = {s / 4, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s};
  } else {
    int i = 0;
    if (m[1][1] > m[i][i])
      i = 1;
    if (m[2][2] > m[i][i])
      i = 2;
    const int j = (i + 1) % 3, k = (i + 2) % 3;
    const double s = 2 * std::sqrt(1 + m[i][i] - m[j][j] - m[k][k]);
    double v[3] = {};
    v[i] = s / 4;
    v[j] = (m[i][j] + m[j][i]) / s;
    v[k] = (m[i][k] + m[k][i]) / s;
    q = {(m[k][j] - m[j][k]) / s, v[0], v[1], v[2]};
  }
  require(normalize(q), "Invalid orientation quaternion.");
  // q 和 -q 表示相同姿态；确定符号便于快照比较。
  const double values[4] = {q.w, q.x, q.y, q.z};
  for (double value : values) {
    if (value == 0)
      continue;
    if (value < 0)
      q = {-q.w, -q.x, -q.y, -q.z};
    break;
  }
  return q;
}
// 运行时只在 DEVICE 上读写：每线程一个 body，成员范围互不重叠。
__device__ Vec3 read_device_vec(const double* values, int i, int N)
{
  const std::size_t index = static_cast<std::size_t>(i), count = static_cast<std::size_t>(N);
  return {values[index], values[index + count], values[index + 2 * count]};
}
__device__ void write_device_vec(double* values, int i, int N, Vec3 v)
{
  const std::size_t index = static_cast<std::size_t>(i), count = static_cast<std::size_t>(N);
  values[index] = v.x;
  values[index + count] = v.y;
  values[index + 2 * count] = v.z;
}
__global__ void sum_body_loads(
  RigidBodySystem::ConstDeviceView view, const double* forces, RigidBodySystem::BodyLoad* loads)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= view.number_of_bodies)
    return;
  const auto& g = view.geometry[b];
  const Quaternion q = view.state[b].quat;
  Vec3 force{0, 0, 0}, torque{0, 0, 0};
  for (int j = g.first_member; j < g.first_member + g.natoms; ++j) {
    const Vec3 f = read_device_vec(forces, view.member_atoms[j], view.number_of_atoms);
    const Vec3 r = rotate(q, view.member_displace[j]);
    force = add(force, f);
    torque = add(torque, cross(r, f));
  }
  loads[b] = {force, torque}; // 覆盖旧值，不累加上一时刻的载荷。
}
__global__ void wrap_body_centers(RigidBodySystem::DeviceView view, Cell cell)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < view.number_of_bodies)
    view.state[b].xcm = cell.wrapped(view.state[b].xcm);
}
__global__ void
reconstruct_positions(RigidBodySystem::ConstDeviceView view, Cell cell, double* positions)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= view.number_of_bodies)
    return;
  const auto& g = view.geometry[b];
  const auto& s = view.state[b];
  for (int j = g.first_member; j < g.first_member + g.natoms; ++j) {
    const Vec3 position = cell.wrapped(add(s.xcm, rotate(s.quat, view.member_displace[j])));
    write_device_vec(positions, view.member_atoms[j], view.number_of_atoms, position);
  }
}
__global__ void reconstruct_velocities(RigidBodySystem::ConstDeviceView view, double* velocities)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= view.number_of_bodies)
    return;
  const auto& g = view.geometry[b];
  const auto& s = view.state[b];
  // conjugate(q)*p = (0,2*L_body)，直接求 omega，避免 L 在两个坐标系之间往返。
  const Quaternion qp = multiply(conjugate(s.quat), s.conjqm);
  const Vec3 omega =
    rotate(s.quat, {0.5 * qp.x / g.inertia.x, 0.5 * qp.y / g.inertia.y, 0.5 * qp.z / g.inertia.z});
  for (int j = g.first_member; j < g.first_member + g.natoms; ++j) {
    const Vec3 r = rotate(s.quat, view.member_displace[j]);
    write_device_vec(
      velocities, view.member_atoms[j], view.number_of_atoms, add(s.vcm, cross(omega, r)));
  }
}
} // namespace

void RigidBodySystem::clear()
{
  geometry_.clear();
  state_.clear();
  load_.clear();
  member_atoms_.clear();
  member_displace_.clear();
  atom_to_body_.clear();

  initialized_ = false;
  number_of_atoms_ = 0;
  number_of_bodies_ = 0;
  number_of_members_ = 0;
  initialized_box_ = {};
  initialized_pbc_ = {};
}
void RigidBodySystem::initialize(
  const Atom& atom, const Box& box, const std::vector<int>& body_labels)
{
  initialize_impl(atom, box, body_labels, nullptr);
}

void RigidBodySystem::initialize(
  const Atom& atom,
  const Box& box,
  const std::vector<int>& body_labels,
  const std::vector<double>& unwrapped_positions)
{
  initialize_impl(atom, box, body_labels, &unwrapped_positions);
}

RigidBodySystem::HostSnapshot RigidBodySystem::create_bodies(
  int number_of_atoms, const std::vector<int>& body_labels, std::vector<int>& atom_to_body)
{
  std::vector<std::pair<int, int>> labeled_atoms;
  labeled_atoms.reserve(number_of_atoms);
  for (int i = 0; i < number_of_atoms; ++i)
    if (body_labels[i] >= 0)
      labeled_atoms.emplace_back(body_labels[i], i);
  std::sort(labeled_atoms.begin(), labeled_atoms.end());

  HostSnapshot snapshot;
  snapshot.member_atoms.reserve(labeled_atoms.size());
  snapshot.member_displace.resize(labeled_atoms.size());
  atom_to_body.assign(number_of_atoms, -1);
  for (const auto& member : labeled_atoms) {
    if (snapshot.bodies.empty() || member.first != snapshot.bodies.back().geometry.id) {
      Body body{};
      body.geometry.id = member.first;
      body.geometry.first_member = static_cast<int>(snapshot.member_atoms.size());
      snapshot.bodies.push_back(body);
    }
    ++snapshot.bodies.back().geometry.natoms;
    snapshot.member_atoms.push_back(member.second);
    atom_to_body[member.second] = static_cast<int>(snapshot.bodies.size()) - 1;
  }
  for (const auto& body : snapshot.bodies)
    require(body.geometry.natoms > 1, "Single-bead molecules must use body label -1.");
  number_of_atoms_ = number_of_atoms;
  number_of_bodies_ = static_cast<int>(snapshot.bodies.size());
  number_of_members_ = static_cast<int>(snapshot.member_atoms.size());
  return snapshot;
}

void RigidBodySystem::initialize_impl(
  const Atom& atom,
  const Box& box,
  const std::vector<int>& body_labels,
  const std::vector<double>* supplied_positions)
{
  clear();
  try {
    const int N = atom.number_of_atoms;
    require(
      N >= 0 && body_labels.size() == static_cast<std::size_t>(N),
      "body_labels must contain one label per atom.");
    for (int label : body_labels)
      require(label >= -1, "A body label must be -1 or nonnegative.");
    const std::size_t count = static_cast<std::size_t>(N);
    require(
      atom.mass.size() == count && atom.velocity_per_atom.size() == 3 * count,
      "Rigid initialization requires current atom mass and velocity.");
    const Cell cell(box);
    if (supplied_positions) {
      require(
        supplied_positions->size() == 3 * count,
        "Unwrapped positions must be a HOST 3N SoA array.");
    } else {
      require(
        atom.position_per_atom.size() == 3 * count,
        "Rigid initialization requires current atom positions.");
    }
    std::vector<int> atom_to_body;
    HostSnapshot snapshot = create_bodies(N, body_labels, atom_to_body);

    if (number_of_bodies_ > 0) {
      std::vector<double> positions(3 * count), masses(count), velocities(3 * count);
      if (count > 0) {
        atom.mass.copy_to_host(masses.data());
        atom.velocity_per_atom.copy_to_host(velocities.data());
      }
      const auto& members = snapshot.member_atoms;
      if (supplied_positions) {
        positions = *supplied_positions;
      } else {
        // 默认以参考 bead 最小镜像恢复紧凑分子的局部几何。
        if (count > 0)
          atom.position_per_atom.copy_to_host(positions.data());
        for (int b = 0; b < number_of_bodies_; ++b) {
          const auto& g = snapshot.bodies[b].geometry;
          const int reference = members[g.first_member];
          const Vec3 anchor = read_vec(positions, reference, N);
          require(finite(anchor), "Non-finite bead position.");
          Vec3 low{0, 0, 0}, high{0, 0, 0};
          for (int j = g.first_member; j < g.first_member + g.natoms; ++j) {
            const int i = members[j];
            const Vec3 position = read_vec(positions, i, N);
            require(finite(position), "Non-finite bead position.");
            Vec3 f = cell.fractional(subtract(position, anchor));
            require(finite(f), "Non-finite relative bead coordinate.");
            if (cell.periodic[0])
              f.x -= std::floor(f.x + 0.5);
            if (cell.periodic[1])
              f.y -= std::floor(f.y + 0.5);
            if (cell.periodic[2])
              f.z -= std::floor(f.z + 0.5);
            low = {std::min(low.x, f.x), std::min(low.y, f.y), std::min(low.z, f.z)};
            high = {std::max(high.x, f.x), std::max(high.y, f.y), std::max(high.z, f.z)};
            write_vec(positions, i, N, add(anchor, cell.cartesian(f)));
          }
          const double limit = 0.5 - 64 * std::numeric_limits<double>::epsilon();
          require(
            (!cell.periodic[0] || high.x - low.x < limit) &&
              (!cell.periodic[1] || high.y - low.y < limit) &&
              (!cell.periodic[2] || high.z - low.z < limit),
            "Ambiguous/noncompact body: supply coherent unwrapped positions explicitly.");
        }
      }
      setup_bodies_static(snapshot, box, positions, masses);
      setup_bodies_dynamic(snapshot, masses, velocities);
    }

    // HOST 工作区到此才上传：静态/动态初始化之间没有重复 GPU 往返。
    geometry_.resize(number_of_bodies_);
    state_.resize(number_of_bodies_);
    load_.resize(number_of_bodies_);
    member_atoms_.resize(number_of_members_);
    member_displace_.resize(number_of_members_);
    atom_to_body_.resize(count);
    if (number_of_bodies_ > 0) {
      std::vector<BodyGeometry> geometry;
      std::vector<BodyState> states;
      std::vector<BodyLoad> loads;
      geometry.reserve(number_of_bodies_);
      states.reserve(number_of_bodies_);
      loads.reserve(number_of_bodies_);
      for (const auto& body : snapshot.bodies) {
        geometry.push_back(body.geometry);
        states.push_back(body.state);
        loads.push_back(body.load); // Body{} 保证初始载荷为零。
      }
      geometry_.copy_from_host(geometry.data());
      state_.copy_from_host(states.data());
      load_.copy_from_host(loads.data());
      member_atoms_.copy_from_host(snapshot.member_atoms.data());
      member_displace_.copy_from_host(snapshot.member_displace.data());
    }
    if (count > 0)
      atom_to_body_.copy_from_host(atom_to_body.data());
    std::copy(box.cpu_h, box.cpu_h + 9, initialized_box_.begin());
    initialized_pbc_ = {box.pbc_x, box.pbc_y, box.pbc_z};
    initialized_ = true;
  } catch (...) {
    clear();
    throw;
  }
}

void RigidBodySystem::setup_bodies_static(
  HostSnapshot& snapshot,
  const Box& box,
  const std::vector<double>& positions,
  const std::vector<double>& masses)
{
  const int N = number_of_atoms_;
  const Cell cell(box);
  const auto& members = snapshot.member_atoms;
  auto& displace = snapshot.member_displace;
  for (int b = 0; b < number_of_bodies_; ++b) {
    auto& g = snapshot.bodies[b].geometry;
    auto& state = snapshot.bodies[b].state;
    const Vec3 anchor = read_vec(positions, members[g.first_member], N);
    require(finite(anchor), "Non-finite bead position.");
    Vec3 weighted{0, 0, 0};
    for (int j = g.first_member; j < g.first_member + g.natoms; ++j) {
      const int i = members[j];
      require(std::isfinite(masses[i]) && masses[i] > 0, "Bead mass must be finite and positive.");
      const Vec3 position = read_vec(positions, i, N);
      require(finite(position), "Non-finite bead position.");
      g.mass += masses[i];
      weighted = add(weighted, scale(subtract(position, anchor), masses[i]));
    }
    require(
      std::isfinite(g.mass) && g.mass > 0 && finite(weighted),
      "Overflow computing rigid mass/center.");
    const Vec3 relative_center{weighted.x / g.mass, weighted.y / g.mass, weighted.z / g.mass};
    Vec3 center = add(anchor, relative_center);
    require(finite(center), "Non-finite body center.");
    cell.wrap(center);
    state.xcm = center;
    double tensor[3][3] = {};
    std::vector<Vec3> relative(g.natoms);
    for (int k = 0; k < g.natoms; ++k) {
      const int i = members[g.first_member + k];
      const Vec3 r = subtract(subtract(read_vec(positions, i, N), anchor), relative_center);
      require(finite(r), "Non-finite body relative position.");
      relative[k] = r;
      const double m = masses[i];
      tensor[0][0] += m * (r.y * r.y + r.z * r.z);
      tensor[1][1] += m * (r.x * r.x + r.z * r.z);
      tensor[2][2] += m * (r.x * r.x + r.y * r.y);
      tensor[0][1] -= m * r.x * r.y;
      tensor[0][2] -= m * r.x * r.z;
      tensor[1][2] -= m * r.y * r.z;
    }
    tensor[1][0] = tensor[0][1];
    tensor[2][0] = tensor[0][2];
    tensor[2][1] = tensor[1][2];
    Vec3 axes[3];
    principal_axes(tensor, g.inertia, axes);
    state.quat = axes_to_quaternion(axes);
    g.rotational_dof = 3;
    for (int k = 0; k < g.natoms; ++k)
      displace[g.first_member + k] = inverse_rotate(state.quat, relative[k]);
  }
}

void RigidBodySystem::setup_bodies_dynamic(
  HostSnapshot& snapshot, const std::vector<double>& masses, const std::vector<double>& velocities)
{
  const int N = number_of_atoms_;
  const auto& members = snapshot.member_atoms;
  const auto& displace = snapshot.member_displace;
  for (int b = 0; b < number_of_bodies_; ++b) {
    const auto& g = snapshot.bodies[b].geometry;
    auto& state = snapshot.bodies[b].state;
    Vec3 momentum{0, 0, 0};
    for (int j = g.first_member; j < g.first_member + g.natoms; ++j) {
      const int i = members[j];
      const Vec3 v = read_vec(velocities, i, N);
      require(finite(v), "Non-finite bead velocity.");
      momentum = add(momentum, scale(v, masses[i]));
    }
    state.vcm = {momentum.x / g.mass, momentum.y / g.mass, momentum.z / g.mass};
    require(finite(state.vcm), "Overflow computing center velocity.");
    Vec3 L{0, 0, 0};
    for (int j = g.first_member; j < g.first_member + g.natoms; ++j) {
      const int i = members[j];
      const Vec3 r = rotate(state.quat, displace[j]);
      L = add(L, cross(r, scale(subtract(read_vec(velocities, i, N), state.vcm), masses[i])));
    }
    require(finite(L), "Overflow computing rigid angular momentum.");
    state.conjqm = angular_momentum_to_conjqm(state.quat, L);
    require(
      std::isfinite(state.conjqm.w) && std::isfinite(state.conjqm.x) &&
        std::isfinite(state.conjqm.y) && std::isfinite(state.conjqm.z),
      "Non-finite Miller conjugate momentum.");
  }
}

bool RigidBodySystem::initialized() const { return initialized_; }
int RigidBodySystem::number_of_bodies() const { return number_of_bodies_; }
int RigidBodySystem::number_of_members() const { return number_of_members_; }

RigidBodySystem::DeviceView RigidBodySystem::device_view()
{
  require_initialized();
  return {
    number_of_atoms_,
    number_of_bodies_,
    number_of_members_,
    geometry_.data(),
    state_.data(),
    load_.data(),
    member_atoms_.data(),
    member_displace_.data(),
    atom_to_body_.data()};
}
RigidBodySystem::ConstDeviceView RigidBodySystem::device_view() const
{
  require_initialized();
  return {
    number_of_atoms_,
    number_of_bodies_,
    number_of_members_,
    geometry_.data(),
    state_.data(),
    load_.data(),
    member_atoms_.data(),
    member_displace_.data(),
    atom_to_body_.data()};
}
void RigidBodySystem::copy_to_host(HostSnapshot& destination) const
{
  require_initialized();
  HostSnapshot snapshot;
  snapshot.bodies.resize(number_of_bodies_);
  snapshot.member_atoms.resize(number_of_members_);
  snapshot.member_displace.resize(number_of_members_);
  if (number_of_bodies_ > 0) {
    std::vector<BodyGeometry> geometry(number_of_bodies_);
    std::vector<BodyState> states(number_of_bodies_);
    std::vector<BodyLoad> loads(number_of_bodies_);
    geometry_.copy_to_host(geometry.data());
    state_.copy_to_host(states.data());
    load_.copy_to_host(loads.data());
    for (int b = 0; b < number_of_bodies_; ++b)
      snapshot.bodies[b] = {geometry[b], states[b], loads[b]};
    member_atoms_.copy_to_host(snapshot.member_atoms.data());
    member_displace_.copy_to_host(snapshot.member_displace.data());
  }
  destination = std::move(snapshot);
}

void RigidBodySystem::require_initialized() const
{
  if (!initialized_)
    throw std::logic_error("Rigid body system is not initialized.");
}
void RigidBodySystem::check_atom(const Atom& atom) const
{
  require_initialized();
  require(
    atom.number_of_atoms == number_of_atoms_,
    "Atom count changed; rigid body membership must be reinitialized.");
}
void RigidBodySystem::check_box(const Box& box) const
{
  require_initialized();
  require(
    std::equal(initialized_box_.begin(), initialized_box_.end(), box.cpu_h) &&
      initialized_pbc_ == std::array<int, 3>{box.pbc_x, box.pbc_y, box.pbc_z},
    "Rigid bodies currently require the fixed box used at initialization.");
}
void RigidBodySystem::compute_forces_and_torques(const Atom& atom)
{
  check_atom(atom);
  if (number_of_bodies_ == 0)
    return;
  require(
    atom.force_per_atom.size() == 3 * static_cast<std::size_t>(number_of_atoms_),
    "Rigid force reduction requires a 3N force array.");
  const auto view = static_cast<const RigidBodySystem&>(*this).device_view();
  sum_body_loads<<<(number_of_bodies_ + 127) / 128, 128>>>(
    view, atom.force_per_atom.data(), load_.data());
  GPU_CHECK_KERNEL
}
void RigidBodySystem::remap_centers(const Box& box)
{
  check_box(box);
  if (number_of_bodies_ == 0)
    return;
  wrap_body_centers<<<(number_of_bodies_ + 127) / 128, 128>>>(device_view(), Cell(box));
  GPU_CHECK_KERNEL
}
void RigidBodySystem::rebuild_positions(Atom& atom, const Box& box)
{
  check_atom(atom);
  check_box(box);
  if (number_of_bodies_ == 0)
    return;
  require(
    atom.position_per_atom.size() == 3 * static_cast<std::size_t>(number_of_atoms_),
    "Rigid reconstruction requires a 3N position array.");
  require(
    atom.unwrapped_position.size() == 0,
    "Rigid bodies do not track continuous trajectories; unwrapped output is unsupported.");
  const auto view = static_cast<const RigidBodySystem&>(*this).device_view();
  reconstruct_positions<<<(number_of_bodies_ + 127) / 128, 128>>>(
    view, Cell(box), atom.position_per_atom.data());
  GPU_CHECK_KERNEL
}
void RigidBodySystem::rebuild_velocities(Atom& atom)
{
  check_atom(atom);
  if (number_of_bodies_ == 0)
    return;
  require(
    atom.velocity_per_atom.size() == 3 * static_cast<std::size_t>(number_of_atoms_),
    "Rigid reconstruction requires a 3N velocity array.");
  const auto view = static_cast<const RigidBodySystem&>(*this).device_view();
  reconstruct_velocities<<<(number_of_bodies_ + 127) / 128, 128>>>(
    view, atom.velocity_per_atom.data());
  GPU_CHECK_KERNEL
}
