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
  Vec3 fractional(Vec3 v) const
  {
    return {
      inv[0] * v.x + inv[1] * v.y + inv[2] * v.z,
      inv[3] * v.x + inv[4] * v.y + inv[5] * v.z,
      inv[6] * v.x + inv[7] * v.y + inv[8] * v.z};
  }
  Vec3 cartesian(Vec3 v) const
  {
    return {
      h[0] * v.x + h[1] * v.y + h[2] * v.z,
      h[3] * v.x + h[4] * v.y + h[5] * v.z,
      h[6] * v.x + h[7] * v.y + h[8] * v.z};
  }
  void wrap(Vec3& v) const
  {
    Vec3 f = fractional(v);
    require(finite(f), "Non-finite fractional coordinate.");
    const double k[3] = {
      periodic[0] ? std::floor(f.x) : 0,
      periodic[1] ? std::floor(f.y) : 0,
      periodic[2] ? std::floor(f.z) : 0};
    // 用包装后的分数坐标恢复 Cartesian 坐标，不保留跨盒计数。
    v = cartesian({f.x - k[0], f.y - k[1], f.z - k[2]});
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
} // namespace

RigidBodySystem::RigidBodySystem() : initialized_(false), number_of_atoms_(0), number_of_bodies_(0)
{
}

RigidBodySystem::~RigidBodySystem() { clear(); }
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

void RigidBodySystem::create_bodies(int number_of_atoms, const std::vector<int>& body_labels)
{
  // initialize 已检查长度和标签。收集 (外部 body 标签, Atom 索引)，忽略自由 bead。
  std::vector<std::pair<int, int>> labeled_atoms;
  labeled_atoms.reserve(number_of_atoms);
  for (int i = 0; i < number_of_atoms; ++i) {
    if (body_labels[i] >= 0) {
      labeled_atoms.emplace_back(body_labels[i], i);
    }
  }
  // 标签升序，同一 body 内 Atom 索引升序；不按最大标签分配内存。
  std::sort(labeled_atoms.begin(), labeled_atoms.end());

  const int number_of_members = static_cast<int>(labeled_atoms.size());
  int number_of_bodies = 0;
  for (int i = 0; i < number_of_members; ++i) {
    if (i == 0 || labeled_atoms[i].first != labeled_atoms[i - 1].first) {
      ++number_of_bodies;
    }
  }

  // natoms 从 0 开始计数；物理字段由后续静态初始化填写。
  std::vector<BodyGeometry> geometry(number_of_bodies, BodyGeometry{});
  std::vector<int> member_atoms(number_of_members);
  std::vector<int> atom_to_body(number_of_atoms, -1);
  int body = -1;
  for (int i = 0; i < number_of_members; ++i) {
    const auto& member = labeled_atoms[i];
    if (i == 0 || member.first != labeled_atoms[i - 1].first) {
      ++body;
      geometry[body].id = member.first;
      geometry[body].first_member = i;
    }
    ++geometry[body].natoms;
    member_atoms[i] = member.second;
    atom_to_body[member.second] = body;
  }

  // 单 bead 分子走普通积分；直接传入的非负标签不得形成单成员 body。
  for (const auto& g : geometry) {
    require(g.natoms > 1, "Single-bead molecules must use body label -1.");
  }

  // 统一分配；state/load/displace 尚未填入物理状态，当前不能读取其内容。
  geometry_.resize(number_of_bodies);
  state_.resize(number_of_bodies);
  load_.resize(number_of_bodies);
  member_atoms_.resize(number_of_members);
  member_displace_.resize(number_of_members);
  atom_to_body_.resize(number_of_atoms);
  // 空数组跳过拷贝，避免空指针参与 GPU memcpy。
  if (number_of_bodies > 0) {
    geometry_.copy_from_host(geometry.data());
    member_atoms_.copy_from_host(member_atoms.data());
  }
  if (number_of_atoms > 0) {
    atom_to_body_.copy_from_host(atom_to_body.data());
  }

  number_of_atoms_ = number_of_atoms;
  number_of_bodies_ = number_of_bodies;
  number_of_members_ = number_of_members;
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
    create_bodies(N, body_labels);
    if (number_of_bodies_ == 0) {
      initialized_ = true;
      return;
    }

    std::vector<double> positions(3 * count);
    std::vector<BodyGeometry> geometry(number_of_bodies_);
    std::vector<int> members(number_of_members_);
    geometry_.copy_to_host(geometry.data());// 存在重复的数据传输。考虑让 create_bodies() 直接返回或保留 CPU 端的拓扑数据，避免上传后立即下载
    member_atoms_.copy_to_host(members.data());// 构造阶段，问题不大
    if (supplied_positions) {
      positions = *supplied_positions;
    } else {
      // 默认以参考 bead 最小镜像恢复紧凑分子的局部几何。
      atom.position_per_atom.copy_to_host(positions.data());
      for (int b = 0; b < number_of_bodies_; ++b) {
        const auto& g = geometry[b];
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
    setup_bodies_static(atom, box, positions);

    setup_bodies_dynamic(atom);
    // 使用 HOST 零载荷上传，完成后首次积分前仍需实际 NEP 力和力矩。
    std::vector<BodyLoad> loads(number_of_bodies_, BodyLoad{});
    load_.copy_from_host(loads.data());
    initialized_ = true;
  } catch (...) {
    clear();
    throw;
  }
}

void RigidBodySystem::setup_bodies_static(
  const Atom& atom, const Box& box, const std::vector<double>& positions)
{
  const int N = number_of_atoms_;
  const Cell cell(box);
  std::vector<double> masses(N);
  atom.mass.copy_to_host(masses.data());
  std::vector<BodyGeometry> geometry(number_of_bodies_);
  std::vector<BodyState> states(number_of_bodies_, BodyState{});
  std::vector<int> members(number_of_members_);
  std::vector<Vec3> displace(number_of_members_);
  geometry_.copy_to_host(geometry.data());
  member_atoms_.copy_to_host(members.data());
  for (int b = 0; b < number_of_bodies_; ++b) {
    auto& g = geometry[b];
    auto& state = states[b];
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
  geometry_.copy_from_host(geometry.data());
  state_.copy_from_host(states.data());
  member_displace_.copy_from_host(displace.data());
}

void RigidBodySystem::setup_bodies_dynamic(const Atom& atom)
{
  const int N = number_of_atoms_;
  std::vector<double> masses(N), velocities(3 * static_cast<std::size_t>(N));
  atom.mass.copy_to_host(masses.data());
  atom.velocity_per_atom.copy_to_host(velocities.data());
  std::vector<BodyGeometry> geometry(number_of_bodies_);
  std::vector<BodyState> states(number_of_bodies_);
  std::vector<int> members(number_of_members_);
  std::vector<Vec3> displace(number_of_members_);
  geometry_.copy_to_host(geometry.data());
  state_.copy_to_host(states.data());
  member_atoms_.copy_to_host(members.data());
  member_displace_.copy_to_host(displace.data());
  for (int b = 0; b < number_of_bodies_; ++b) {
    const auto& g = geometry[b];
    auto& state = states[b];
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
  state_.copy_from_host(states.data());
}

bool RigidBodySystem::initialized() const { return initialized_; }
int RigidBodySystem::number_of_bodies() const { return number_of_bodies_; }
int RigidBodySystem::number_of_members() const { return number_of_members_; }

RigidBodySystem::DeviceView RigidBodySystem::device_view()
{
  if (!initialized_)
    throw std::logic_error("Rigid body system is not initialized.");
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
  if (!initialized_)
    throw std::logic_error("Rigid body system is not initialized.");
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
  if (!initialized_)
    throw std::logic_error("Rigid body system is not initialized.");
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
