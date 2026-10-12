#pragma once

#include "utilities/gpu_vector.cuh"
#include "utilities/math.cuh"
#include <vector>

using rigid_math::Quaternion;
using rigid_math::Vec3;

class Atom;
class Box;

class RigidBodySystem
{
public:
  struct BodyGeometry {
    int id;                // 非负外部标签，区别于数组中的内部 body 索引
    int first_member;      // 扁平成员数组起点
    int natoms;            // 成员数 >= 2；范围 [first_member, first_member+natoms)
    double mass;           // 总质量 M；sum(m_i*s_i)=0
    Vec3 inertia;          // 主惯量，有限/非负；数值零轴精确存 0
    int rotational_dof;    // 按惯量秩确定；当前仅支持非线性刚体（3）
  };

  struct BodyState {
    Vec3 xcm;          // 空间质心位置；周期方向的分数坐标在 [0,1)
    Vec3 vcm;          // 空间质心速度
    Quaternion quat;   // 单位四元数 (w,x,y,z)，body -> space
    Quaternion conjqm; // Miller 共轭动量 p；quat dot conjqm=0
  };

  struct BodyLoad {
    Vec3 fcm;    // 已计算 bead 力的空间合力
    Vec3 torque; // 已计算 bead 力绕质心的空间力矩
  };

  // 一个刚体的逻辑记录；成员通过 geometry 的范围访问独立数组。
  struct Body {
    BodyGeometry geometry;
    BodyState state;
    BodyLoad load;
  };

  // 自包含 HOST 快照；BodyGeometry 的范围索引下面两个等长数组。
  // 避免只拷贝 Body，却遗漏其固定成员几何。快照不参与运行时演化。
  struct HostSnapshot {
    std::vector<Body> bodies;
    std::vector<int> member_atoms;
    std::vector<Vec3> member_displace;
  };

  // kernel 参数：非拥有 DEVICE 指针；初始化/clear/析构使旧 view 失效。
  // 几何与载荷对积分器只读，仅 state 可写；不在 HOST 上解引用。
  struct DeviceView {
    int number_of_atoms;
    int number_of_bodies;
    int number_of_members; // R=sum(geometry[b].natoms)，不含自由 bead
    const BodyGeometry* geometry;
    BodyState* state;
    const BodyLoad* load;
    const int* member_atoms;      // R 个 Atom 索引，每个刚体成员只出现一次
    const Vec3* member_displace;  // R 个固定主轴体坐标 s_i；与 member_atoms 一一对应
    const int* atom_to_body; // N 个内部 body 索引；自由 bead 为 -1
  };
  struct ConstDeviceView {
    int number_of_atoms;
    int number_of_bodies;
    int number_of_members;
    const BodyGeometry* geometry;
    const BodyState* state;
    const BodyLoad* load;
    const int* member_atoms;
    const Vec3* member_displace;
    const int* atom_to_body;
  };

  // GPU_Vector 所有权留在管理层，单个记录不拥有内存。
  // 初始化在 HOST 上构建几何与动态状态，再上传 GPU；不执行时间积分。
  RigidBodySystem();
  ~RigidBodySystem();
  RigidBodySystem(const RigidBodySystem&) = delete;
  RigidBodySystem& operator=(const RigidBodySystem&) = delete;

  // 从当前 Atom 坐标按参考 bead 最小镜像展开紧凑分子，再构建刚体。
  // 每个周期方向的成员分数坐标跨度必须 < 1/2；body_labels 中 -1 为自由 bead。
  // 非负标签至少对应两个 bead；单 bead 分子必须标记为 -1。
  void initialize(
    const Atom& atom,
    const Box& box,
    const std::vector<int>& body_labels);

  // 一般几何可显式提供 HOST 3N SoA 展开坐标；不受默认半盒跨度限制。
  void initialize(
    const Atom& atom,
    const Box& box,
    const std::vector<int>& body_labels,
    const std::vector<double>& unwrapped_positions);

  void clear();
  bool initialized() const;
  int number_of_bodies() const;
  int number_of_members() const;
  DeviceView device_view();
  ConstDeviceView device_view() const;

  // 每个 body 独立遍历自己的成员范围求和，每次只写自己的载荷，无需原子加法。
  // 1/3 bead 混合先采用每线程一个 body；以后长 body 可另选 warp/block 归约。
  // F=sum(f_i)，tau=sum((R(q)*s_i) x f_i)，每次覆盖旧载荷。
  // 力与当前位姿必须对应；不修改 Atom 的力/能量/virial。
  void compute_forces_and_torques(const Atom& atom);

  // 质心漂移后调用：s=H^-1*xcm，周期方向 s-=floor(s)，再转换回空间坐标。
  // 非周期方向不包装；只改 xcm，不改姿态、速度或固定几何，可合并到积分 kernel。
  void remap_centers(const Box& box);

  // 前置条件：质心已 remap。求局部位置 y_i=xcm+R(q)*s_i，再按完整盒矩阵包装。
  // 只写受管理 bead 的 position_per_atom，保持自由 bead 原值。
  // 刚体不维护跨盒历史；连续展开轨迹输出需另行设计。
  void rebuild_positions(Atom& atom, const Box& box);

  // 有转动自由度时从 quat/conjqm 派生空间 L/omega，v_i=vcm+omega x R(q)*s_i。
  // 只写刚体成员；不缓存 angmom/omega/主轴，避免额外同步 kernel。
  void rebuild_velocities(Atom& atom);

  // 仅用于诊断：整体状态与全部成员数组一起同步到 HOST。
  void copy_to_host(HostSnapshot& destination) const;

private:
  void initialize_impl(
    const Atom& atom, const Box& box, const std::vector<int>& body_labels,
    const std::vector<double>* unwrapped_positions);
  // 构建变长成员范围，分配 body 与成员数组；标签/Atom 索引变化需重新构建。
  void create_bodies(int number_of_atoms, const std::vector<int>& body_labels);
  // 从一致展开成员求 M/X_cm/惯性张量，再把质心包装到盒内。
  // 主惯量升序、右手主轴、quat 和固定 displace。
  // 重根主轴需可复现；初始化时用相对容差识别数值零惯量，不能在热路径猜测。
  void setup_bodies_static(
    const Atom& atom, const Box& box, const std::vector<double>& unwrapped_positions);
  // V_cm=sum(m_i*v_i)/M，L=sum(r_i x m_i*(v_i-V_cm))，p=2*q*(0,R(q)^T*L)。
  void setup_bodies_dynamic(const Atom& atom);

  bool initialized_ = false;
  int number_of_atoms_ = 0;
  int number_of_bodies_ = 0;
  int number_of_members_ = 0;
  GPU_Vector<BodyGeometry> geometry_;
  GPU_Vector<BodyState> state_;
  GPU_Vector<BodyLoad> load_;
  GPU_Vector<int> member_atoms_;
  GPU_Vector<Vec3> member_displace_;
  GPU_Vector<int> atom_to_body_;
};
