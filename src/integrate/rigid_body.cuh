#pragma once

#include "utilities/gpu_vector.cuh"
#include "utilities/math.cuh"
#include <vector>

using rigid_math::Vec3;
using rigid_math::Quaternion;

class Atom;
class Box;

// 固定三成员刚体管理层；单 GPU、固定盒、固定成员与质量。
// 首版使用 Miller：常驻转动状态为 quat/conjqm；管理层不推进时间、不调用 NEP。
class RigidBodySystem
{
public:
  static constexpr int beads_per_body = 3;

  struct BodyGeometry {
    int id;                        // 非负外部标签，区别于数组中的内部 body 索引
    int atoms[beads_per_body];      // 3 个 bead 的 Atom 索引；每个 bead 只属于一个刚体
    Vec3 displace[beads_per_body];  // 各 bead 相对质心的固定主轴体坐标 s_i
    double mass;                   // 总质量 M；sum(m_i*s_i)=0
    Vec3 inertia;                  // 主惯量，有限/非负；数值零轴精确存 0
    int rotational_dof;            // 按惯量秩确定，线性为 2，非线性为 3，重合点状为 0
  };

  struct BodyState {
    Vec3 xcm;                      // 连续展开的空间质心位置
    Vec3 vcm;                      // 空间质心速度
    Quaternion quat;               // 单位四元数 (w,x,y,z)，body -> space
    Quaternion conjqm;             // Miller 共轭动量 p；quat dot conjqm=0
  };

  struct BodyLoad {
    Vec3 fcm;                      // 已计算 bead 力的空间合力
    Vec3 torque;                   // 已计算 bead 力绕质心的空间力矩
  };

  // 一个刚体的逻辑记录/诊断快照；不作为重复常驻的 GPU 状态。
  struct Body {
    BodyGeometry geometry;
    BodyState state;
    BodyLoad load;
  };

  // kernel 参数：非拥有 DEVICE 指针；初始化/clear/析构使旧 view 失效。
  // 几何与载荷对积分器只读，仅 state 可写；不在 HOST 上解引用。
  struct DeviceView {
    int number_of_atoms;
    int number_of_bodies;
    const BodyGeometry* geometry;
    BodyState* state;
    const BodyLoad* load;
    const int* atom_to_body;        // N 个内部 body 索引；自由 bead 为 -1
  };
  struct ConstDeviceView {
    int number_of_atoms;
    int number_of_bodies;
    const BodyGeometry* geometry;
    const BodyState* state;
    const BodyLoad* load;
    const int* atom_to_body;
  };

  // 下列方法均待实现；GPU_Vector 所有权留在管理层，单个记录不拥有内存。
  RigidBodySystem();
  ~RigidBodySystem();
  RigidBodySystem(const RigidBodySystem&) = delete;
  RigidBodySystem& operator=(const RigidBodySystem&) = delete;

  // body_labels：HOST N 标签，-1 为自由 bead，其余非负，可不连续。
  // 按标签升序压缩为内部索引，每个标签必须恰好对应 3 个不同 bead。
  // unwrapped_positions：HOST 3N SoA；每个刚体成员处于一致周期像，调用方负责展开。
  // 当前 Atom DEVICE mass/velocity 是来源，不依赖 CPU 缓存；不改写 Atom。
  // 初始速度投影到 V_cm/L，然后由 q/L 构建 conjqm；非刚性速度在重建时消除。
  // 检查有限数、正质量、惯量、标签和尺寸；允许零个刚体。
  // 三个点质量不保证非线性几何，仍需验证惯量秩和零轴角动量。
  // 载荷清零；首次积分前必须计算并汇总实际 bead 力。失败不得留下 initialized 状态。
  void initialize(
    const Atom& atom, const Box& box,
    const std::vector<int>& body_labels,
    const std::vector<double>& unwrapped_positions);

  void clear();
  bool initialized() const;
  int number_of_bodies() const;
  DeviceView device_view();
  ConstDeviceView device_view() const;

  // 每个 body 独立读取 3 个 bead 的力，无需原子加法或通用 CSR 归约。
  // F=sum(f_i)，tau=sum((R(q)*s_i) x f_i)，每次覆盖旧载荷。
  // 力与当前位姿必须对应；不修改 Atom 的力/能量/virial。
  void compute_forces_and_torques(const Atom& atom);

  // x_unwrapped_i=xcm+R(q)*s_i；写回 wrapped position 和绝对 position_image。
  // 使用完整盒矩阵，保持自由 bead 原值；image 存储存在且不能重复累加旧周期像。
  // 若启用 Atom.unwrapped_position，也同步对应成员；和 Force::compute 的 PBC 配合。
  void rebuild_positions(Atom& atom, const Box& box);

  // 从 quat/conjqm 派生空间 L/omega，v_i=vcm+omega x R(q)*s_i。
  // 只写刚体成员；不缓存 angmom/omega/主轴，避免额外同步 kernel。
  void rebuild_velocities(Atom& atom);

  // 仅用于诊断：三个 GPU 数组组合为 HOST Body 快照，不维持第二份演化状态。
  void copy_bodies_to_host(std::vector<Body>& destination) const;

private:
  // 构建固定三成员关系并分配三个 GPU 数组。
  void create_bodies(int number_of_atoms, const std::vector<int>& body_labels);
  // 求 M/X_cm/惯性张量；主惯量升序、右手主轴、quat 和固定 displace。
  // 重根主轴需可复现；初始化时用相对容差识别数值零惯量，不能在热路径猜测。
  void setup_bodies_static(
    const Atom& atom, const Box& box, const std::vector<double>& unwrapped_positions);
  // V_cm=sum(m_i*v_i)/M，L=sum(r_i x m_i*(v_i-V_cm))，p=2*q*(0,R(q)^T*L)。
  void setup_bodies_dynamic(const Atom& atom);

  bool initialized_ = false;
  int number_of_atoms_ = 0;
  int number_of_bodies_ = 0;
  GPU_Vector<BodyGeometry> geometry_;
  GPU_Vector<BodyState> state_;
  GPU_Vector<BodyLoad> load_;
  GPU_Vector<int> atom_to_body_;
};

// Miller NVE 的阶段由 Ensemble_Rigid_NVE 实施：
// 初始化并重建 bead -> 计算 bead 力 -> 汇总初始载荷。
// compute1：平动/力矩半步 kick -> 质心整步漂移 -> Miller 3-2-1-2-3 自由转动
//           -> 重建 bead 位置（以及需要暴露给其他模块的半步速度）。
// 力模块计算新 bead 力；compute2：汇总新载荷 -> 第二次半步 kick -> 重建整步速度。
// 每个线程处理一个 body；逻辑操作可合并到阶段 kernel，避免每个接口单独启动 kernel。
// quat/conjqm 是唯一权威转动状态；派生量不持久化，积分器维护范数和切空间约束。
// 线性/点状退化情形必须单独验证 Miller 的约束处理，不能仅靠跳过零惯量宣称支持。
// 自由 bead 走普通积分，刚体成员不能再次进入逐 bead velocity-Verlet。
// 外部速度修改必须更新整体状态；约束 virial/温度 DOF 由上层处理。
// Atom 索引、成员、质量变化需重新初始化；不移植 MPI/ghost/温压控制字段。
