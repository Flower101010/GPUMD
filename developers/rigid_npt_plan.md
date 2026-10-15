# 刚体 NPT 实现方案与开发路径

状态：方案讨论，尚未实现。2026-10-15 根据用户目标修订：以 LAMMPS 刚体
NPT 的 Nose–Hoover/MTK 扩展系统为主要参考；首版支持非正交全周期盒的
各向同性体积耦合。此前 SCR 优先的建议不再作为实施基线。

现有 Miller NVE、单全局 NHC NVT 的实现和运行行为保持不变。

## 第一版范围与算法依据

- 参考本地 LAMMPS `src/RIGID/fix_rigid_nh_small.cpp` 的刚体 NPT 方程、
  MTK 修正、恒温链和算子顺序，结合 GPUMD 的力计算边界实现 GPU 版本。
- 正交或非正交的三维全周期盒，只有一个各向同性体积自由度。
- 保持盒形、角度与晶格向量长度比，不支持独立边长耦合或剪切。
- 自由单 bead 与任意成员数的非线性刚体混合，沿用现有成员关系。
- 首版固定步长、恒定目标 T/P；斜坡、部分膨胀、精确 checkpoint 后续再做。
- 保留 Miller 自由转动原语；与参考 NPT 算子核对兼容性，不因为 NVE
  已通过就假定整个 NPT 转动分裂自动正确。

重要边界：当前 LAMMPS `rigid/npt`、`rigid/nph` 及 small 版本明确不支持
triclinic 盒；本地代码在 `pstat_flag` 分支主动报错。普通 `fix npt` 支持
triclinic。我们参考刚体方程并扩展各向同性盒映射，不声称直接复制
LAMMPS 已有的 triclinic 刚体 NPT 功能，也不承诺逐步轨迹与 LAMMPS 相同。

## 为什么改用 MTK 作为主线

SCR 和 MTK 都有正确 NPT 分布的理论依据，离散实现仍需数值验收。
不存在脱离体系、参数和实现的普遍优劣排序。

| 比较 | SCR | Nose–Hoover/MTK |
| --- | --- | --- |
| 体积演化 | 随机一阶、过阻尼型 | 带活塞惯性的扩展系统 |
| 所需状态 | 体积相关标量、随机状态 | 体积变量/动量、质量、恒温链 |
| 初始松弛 | 论文强调平滑、适合非平衡初态 | 耦合不当或初态过差时可能振荡 |
| 参考实现 | GPUMD SCR、作者参考代码 | LAMMPS rigid/npt/small、GPUMD MTTK |
| 本项目目标 | 合理的备选实现 | 与用户学习路线和 LAMMPS 功能目标更一致 |

此前 SCR 优先主要基于代码量和模块复用，不是基于它比 MTK 更准确。
该理由不足以改变用户已明确的算法目标。SCR 保留为未来可选方案，
不在首版并行实现两套 barostat。

## 非正交盒的各向同性几何映射

设初始完整盒矩阵为 H0，eta = log(V/V0)/3，则 H = exp(eta)*H0。
一次增量映射使用 s = exp(delta_eta)：

- H_new = s*H_old，完整 9 个矩阵元素同比更新，包括非零倾斜分量。
- H_new_inverse = H_old_inverse/s，V_new = s^3*V_old；刷新 Box 派生缓存。
- 自由粒子位置与刚体 COM 围绕约定固定原点同比缩放。
- 固定 body-frame 位移、质量和惯量不变，从质心和姿态重建 bead。
- 几何缩放算子不改变姿态；自由转动子步仍正常推进四元数。
- 分数坐标映射使用完整 H_inverse，不写仅适用于对角盒的分支。

这比正交盒增加的主要是完整矩阵和斜盒测试，不增加剪切自由度。
复杂度主要来自 NPT 方程与时间层，而不是这个几何扩展。

## 必须先审核的动力学约定

### 平动、转动与活塞的不同耦合

保持计数明确：

- N_objects = N_free + N_body。
- f_translation = 3*N_objects。
- f_rotation = 3*N_body；f_thermal = f_translation + f_rotation。

本地 LAMMPS 把刚体平动和转动动能分别送入两条恒温链，并另有活塞链。
现有 NVT 是一条全局链，是有效的已有方案，但不是该参考的完整复刻。
NPT 先按平动/转动/活塞分别列出方程；自由粒子归入平动部分，不能作为
虚构的单成员旋转刚体。具体链组合和所控 DOF 在方程审核阶段定稿。

逐项核对 `initial_integrate/final_integrate` 中的 `scale_t/scale_r`，
以及 `nh_epsilon_dot()` 的 `mtk_term1/mtk_term2`。MTK 修正可影响 Miller
共轭动量，不能沿用 SCR 方案中“体积映射不缩放转动动量”的规则。
活塞质量、三个耦合轴归一化到单个 eta 的系数必须从参考定义推导；
不能直接把轴质量相加，也不能照搬某个原子数公式。

### 压力、约束 virial 和测度

在确定参考方程后，统一控制压力、输出压力与 MTK 修正项。
当前 `RigidThermo` 是固定盒物理半步的离散约束冲量估计，不能将包含
体积/恒温动量变化的速度差也解释为约束力。

分子质心缩放的虚功关系仍应独立验证：

W_molecular = W_bead - sum_over_rigid_members(rho_i dot force_i)。

它提供体积能量差分的验证途径；是否将其对应的分子压力用于控制，要与
选定 MTK 方程一起确定。不能把 SCR 的控制压力、LAMMPS 的 MTK 修正和
另一份约束 virial 直接拼接，造成遗漏或重复计数。

全局 COM 约束、有限体系测度、热自由度与活塞驱动力中的计数，都须在
解析理想气体测试前写清楚。NEP 的力、能量、原子 virial 不改写。

### 时间层与力计算

按 LAMMPS 的 `setup -> initial_integrate -> final_integrate` 梳理标量方程，
映射到 `compute1 -> Force::compute -> compute2`。每次 kick 使用配套
位置和盒子的力；盒缩放后的势能/virial 不沿用旧状态。
先列出完整算子顺序及输入输出，再优化 GPU kernel、归约与同步。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| RigidBodySystem | 保留固定盒默认检查，增加受控等比盒更新和完整 H remap/rebuild，不重新初始化几何 |
| rigid_body_integrator | 区分平动/转动动量因子，实现带活塞耦合的 COM/自由粒子传播，保留 Miller 转动原语 |
| 刚体热力学统计 | GPU 汇总分开的 KE、当前压力及正确时间层 virial，提供独立验证估计器 |
| NHC 标量工具 | 尽量复用链推进，明确不同链的动量/摩擦系数表示和自由度 |
| 各向同性 MTK 标量状态 | eta、活塞动量或速率、质量、修正项和活塞链，不管理 bead |
| Ensemble_Rigid_NPT | 组合算子，安排力与统计阶段，不复制整份 NVT 或原子 MTTK |
| 上层输入与输出 | 目标/耦合时间单位、体积/密度、扩展系统诊断和 run/restart 语义 |

运行期成员遍历与归约保留在 GPU，CPU 只处理少量全局标量。不引入
六分量应力控制、MPI 迁移、动态成员或通用插件框架。

## 开发顺序与验收

1. **方程对照。** 记录 LAMMPS 变量、公式、DOF、时间层与我们的表示；
   先审核单个 eta 的归一化、混合体系和 Miller 子步。
2. **可变盒原语。** 正交、斜盒、非零倾斜量、膨胀/压缩和跨边界；
   验证分数 COM 坐标、盒角保持、内部距离、惯量和重建速度。
3. **压力与虚功。** 用固定内部几何的体积扰动差分检查 NEP 小体系，
   区分约束/活塞/恒温项，验证压力与 MTK 修正的完整组合。
4. **标量 MTK 与组合积分。** 验证链、活塞和修正项，冻结活塞退化到
   已定义的 NVT 极限；检查可逆性、步长收敛及扩展系统能量诊断。
   若 NPT 使用分开的物理链，极限比较对应方程，不要求逐步复现现有
   单链 NVT 的轨迹。
5. **上层接入。** 初始化、邻居表、整步 thermo/XYZ、连续 run 与明确的
   restart 语义，既有 NVE/NVT 回归通过。
6. **统计验收。** 小规模正交盒对照 LAMMPS；斜盒用体积虚功、等价
   晶格基表示和解析理想分子气体检验，再做 mixed NEP smoke/long。

温压均值、动能分配、体积/密度、体积涨落和分块稳定性均在明确平衡
窗口中检查。对照前统一链长、耦合参数、DOF/COM、压力定义和单位。
不以平均压力达标替代 NPT 统计，不要求物理 K+U 或 K+U+PV 严格守恒。
长测试由用户执行，不运行 3000 刚体大案例。

## 参考

- [LAMMPS rigid NPT 与 triclinic 限制](https://docs.lammps.org/fix_rigid.html)
- [普通 fix npt 的 MTK 与 tilt scaling](https://docs.lammps.org/fix_nh.html)
- 本地 `lammps/src/RIGID/fix_rigid_nh_small.cpp` 的 setup、initial_integrate、
  final_integrate、compute_dof 和 nh_epsilon_dot。
- GPUMD `src/integrate/ensemble_mttk.cu` 用于盒数据、单位和运行边界参考；
  其原子缩放不能直接用于刚体。
- [SCR 备选方案原论文](https://arxiv.org/abs/2006.09250)。
