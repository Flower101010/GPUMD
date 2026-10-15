# 刚体 NPT 实现方案与开发路径

状态：设计建议，尚未实现。现有 Miller NVE、NHC NVT 及其运行行为保持不变。

## 第一版范围与算法

建议先实现各向同性 SCR barostat + 现有 NHC thermostat + Miller 刚体积分：

- 三维正交盒、三个方向均周期，只改变体积、保持盒形。
- 自由单 bead 与任意成员数的非线性刚体混合；沿用现有分组适配。
- 固定步长、恒定目标温度和静水压力；后续再扩展斜坡。
- 一个全局 barostat；提供显式随机种子。
- 不在首版加入各向异性、剪切、非周期压强耦合或新的 restart 系统。

SCR 的依据是 Bernetti–Bussi 的随机盒缩放方法，目标包括正确的体积
涨落，而不是仅把平均压力调到指定值。GPUMD 的 `Ensemble_NPT_SCR` 已有
标量压力反馈、随机数和盒缩放的参考实现，但它还绑定了 BDP thermostat
和原子级缩放，不能作为刚体 NPT 直接继承使用。

MTK/MTTK 是可选的第二条路线。它延续 Nose–Hoover 扩展系统思想，但必须
额外处理活塞动量、MTK 测度修正、平动/转动不同的动量耦合以及可能的
活塞恒温链。若选择该路线，应单独推导，不能与 SCR 的映射规则混用。
本地 LAMMPS `fix_rigid_nh_small.cpp` 的 `mtk_term1/2` 和 `scale_t/scale_r`
说明转动动量也可能参与 MTK 修正，不能假定 barostat 永远不改变它。

## 必须先确定的物理约定

### 1. 缩放分子位置，保持分子内部几何

设 `s=(V_new/V_old)^(1/3)`，采用盒原点为缩放中心：

- `H <- s*H`；自由粒子位置和刚体 `xcm` 同比缩放。
- 固定 body-frame 位移、质量、惯量和四元数保持不变。
- 从新的 `xcm` 和原有姿态重建 bead；不直接缩放全部 bead 坐标。
- 首选 SCR 的位置/平动动量互为逆缩放版本：自由粒子速度和刚体 `vcm`
  乘 `1/s`；这个独立体积映射不缩放 Miller 共轭动量。
- NHC 仍按原有规则同时缩放所有平动和转动动量。

SCR 也有不缩放动量的版本，但其压力表达式不同。第一阶段必须明确选定
一个版本并完成推导，不从不同实现中各取一半。

### 2. 体积驱动力必须与上述缩放共轭

分别记录：

`N_objects = N_free + N_body`

`f_translation = 3*N_objects`

`f_thermal = 3*N_free + 6*N_body`

三者用途不同，不能统一替换成 bead 数或总热自由度。

对上述分子质心缩放，待验证的标量压力为：

`P_molecular = (2*K_translation + W_molecular)/(3*V)`

`W_molecular = W_bead - sum_over_rigid_members(rho_i dot force_i)`

其中 `rho_i` 是空间中的 COM 相对位移；`W_bead` 是当前盒和坐标下的模型
virial 的迹，符号按 GPUMD 约定。NEP 力、能量和原子 virial 数组不改写。
此式须通过保持姿态/内部几何的体积微扰能量差分独立验证。

现在 `RigidThermo` 的约束 virial 是固定盒物理半步的离散速度差估计。
它可作为已有诊断和交叉验证，不能未经论证就用上一整步结果驱动当前
barostat。NPT 控制压力及输出标量压力应采用一致的定义和时间层。
张量输出也须说明估计器，不能拼接不同盒体积、不同阶段的贡献。

### 3. 力、盒和坐标必须对应

缩放位置后，旧力和旧 virial 不再是新状态的力/virial。先按参考 SCR
离散化确定算子顺序，再安排当前 `compute1 -> Force::compute -> compute2`
边界，保证每次 kick 使用配套状态的力，输出对应最终状态。
不能直接在 NVT 两端插入改变坐标的操作却继续沿用旧力；若正确的首版
需要额外力计算，应明确实现和计时，再优化。

## 代码职责

| 模块 | 改动职责 |
| --- | --- |
| `RigidBodySystem` | 保留固定盒默认约束；增加显式授权的各向同性盒更新，校验 PBC/盒形、更新当前盒记录、remap/rebuild；几何不重新初始化 |
| `rigid_body_integrator` | 新增质心/自由粒子缩放；动量缩放区分平动因子与转动因子，单因子 NHC 接口继续可用 |
| 刚体压力统计 | GPU 汇总平动/转动 KE、分子 virial 与标量压力；不依赖历史约束冲量猜测当前压力 |
| 共用刚体 NHC 操作 | 从 NVT 的恒温半步提取必要公共部分，继续调用已有 `nose_hoover_chain::propagate` |
| SCR 标量推进 | 持有随机状态、耦合参数和体积推进；不依赖 Atom/刚体成员，不自行计算模型力 |
| `Ensemble_Rigid_NPT` | 组合上述操作，安排时间层与同步；不复制整份 NVT，也不继承原子 NPT 的缩放路径 |
| 输入、运行、输出 | 接入独立 `rigid_npt_scr` 入口；压力单位、压缩率/弹性模量及耦合时间约定写清；增加体积/密度及统计记录 |

运行期成员遍历与归约保留在 GPU；CPU 只处理少量标量与一个全局随机过程。
不为当前需求引入通用插件体系或提前实现完整六分量 barostat。

## 分阶段交付

1. **确定数学与接口契约。** 固定 SCR 变体、随机离散化、压力定义、
   时间层、随机种子及输入单位；写出每个算子的输入/输出与目标分布。
2. **可变盒几何原语。** 测试膨胀/压缩、跨周期边界、质量/惯量/内部
   距离保持不变、动量缩放及重建一致性。NVE/NVT 继续拒绝意外盒变化。
3. **分子压力。** 独立体积能量差分、理想刚体转动不贡献气体平动压力、
   全自由粒子退化情况和 NEP 小体系。通过后才让压力进入反馈环。
4. **独立 barostat 与组合积分。** 先验证随机推进、有限正体积、可复现
   seed，再组合 NHC/Miller。验证关闭压力耦合退化为现有 NVT、每次力
   计算与盒/坐标一致、缩放不混入约束冲量统计。
5. **上层接入。** 初始化在首次 Force::compute 之前；检查盒变化后的
   邻居表处理、thermo/XYZ 同步、连续 run 和 restart 的明确语义。
6. **统计验收。** CPU/GPU 小测试与 smoke 先行；mixed 体系 long 由用户
   运行；温度、压力、密度、体积及体积涨落均在明确平衡窗口内分析。

## NPT 验收重点

- 不要求每帧压力等于目标；使用分块均值、误差和稳定性判据。
- 不用单一 NEP 人工体系的未知真实压缩率硬判体积方差。
- 使用有解析 NPT 体积分布的理想分子气体检查平均体积与体积涨落，
  明确是否约束全局 COM，统一有限体系测度，验证 `N_objects` 而非 bead
  数决定平动气体行为。
- 体积方差和温压状态的联合分布才是 NPT 统计的重要补充；平均 P/T
  达标不能替代它们。相互作用小体系可再与独立 NPT 参考比较。
- 检查减半步长、改变压力耦合时间、不同随机种子的统计一致性。
- NEP mixed 用例保留刚体约束、力/能量静态复算、温度分配及原始轨迹。
- 不以物理系统 `K+U` 或 `K+U+P*V` 的严格守恒作为 NPT 验收条件。
- 不执行 3000 刚体大案例；小规模全刚体/全自由粒子是单独的边界测试。

## 参考资料

- [Bernetti–Bussi SCR 原论文](https://arxiv.org/abs/2006.09250)
- [作者参考实现](https://github.com/bussilab/crescale)
- [GPUMD ensembles](https://gpumd.org/theory/ensembles.html)
- [LAMMPS rigid NPT](https://docs.lammps.org/fix_rigid.html)
- 本地 `src/integrate/ensemble_npt_scr.cu`、`ensemble_mttk.cu` 及 LAMMPS
  `src/RIGID/fix_rigid_nh_small.cpp` 用于实现比对；不直接复制它们的原子
  缩放、自由度计数或压力估计。
