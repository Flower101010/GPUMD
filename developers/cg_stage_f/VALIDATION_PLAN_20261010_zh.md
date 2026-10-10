# 阶段 F 验证方案：公开数据与独立参考

规划日期：2026-10-09；拟于 2026-10-10 开始执行。
当前状态：2026-10-10 已按修订范围执行计算正确性验证，结果见 [STATUS.md](STATUS.md)。
下文原始科学验证方案保留作历史参考，不是当前验收条件。
基线为阶段 E 完成后的开发分支；[已有验收](../cg_stage_e/STATUS.md)不能替代本计划。

## 2026-10-10 验收范围修订（优先于下文历史方案）

用户明确要求只验收计算和软件接口正确性，不将 NEP 拟合质量及科学上的模型效果混入
功能验收。下文公开 force matching、泛化、分布、折叠/物理效果属于历史候选研究方案，
从本次通过条件和执行清单中移除，不开展拟合质量优化。

本次执行：独立软件 E/F/W 对照，混合大小/拓扑、共享参数、batch/stream/预测与 MD
一致性，以及 GPU 内存和重复运行。公开数据仅可提供独立构型来源；自行构造测试几何
也允许，但参考答案必须由独立软件/独立数值导数给出。

采用 OpenMM native HarmonicBond/Angle/PeriodicTorsion 的
Reference CPU 实现；它不调用 GPUMD evaluator。virial 由其能量对应变的数值导数核对。
用户已安装 GROMACS 2026.3 double，三个已有 NVE 对照实际通过；规模矩阵由 OpenMM
独立参考覆盖。两者的覆盖范围分别记录，不把小体系 GROMACS 结果称为大体系验证。

## 1. 验证目标与顺序

建议两条验证线并行安排，先通过基础检查再扩大运行量：

1. **实现正确性/规模**：由 GROMACS 独立评价相同 Hamiltonian 的 E/F/W，逐步扩大
   bead 数、分子数、连接度、PBC 和 interaction 类型；GPUMD evaluator 不生成参考标签。
2. **真实 CG 训练/物理效果**：优先用作者发布的 CG 坐标—力数据做 force matching，
   再比较独立保留数据和构象分布；不把自行生成的合成标签当作物理数据。

自己构造测试几何可以有效检出索引、累加和边界错误，关键是答案由独立软件生成。
这种测试仍只证明实现正确；公开分子数据负责另一层验证。扩大轨迹数不等于扩大 bead 数，
单一分子的公开数据也不自动覆盖共享参数、变拓扑训练。

## 2. 数据来源与选择

2026-10-09 查阅作者仓库/论文/官方说明；只读取说明、代码和 notebook，不下载数据集。

| 候选 | 已确认的内容 | 用途与限制 | 优先级 |
| --- | --- | --- | --- |
| CGnet alanine dipeptide | 作者仓库直接提供 10,000 帧、5 bead 的坐标和力；300 K，ff99SB-ILDN，10 ps 采样间隔 | 小体积真实 CG 力匹配与二面角分布；只有固定 5 bead，不验证变大小；无现成 CG E/W 标签 | 明天首选 |
| TorchMD Chignolin tutorial | 作者教程列出 CA 坐标、原始力、delta-force、PSF、prior 与模型/轨迹资源 | 更复杂链构象；下载地址的实际可用性/包大小未确认；prior 含排斥，不完全等同本项目 bonded | 第二步候选 |
| TorchMD protein thermodynamics | 作者仓库有数据/模型下载说明，论文软件有 Zenodo 固定版本 | 后续检查多种蛋白长度和共享类型；先核对实际可获得的数据字段与体积 | 扩展候选 |
| Martini 官方教程 | 提供映射/拓扑/参数和参考构象工作流 | 可借用受支持 bonded 部分做 GROMACS 单点；许多完整模型含约束、improper、virtual site/排除 | 辅助，不直接移植完整力场 |

### 首选：CGnet 数据固定版本

仓库：<https://github.com/coarse-graining/cgnet>

调查时 master commit：`a3e0e8ddc06f4b6a9f48f4886b73b4cf372ff481`。
采用此 commit 的固定下载路径，防止明天 master 更新：

- [数据说明](https://github.com/coarse-graining/cgnet/blob/a3e0e8ddc06f4b6a9f48f4886b73b4cf372ff481/examples/data/README.md)
- [坐标](https://raw.githubusercontent.com/coarse-graining/cgnet/a3e0e8ddc06f4b6a9f48f4886b73b4cf372ff481/examples/data/ala2_coordinates.npy)
- [力](https://raw.githubusercontent.com/coarse-graining/cgnet/a3e0e8ddc06f4b6a9f48f4886b73b4cf372ff481/examples/data/ala2_forces.npy)
- [作者教程](https://github.com/coarse-graining/cgnet/blob/a3e0e8ddc06f4b6a9f48f4886b73b4cf372ff481/examples/Training-A-Coarse-Grained-Force-Field.ipynb)
- [原论文](https://doi.org/10.1021/acscentsci.8b00913)

GitHub 文件元数据中两数组各 600,128 bytes，合计约 1.2 MB。
数据说明给出 shape `(10000,5,3)` 和 bead 顺序：ACE C、ALA N、ALA CA、ALA C、NME N。
这些是保留 backbone atom 的 CG 映射，不是五个残基 COM，也不是五种任意独立粒子。

作者实现中的 beta 与教程使用 kcal/mol 能量尺度；**原始数组 README 没有完整的单位
声明**。明天要将教程/几何单位约定与原始数值共同核对，再锁定源单位和转换。
若仍无法核实 force 单位或投影定义，记录阻塞，不根据常见单位猜测训练。
不要因代码许可证存在就推断所有外部数据/模型许可相同，保存原始来源和引用要求。

CGnet 代码已停止常规维护；我们读取数据与方法，不需要把旧版 PyTorch 环境整体装进
GPUMD 开发环境。其 1% 教程子集也不代表论文完整结果能被复现。

### 第二步：Chignolin / 多蛋白

- [TorchMD-CG 作者仓库](https://github.com/torchmd/torchmd-cg)
- [固定版本教程](https://github.com/torchmd/torchmd-cg/blob/c6cc71c779b26c404a559885809e3c42934b7948/tutorial/Chignolin_Coarse-Grained_Tutorial.ipynb)
- 教程原始下载入口：`pub.htmd.org/torchMD_tutorial_data.tar.gz`。
  本次网页工具未成功读取压缩包，不保证明天可下载；不以此作为第一天必须通过的依赖。
- [protein thermodynamics 仓库](https://github.com/torchmd/torchmd-protein-thermodynamics)
- [版本归档](https://zenodo.org/records/8155343)：网页列出的约 552 kB 是软件版本包，
  **不是完整训练轨迹的大小**；实际数据需继续沿作者下载说明核实。
- [Martini 官方参数化教程](https://cgmartini.nl/docs/tutorials/Martini3/Small_Molecule_Parametrization/index.html)

TorchMD 教程既有原始 `chignolin_ca_forces.npy`，也有已经减 prior 的
`chignolin_ca_deltaforces.npy`。本项目入口加 baseline 后匹配 total，优先使用原始力；
不能将 delta-force 再作为 total 标签。排斥 prior 也不能无声删掉后宣称完整复现作者模型。
混合不同蛋白前必须统一映射、温度、源 Hamiltonian、类型含义、prior 和单位。

## 3. 数据进入 GPUMD 前的检查门槛

下载后的原文件只读保留，记录 URL、commit/DOI、文件长度和 SHA-256。派生数据、日志与
大轨迹不提交 Git；Git 只保留转换脚本、来源清单、小摘要/必要图。建议数据放在仓库外的
`/home/flos/Documents/Project/GPUMDs/cg_validation_data/`，本次尚未创建或下载。

先审计：shape、帧对应关系、finite、bead 顺序、映射/force 投影、温度、约束信息、单位、
重复帧和几何范围。检查净力/力矩作为诊断，不强行要求溶剂消去后的瞬时投影力逐帧为零。
如需约束体系的 force mapping，不能任意取原子力或猜测 COM 合力；参考
[force aggregation 方法论文](https://arxiv.org/abs/2302.07071)。

**标签关键点：** 投影瞬时力用于统计 force matching，并不是每个构型的精确 PMF 梯度；
存在不可消去的噪声，不能套用合成标签的近零 RMSE 阈值。
AA 势能不等于 CG PMF，也不能把 AA virial 直接当作当前 CG 的压力监督。
无可靠 E/W 标签时只训练力，不伪造物理能量/virial。

当前 `structure.cu` 要求 energy 字段存在，而 `lambda_e` 允许零。因此明天先验收：

```text
lambda_e 0
lambda_f 1
lambda_v 0
```

XYZ 的 `energy=0` 仅是读取器占位，派生数据清单必须标记 `energy_label=placeholder`，
不提供 virial 标签、不统计 E/W 标签误差。先通过小测试证明改变占位常数不改变 force
目标/结果，再启动正式训练；若有隐藏依赖，先修正输入/loss 支持再继续。
不向用户报告占位 energy RMSE 为模型精度。预测模型自身 E/F/W 的训练/MD 一致性仍可测。

无原始周期盒的孤立肽数据，嵌入足够大的三维周期盒以满足当前输入限制。统一居中，
确保任意相关距离的最小像不变且周期镜像不进入 cutoff；单点加倍盒边长核对结果。
这个人为大盒只用于孤立体系适配，不用于液体压力/NPT 或浓度研究。

数据划分在任何 scaler/prior 拟合前完成。采用连续时间块/轨迹块并留隔离区，结合
自相关检查隔离长度；不随机打散相邻帧到 train/test。优先 60/15/15% 加约 10% 隔离区，
最终比例随相关性和构象覆盖冻结。测试集保留独立目录/索引；其结果不用来选参数。
不同拓扑的后续评估另设未见拓扑/长度 holdout，不以帧级随机拆分替代。

## 4. 独立实现验证：先单点，再轨迹，再规模

[官方 GROMACS bonded 定义](https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html)
支持本项目三类势函数。先确认本机 GROMACS 可执行文件与精度；本次未在 PATH/常规
检查位置找到 gmx，尚未定位已有安装，不代表用户必须重新安装。

1. 恢复已有三层参考：bond-only → bond+angle → full bonded。
   目录 `tests/validation/gromacs_*` 已存在，但阶段 E 没有注册/执行这些外部测试。
   wrapper 仍使用旧 `dump_position` 和固定 8 bead 比较器；明天先检查其与当前
   `dump_xyz`、thermo header 的兼容性，不能把历史用例当作当前已通过。
2. 冻结相同坐标与盒，GROMACS 使用 CPU bonded、足够精度、不混入 Coulomb/LJ/约束。
   核对 0/1-based 编号、angstrom/nm、eV/kJ mol^-1、rad/degree、k/2 与 dihedral 符号。
   在进入规模测试前，手算一个 dimer 和带非零 phase 的 quartet，锁定符号/单位转换。
3. 导出 GROMACS 单点 E/F；W 首先用其能量对应变的数值导数独立检查，再对照直接
   输出的 virial。先确定张量顺序、符号和因子，不将压力或含动能项的量直接当作 W。
4. 使用公共数据的真实几何，加独立定义的受支持 bonded 参数，在 GROMACS 评价参考；
   再覆盖多项 dihedral、链/支化、共享类型、PBC 跨盒和三斜盒。
   参考参数用于软件验收，不宣称它就是原论文真实 CG Hamiltonian。

规模阶梯计划：约 32 → 256 → 2,048 → 8,192 bead，根据显存与时间决定是否再扩大。
每阶梯采样少量独立构型，比较逐 bead F、E/N、W/N 和总量；避免只看总量抵消。
扩大 bead 数、分子数、连接度分别记录，覆盖同一原子多 interaction 的原子累加竞争。
重复 3 次比较结果；记录 ms/step、峰值显存、interaction 数、GPU 错误和 NaN。

先在小体系校准并冻结误差门槛，参考 GROMACS double 输出时可将
`1e-8 absolute + 1e-7 relative` 作为初始调查阈值（E/W 为 eV、F 为 eV/angstrom），
须按实际参考精度、文件截断与 float 坐标差异说明。不能看到失败后无理由放宽阈值。
轨迹不要求长期逐帧一致；相同初值先比较最初几步，再做 dt 减半的 NVE 漂移检查。
遇到单点差异立即停止扩展，按 interaction 家族定位。

## 5. 公开数据训练与动力学验收

首轮用 alanine 的少量 train/validation 帧检查读取、force-only、unit、盒大小及 baseline
接入；随后扩大到冻结划分后的数据量。bond/angle prior 只用训练块估计或用可靠外部
参数，freeze 后共享；dihedral 若暂无可靠 prior 允许空表，不能为了凑三类功能任意拟合
并声称真实势。非零 dihedral 的独立正确性由 GROMACS 线覆盖。

比较 fixed prior-only、NEP-only、prior+NEP；网络预算、划分和选择规则保持可比。
先用一个 seed 检查可行性，再对最终配置用 3 个 seed，报告 test force RMSE/MAE、
逐 bead 误差和时间块 bootstrap 不确定度。force-only 模型能量零点不确定。
不要求 noisy-force 误差接近零；首先判断能否稳定改善 prior-only 与构象分布。

生产 MD 前，用保留构型核对同一导出包在 prediction/MD 的 total E/F/W；扫描实际
baseline 与 residual 的量级、float packing 误差，不只复用阶段 D 的人为 1e7 放大。
若该误差已超过目标 residual 的约 1% 或目标训练精度，先作为精度问题处理。

动力学从多种保留构象启动，先短 NVE 与减半步长，再做目标温度 NVT 的多个独立重复。
优先评估 bond/angle 分布、周期性的 phi/psi 二维分布、盆地占据及自由能差，采用
JS divergence/盆地概率差和 block bootstrap；不使用低采样 bin 的无界 KL 作为唯一指标。
学习到的 equilibrium CG PMF 不自动保证真实时间尺度；不以 protein folding rate 作为
第一天必须通过的目标。检查镜像/手性行为，必要时区分 NEP 表示能力与 bonded 实现问题。

NVT 总体统计以参考轨迹分块之间的差异作为有限样本基线，不宣称一条短轨迹能复现论文。
只有独立重复和采样覆盖足够后才判断物理效果；NPT/压力验收另需可解释的 CG W 标签。

## 6. 明天的执行清单与停止条件

| 次序 | 明天执行 | 输出/判断 |
| --- | --- | --- |
| 1 | 核对 Git HEAD、工具/GPU与 GROMACS；下载 alanine 两数组及说明并锁哈希 | provenance 清单；无来源/单位不继续 |
| 2 | 修复旧外部测试 wrapper 必要的输入兼容；三层 GROMACS 小体系单点 | 独立 E/F/W 对照；失败先定位 |
| 3 | 数据审计、时间块划分、force-only 占位检查及真空盒检查 | 可追溯数据转换与最小 smoke |
| 4 | 单 GPU 规模阶梯，先 256/2,048，资源允许再 8,192 | 正确性/显存/耗时报告，不同时启动重训练 |
| 5 | alanine pilot force matching；通过后扩大数据与 seed | 独立 holdout；没有真实 E/W 指标 |
| 6 | 导出包 prediction/MD；短 NVE/NVT 多初态 | 先稳定性，再分布；充分采样另安排 |

第一天交付目标是：独立 bonded 核对、较大规模正确性、公开数据可训练的 pilot 和明确的
后续物理验证入口。不能承诺一天内完成真实模型全部物理验收。

Chignolin 下载、复杂多蛋白共享模型与长时间分布是后续扩展；它们不可用时不阻塞前四项。
不启动全量蛋白数据下载或长任务来掩盖小参考测试失败。

本计划新增内容今天均未运行；不安排自动明日执行/定时任务。明天从第 1 项开始。
