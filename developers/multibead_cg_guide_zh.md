# 多 bead CG：训练、模型交付与 MD 使用指南

更新于 2026-10-09，适用于 `codex/multibead-cg-stage-b` 分支。
阶段 A–E 已完成，当前为已进行小规模软件验收的开发实现。

目标是训练和运行 `U_total = U_bonded + U_NEP_residual`：训练帧可以具有不同 bead 数、
分子数和连接关系，但共享 bead 类型与 bonded 系数。每次 MD 的拓扑固定。
固定系数不参与优化；普通 NEP 描述符不根据连接关系变化。

## 1. 构建与检查

从仓库根目录执行，使用安装好的 CMake（至少 3.24）、CUDA 和兼容的 host 编译器：

```bash
cmake -S . -B build-cg -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_TESTING=ON
cmake --build build-cg --parallel 2 --target gpumd nep check
```

`native` 对应本机 GPU；集群可显式设置架构，例如 `70;80`。
本地验收为 RTX 3060、CUDA 13.4.92、GCC 16.2.1、sm_86，19/19 CTest 通过。
该编译器/CUDA 组合在本次本地构建使用了 `--allow-unsupported-compiler`；移植时优先
选择 CUDA 支持的 host 编译器，不应将此开关当作其他环境已经兼容的证据。
源码仍保留 Makefile，但本阶段没有对完整 make 构建单独验收。

## 2. 准备共享参数和逐帧训练数据

共享 `bonded.in` 文件包含三类表，即使某一类没有参数也要给出零计数：

```text
gpumd_bonded_parameters 1
harmonic_bond_parameters 2
1.2 5.0
1.4 3.0
harmonic_angle_parameters 1
1.9 2.0
periodic_dihedral_parameters 2
0.2 3 0.4
0.07 1 -0.3
```

| 行类型 | 列含义 | 势能 |
| --- | --- | --- |
| bond | r0 / angstrom，k / eV/angstrom² | k(r-r0)²/2 |
| angle | theta0 / rad，k / eV/rad² | k(theta-theta0)²/2 |
| proper dihedral | k / eV，正整数 n，phase / rad | k[1+cos(n phi-phase)] |

系数须有限、强度非负；零强度关闭该项。bead 类型与 interaction 类型是独立编号空间。
各类表的行号就是从零开始的 interaction type。来自采用 `K(r-r0)²` 势型的外部数据时，
需转换为这里的 `k=2K`；degree 和 1-based 编号也必须明确转换，不能直接混用。

`nep.in` 中保留正常 NEP 配置，并添加：

```text
type 2 C O
molecular_force bonded.in per_frame
```

这里 C/O 是软件例子中的 bead 标签。`type` 仅写一次，顺序必须贯穿训练和部署。
其他 cutoff、网络、batch、generation 等参数按目标数据配置。
普通 potential NEP 的 resident/streaming、generic/specialized 路径已接入；
GNEP、charge/vdW/temperature/dipole/polarizability 模型及 atomic tensor 标签不支持。

每帧 `train.xyz` 和存在的 `test.xyz` 都要在原有 extended XYZ header 中加入：

```text
cg_topology_version=1 cg_bonds="0,1,0;1,2,1;2,3,0" cg_angles="0,1,2,0;1,2,3,0" cg_dihedrals="0,1,2,3,0;0,1,2,3,1"
```

这段对应 4 bead 链。每个元组最后一列是相应 interaction type；atom 编号从零开始、
只对当前帧有效。列表必须双引号包围，元组内无空白；空表写 `"none"`，不能省略字段。
同一四元组可以包含多个 dihedral 项。`mol_id` 不参与推断连接关系。

标签提供 **total E/F/W**，不要预先扣掉 bonded。XYZ energy/virial 为帧总量，force
为逐 bead 总力。已有 NEP prediction 文本中的 energy/virial 为每 bead 值，比较 MD
帧总量时要乘以该帧 N；force 不乘除 N。这里 W 表示 virial，遵循原 GPUMD 符号约定。
输入使用三维周期盒以及 wrap 到原始盒内的坐标；连接向量采用 consecutive MIC，
长键或半盒附近的图像歧义不会由拓扑自动修复。

运行训练和预测：

```bash
/path/to/nep
```

预测时在同一组配置、参数和数据下设置 `prediction 1`，读取当前 `nep.txt`。
它重新生成 train 以及存在的 test 的输出。输入格式细节见
[NEP molecular_force 文档](../doc/nep/input_parameters/molecular_force.rst)。

## 3. 完整模型交付

启用上述训练功能后，当前模型由三个文件组成：

```text
nep.txt                 # residual NEP
cg_model.json           # 类型、单位、势型、MIC、版本、文件名及 SHA-256
cg_model.bonded.in      # 实际载入训练的系数快照
```

Checkpoint 也导出同类 companion，以 checkpoint 的完整 NEP 文件名为前缀。
将对应三个文件一起复制，保持文件名和清单内容一致。清单成员是同目录文件名，
不依赖原始 `bonded.in` 路径。任何字节变更，包括参数文件注释，都改变校验和；
部署更新后的模型应使用对应的重新导出包。

质量和具体连接关系属于 MD 体系，另行提供。裸 residual 与已知 companion 共存时，
`potential` 会拒绝加载；**删除 companion 或单独重命名 residual 后无法自动辨认**。
这项保护不能替代正确交付完整模型。

## 4. 用同一模型运行不同体系

每个运行目录需要 `model.xyz`、`topology.in` 和 `run.in`。
`model.xyz` 提供盒、bead species、质量和所需初速度；可只包含共享类型的子集。
例如独立拓扑：

```text
gpumd_topology 1
number_of_atoms 4
bonds 3
0 1 0
1 2 1
2 3 0
angles 2
0 1 2 0
1 2 3 0
dihedrals 2
0 1 2 3 0
0 1 2 3 1
```

`run.in`：

```text
cg_model ../package/cg_model.json topology.in
time_step 0.05
ensemble nve
dump_thermo 1
run 1000
```

此步长仅用于例子，不是目标 CG 模型的稳定步长建议。不要与 `potential` 或
`molecular_force` 混合声明。任何 replicate 必须在 cg_model 之前，随后提供复制后
完整体系的拓扑。restart XYZ 重新作为 model.xyz 使用时，仍需声明同一包和对应拓扑。

底层 `molecular_force parameters.in topology.in` 和旧 combined v1/v2 输入仍支持。
推荐完整 CG 模型使用清单入口；格式说明见
[MD cg_model](../doc/gpumd/input_parameters/cg_model.rst) 和
[MD molecular_force](../doc/gpumd/input_parameters/molecular_force.rst)。

仓库提供两个可运行的合成体系，见 [example/README.md](cg_stage_e/example/README.md)。
它们共享一套模型；4 bead 链含 dihedral，6 bead 体系含两个三 bead 片段。
它们不是实际材料的 CG 力场。

## 5. 当前证据与后续使用限制

| 项目 | 当前证据 |
| --- | --- |
| 核心与读取/打包 | CPU/GPU E/F/W、有限差分、异常输入、PBC、batch/stream 检查 |
| 训练闭环 | 两帧合成 total 标签；零 residual 与已知非零 residual；generic/specialized |
| 模型部署 | 两种拓扑训练/MD 对齐；split/legacy/package 对照；缺失 bead 类型、restart、replicate |
| 短 NVE | 两个体系各 50 fs；时间步减半后能量偏差约降至四分之一 |
| GPU 内存 | 两个完整包示例各 1000 步 memcheck，0 errors |
| 普通 NEP | 不启用 bonded 的训练与 fresh train/test prediction smoke |

训练坐标/盒与 total 预测/loss 仍含 float 路径。强 bonded 加小 residual 可能丢失小信号；
阶段 D 的放大 1e7 用例最大力误差为 1.13261e-3 eV/angstrom，此问题尚未解决。
同坐标/类型但不同拓扑，普通 NEP residual 无法仅靠连接表区分。

当前只验收一张 GPU、两个合成帧；train/test 同帧不是泛化验证。
真实 CG 数据、独立 holdout、长期动力学、NVT/NPT 分布、大体系/多 GPU 性能及新增
GROMACS 对照尚未验收。未实现 improper、约束、刚体、virtual site、拓扑非键排除
和 1–4 scaling；此实现不等同完整原子力场。

详细结果与复现脚本见 [阶段 D](cg_stage_d/STATUS.md)、[阶段 E](cg_stage_e/STATUS.md)。
开发续接见 [交接文档](HANDOFF_20261009_CG_STAGE_E.md)。阶段 A/B/C 文档是当时状态
快照；其中“尚未开放”的表述不能用于判断当前接口。
