# 阶段 A：多 bead CG 输入草案与测试基线

目标：为后续合并固定接口与验收用例。本目录不代表逐帧拓扑训练功能已经实现。

## 两帧最小用例

共享参数见 `input_draft/bonded_parameters.in.draft`。bead 类型为 C、O，分别作为两个测试类型的占位符；质量 12、16 是测试值，没有化学含义。

| 帧 | bead 数 | 分子组成 | bond | angle | dihedral 项 |
| --- | --- | --- | --- | --- | --- |
| frame4 | 4 | 一条四 bead 链 | 3 | 2 | 同一四元组的两项 |
| frame6 | 6 | 两条三 bead 链 | 4 | 2 | 0，显式 `none` |

两帧采用同一参数表、20 Å 三维周期正交盒。构型均远离 MIC 边界和退化几何。标签是解析 bonded 总能量、总力、总 virial，NEP residual 为零；这是软件验收用例，不是物理训练集，也不能用于评估泛化。`mol_id` 是解释分子分组的辅助属性，连接关系以显式 interaction 表为准。

`make_fixture.cpp` 使用当前 host angle/dihedral evaluator 生成标签，以坐标和全 9 分量应变中心有限差分检查 E/F/W，并检查净力、净力矩；bond 为独立简谐公式。它通过当前 v2 读取器回读拓扑与参数。这个检查验证数学和旧格式，不验证新草案读取器或 GPU。

## 建议冻结的接口

### 共享参数文件

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

顺序与 v2 相同，去掉体系原子数与连接表。类型编号等于参数行的 0-based 位置。单位为 eV、Å、rad；bond/angle 的能量包含 `1/2`。

### 训练输入：拟议接口，当前不可用

拟在 `nep.in` 增加：

```text
type 2 C O
molecular_force bonded_parameters.in per_frame
```

每帧 extended XYZ 头包含：

```text
cg_topology_version=1
cg_bonds="atom_i,atom_j,type;..."
cg_angles="atom_i,atom_j,atom_k,type;..."
cg_dihedrals="atom_i,atom_j,atom_k,atom_l,type;..."
```

索引是帧内 0-based，type 放在最后，与当前内部拓扑/旧 v2 一致；版本字段及三类列表均必需。空列表使用 `"none"`，空字符串不作为正常语法。每个属性只出现一次；值内不允许空格。不按 species 或 `mol_id` 猜连接关系。

此约定与对方的 1-based、type-first 列表不同，必须按源格式显式转换，不能复用未转换的对方 XYZ。旧 NEP 可能忽略未知 XYZ 属性，因此不能只给旧训练器提供这些头字段就认为 bonded 已生效；新增关键字和启动日志需要确认功能实际启用。

`train.xyz.draft` 中 `energy` 为全帧总能量（eV），`force` 为各 bead 力（eV/Å），`virial` 为全帧总量（eV），以行优先 `xx xy xz yx yy yz zx zy zz` 输出，不预除以 bead 数或体积。当前 NEP 读取代码在读入后执行自身的 per-bead 归一化。GPU baseline 的 6 分量布局需单独转换验证。

### MD 输入：拟议接口，当前不可用

拟扩展已有关键字：

```text
potential nep.txt
molecular_force bonded_parameters.in frame4.topology.in
```

独立拓扑文件：

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

保留已有 `molecular_force <v1/v2完整文件>`；新双文件形式只负责参数/拓扑分离。先 replicate，再加载与复制后索引匹配的拓扑；第一版不自动复制拓扑。

`frame*.molecular_force.in` 与 `frame*.model.xyz` 已使用当前支持的格式。它们是未来 GPU/MD 对照的入口，仍需配套普通 potential；纯 bonded 验证使用 epsilon=0 的 potential。

### 模型交付

保持 `nep.txt` 原格式，第一版建议增加 `cg_model.json` 伴随清单，记录 schema version、residual NEP 文件、共享参数文件与 SHA256、bead 类型映射、势函数/单位约定。每次 MD 单独指定连接表和 bead 质量，模型清单不固定体系原子数。清单读取与缺失检查在后续实现，当前没有生成已可加载的模型包。

## 校验规则

- 参数/拓扑头版本明确；未知版本报错。
- 检查计数、非有限数、非法参数、索引越界、interaction 内重复原子、重复/缺失 XYZ 属性。
- 合法叠加项保留；不对四元组简单去重。
- 不同帧可使用不同参数类型的子集，但不能改变共享参数含义。
- 第一批端到端测试采用三维周期盒；其他边界必须明确支持或拒绝。
- 同一有序四元组重复同一个 type 是否确为有意，应提供诊断；不自动合并项。

## 重现 host 基线

```bash
python3 developers/cg_stage_a/run_host_baseline.py
```

脚本在临时目录编译，assert 保持启用；执行本仓库 4 项现有 host 测试、对方 1 项 CPU 测试和 fixture 检查；生成 `baseline.json` 和最小输入文件。编译参数、执行结果、源文件 SHA256 记录在报告中。报告内的临时 executable 路径用于审计；重跑请执行脚本，而不是复制已删除的路径。

GPU、MD 和 GROMACS 结果另行记录，不把 host PASS 或跳过当作 GPU PASS。

## 重现 GPU 基线

有 CUDA 与 CMake 工具链时执行：

```bash
bash developers/cg_stage_a/run_gpu_baseline.sh
```

本次使用 RTX 3060（架构 86），构建目录为 `/tmp/gpumd-stage-a-build`。CUDA 通过绝对路径使用；系统尚无 CMake/Ninja，因此工具临时安装于 `/tmp/gpumd-stage-a-tools` 的独立 Python 环境，没有修改系统安装。

```bash
export PATH="/tmp/gpumd-stage-a-tools/bin:$PATH"
CG_CUDA_ARCH=86 bash developers/cg_stage_a/run_gpu_baseline.sh /tmp/gpumd-stage-a-build
```

脚本运行现有 CTest 测试并输出 `gpu_tests.xml`，随后使用当前可加载的 v2 完整输入运行两个 fixture，输出 `fixture_md_baseline.json`。fixture 在零速度、`1e-8 fs` 步长下运行一步，只比较近静态 E/F/W，不表示已通过 NVE 稳定性、NEP 训练或新接口验证。

GROMACS 外部验证只有在配置时找到可执行文件才会注册。本次未发现 GROMACS，应明确记录为未运行，不能以 CTest 总通过数隐含已经完成跨软件验证。
