# GPUMD 分子 bonded force 快速开始

本文说明 `polymer-development` 分支中新增的固定拓扑分子力功能，包括如何编译、如何在
`run.in` 中启用，以及如何运行 bond、angle 和 proper dihedral 的验证。

## 当前支持范围

当前 molecular force 模块支持：

- harmonic bond（简谐键）；
- harmonic angle（简谐键角）；
- periodic proper dihedral（周期型 proper 二面角）；
- 基于 GPU 的能量、力和 per-atom virial 计算；
- 正交和 triclinic 周期边界下的 minimum-image 几何计算。

详细的输入字段定义见：

- [格式版本 1：harmonic bond](molecular_force_input_format_v1_zh.md)
- [格式版本 2：harmonic bond、harmonic angle、periodic proper dihedral](molecular_force_input_format_v2_zh.md)

当前不支持 improper dihedral、SHAKE/RATTLE、rigid water、virtual site、topology-aware
非键排除、固定电荷 Coulomb 和 1--4 scaling。因此该模块目前适合固定拓扑的 bonded/NEP
或简化粗粒化模型，不等同于完整的 GROMACS、CHARMM 或 AMBER 力场。

## 编译

需要 CUDA 工具链和可用的 NVIDIA GPU。建议使用分支自带的 CMake 配置，并使用一个新的构建
目录，避免复用其他分支生成的缓存：

```bash
cmake -S . -B build-polymer -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-polymer --target gpumd --parallel 2
```

编译完成后，可执行文件为：

```text
build-polymer/gpumd
```

也可以使用传统 Makefile：

```bash
make -C src -j2
```

此时可执行文件为 `src/gpumd`。如需指定 GPU 架构，请按 `src/makefile` 开头的说明调整
`CUDA_ARCH`。

## 在 run.in 中启用

`molecular_force` 是对已有 GPUMD potential 的叠加项，因此 `run.in` 仍然需要至少一个
`potential`。最小配置如下：

```text
potential zero_lj.txt
molecular_force molecular_force.in

time_step 0.05
ensemble nve
dump_thermo 10
dump_position 100 precision double
run 4000
```

如果使用 `replicate`，必须先复制体系，再读取与复制后原子编号匹配的拓扑文件：

```text
potential potential.txt
replicate 2 2 1
molecular_force replicated_molecular_force.in
```

`molecular_force` 文件中的 `number_of_atoms` 必须等于当前 `model.xyz` 的原子数，原子编号
从 `0` 开始。

## bond、angle 和 dihedral 输入示例

下面是格式版本 2 的最小四原子示例。所有角度和相位均使用弧度：

```text
gpumd_molecular_force 2
number_of_atoms 4

harmonic_bond_parameters 1
1.0 10.0

harmonic_angle_parameters 1
1.0471975511965976 2.0

periodic_dihedral_parameters 1
1.7 3 0.4

bonds 1
0 1 0

angles 1
0 1 2 0

dihedrals 1
0 1 2 3 0
```

其中：

- bond 参数行是 `r0 k_bond`，势能为 `1/2 * k_bond * (r-r0)^2`；
- angle 参数行是 `theta0 k_angle`，角度顺序为 `i-j-k`，其中 `j` 是中心原子；
- dihedral 参数行是 `k_dihedral multiplicity phase`，势能为
  `k_dihedral * [1 + cos(multiplicity * phi - phase)]`；
- 每条 interaction 的最后一个整数是对应参数表的 0-based type。

完整的参数约束、周期边界约定和退化几何处理见
[格式版本 2 文档](molecular_force_input_format_v2_zh.md)。

## 直接运行验证

### Harmonic bond NVE 示例

该示例使用三个 Ar 粒子和两条 harmonic bond：

```bash
./examples/gpumd_harmonic_trimer/run_example.sh ./build-polymer/gpumd
```

脚本会运行短时间 NVE，并检查总能量近似守恒以及两条键长确实发生振动。

### 同时验证 bond、angle、dihedral

配置测试并编译测试目标：

```bash
cmake -S . -B build-polymer -DBUILD_TESTING=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-polymer --target check --parallel 2
```

也可以只运行 `run.in` 集成测试：

```bash
bash tests/test_molecular_force_run_in.sh \
  ./build-polymer/gpumd ./build-polymer/test_gpu_vector
```

该测试会临时生成包含一条 bond、一条 angle 和一条 periodic proper dihedral 的四原子体系，
从 `run.in` 读取 `molecular_force.in`，并检查 GPUMD 输出的 bonded potential energy。

没有可用 GPU 时，GPU 测试可能会跳过；如果出现编译错误或输入错误，则不会被视为通过。

## 启动日志和常见问题

成功读取拓扑后，`gpumd.log` 中应出现类似信息：

```text
Initialized molecular force from molecular_force.in.
    number of harmonic bonds = ...
    number of harmonic angles = ...
    number of periodic dihedrals = ...
```

常见问题：

- 报原子数不一致：检查 `model.xyz`、`number_of_atoms` 和 `replicate` 后的拓扑是否对应；
- 报参数 type 越界：确认 bond、angle、dihedral 的 type 分别从各自参数表的 `0` 开始；
- angle 或 phase 数值错误：输入单位必须是弧度，不是 degree；
- `molecular_force` 放在 `replicate` 前后顺序错误：应先 `replicate`，再读取拓扑；
- 只写 `molecular_force` 而没有 `potential`：当前实现仍要求保留一个 GPUMD potential，
  若只想测试 bonded force，可使用 epsilon 为 0 的 `zero_lj.txt`。
