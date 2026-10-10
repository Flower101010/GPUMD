# 多 bead CG 入门：从训练到 MD

这个例子演示同一套 bonded 参数和 NEP 如何用于不同大小、不同拓扑的体系：

- `md4`：一条 4 bead 链，包含 bond、angle 和两项 proper dihedral。
- `md6`：两个 3 bead 分子，包含 bond 和 angle，dihedral 列表为空。
- 两种体系共享 C/O 类型和 `train/bonded.in`。

完整势能为 `U_total = U_bonded + U_NEP`。bonded 参数固定，训练只更新 NEP。
这里的标签由独立 OpenMM 计算，刻意只包含 bonded，所以目标 residual 为零。
它是学习文件格式与操作流程的小例子，不是实际材料的 CG 模型。
C/O 是占位的 bead 标签；示例质量分别为 12/16 amu，不代表你的映射质量。

## 1. 编译

下面的命令和脚本使用 Linux/Bash。
需要 NVIDIA GPU、驱动、CUDA Toolkit、CMake >= 3.24 和 CUDA 支持的 C++ 编译器。
Python、uv、OpenMM、GROMACS **不是运行本例的依赖**，训练数据已经包含在仓库中。

在仓库根目录执行：

```bash
cmake -S . -B build-cg -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_TESTING=ON
cmake --build build-cg --parallel 2 --target gpumd nep
cmake --build build-cg --parallel 2 --target check
```

编译后得到 `build-cg/nep`（训练/预测）和 `build-cg/gpumd`（MD）。
`check` 运行测试；GROMACS 外部测试只在找到对应程序时注册。
`native` 用编译机器的 GPU 架构；交叉编译到其他机器时显式指定架构，
例如 RTX 3060 为 `-DCMAKE_CUDA_ARCHITECTURES=86`。

若找不到 CUDA，可先设置 `export CUDACXX=/usr/local/cuda/bin/nvcc`。
若 nvcc 报 host compiler 不支持，优先安装兼容编译器，并在首次配置时用
`-DCMAKE_CUDA_HOST_COMPILER=/path/to/g++` 指定。更换工具链时使用新的 build 目录。
本机曾用 `-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler` 编译，
这只是绕过版本检查，不是通用安装建议。

## 2. 一条命令跑通

仍在仓库根目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/multibead_cg/run_example.sh
```

脚本依次执行 **训练 → 预测 → 复制完整模型 → 运行 md4/md6**。
40 代训练仅用于演示流程，不以拟合误差作为成功条件。
脚本不改变仓库中的输入，输出放在 `examples/multibead_cg/work/`，此目录已被 Git 忽略。
再次运行需指定一个尚不存在的输出目录，避免旧模型、restart 或追加轨迹混入结果：

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/multibead_cg/run_example.sh \
  /absolute/path/to/build-cg /tmp/my-cg-example-2
```

第一参数是含 `gpumd` 和 `nep` 的构建目录；第二参数是新输出目录。
脚本保留日志。若中途失败，查看相应目录的 `train.log`、`predict.log` 或 `gpumd.log`。
集群已分配 GPU 时，沿用调度器设置的 `CUDA_VISIBLE_DEVICES`，不要照抄覆盖为 0。

2026-10-10 已实际运行本流程：40 代训练、全部 12/4 帧预测、完整模型哈希检查，
两套体系各 1,000 步 MD；预测和热力学输出均为有限值，轨迹及 restart 完整。

## 3. 文件放在哪里

```text
multibead_cg/
├── run_example.sh
├── train/
│   ├── nep.in              训练设置
│   ├── nep.predict.in      预测设置模板
│   ├── bonded.in           所有训练帧共享的固定参数
│   ├── train.xyz           12 个训练帧，包含各自拓扑和总量标签
│   └── test.xyz            4 个测试帧
├── md4/                    4 bead 体系
│   ├── model.xyz           坐标、盒子、类型、质量、速度
│   ├── topology.in         这个体系的连接关系
│   └── run.in              MD 设置
└── md6/                    同样的三个文件，换成 6 bead 体系
```

运行后的 `work/` 还有 `package/`，其中的三个文件构成可部署模型：

| 文件 | 内容 |
| --- | --- |
| `nep.txt` | 学到的 residual NEP |
| `cg_model.bonded.in` | 训练时实际使用的 bonded 参数快照 |
| `cg_model.json` | 类型顺序、单位、约定、成员文件名和 SHA-256 |

部署时三个文件一起复制，不要修改文件内容，也不要只加载 residual `nep.txt`。
清单由程序自动生成，不需要手写。体系的质量和拓扑另由 `model.xyz/topology.in` 提供。

## 4. 怎么训练和预测

程序从**当前工作目录**读取输入；路径也相对于该目录。
若想逐步操作，先在仓库根目录执行以下命令。`/tmp/cg-manual` 必须尚不存在，
可以换成自己的长期工作目录；同一终端保留 `CG_BUILD/CG_EXAMPLE` 两个变量：

```bash
CG_BUILD=$(realpath build-cg)
CG_EXAMPLE=$(realpath examples/multibead_cg)
mkdir /tmp/cg-manual
cp -R "$CG_EXAMPLE/train" /tmp/cg-manual/train
cd /tmp/cg-manual/train
"$CG_BUILD/nep" > train.log 2>&1
```

本例 `nep.in` 中的主要设置：

```text
type 2 C O
cutoff 4 3
n_max 1 1
basis_size 1 1
l_max 2 0 0
neuron 4
molecular_force bonded.in per_frame
lambda_e 1
lambda_f 1
lambda_v 0.1
batch 4
stream_train 1
generation 40
population 10
output_interval 10
nep_compile off
```

`type` 给共享 bead 标签及顺序；`cutoff` 为径向/角向截断半径，单位 angstrom。
接下来的四项定义示例的小 NEP 网络；`molecular_force` 开启固定 bonded 与逐帧拓扑。
`lambda_e/f/v` 分别为能量、力、virial 误差权重；`batch` 是每批构型数，
`stream_train 1` 按批加载以节省显存；`generation/population` 控制优化次数和种群。
`nep_compile off` 避免例子额外依赖运行时编译，正常训练也支持 `on`。

训练结束会写 `loss.out`、`nep.txt`、`nep.restart` 和完整 CG 包。
`nep.restart` 是训练继续所用的优化状态，不是 MD 的 restart 文件。

预测时保持数据、bonded 参数、网络尺寸和 `type` 顺序不变，使用预测配置：

```bash
cp nep.in nep.training.in
cp nep.predict.in nep.in
"$CG_BUILD/nep" > predict.log 2>&1
cp nep.training.in nep.in
```

`prediction 1` 读取 `nep.txt`，重新评价 train 和存在的 test，不进行优化。
输出列的含义：

| 输出 | 每行含义 | 列顺序 |
| --- | --- | --- |
| `energy_train.out / energy_test.out` | 一个构型 | 预测 E/N，参考 E/N |
| `force_train.out / force_test.out` | 一个 bead | 预测 Fx Fy Fz，参考 Fx Fy Fz |
| `virial_train.out / virial_test.out` | 一个构型 | 预测 6 分量，参考 6 分量，均为 W/N |

virial 六分量顺序为 `xx yy zz xy yz zx`。E/N、W/N 单位 eV/bead；力单位 eV/angstrom。
这里输出的是 **bonded + NEP 的总预测**。原始 XYZ 的 E/W 是帧总量，比较时注意 N。

## 5. 训练数据格式

`train.xyz/test.xyz` 是多个 extended XYZ 帧直接拼接，每帧由三部分构成：

1. 第一行：当前帧的 bead 数 N。
2. 第二行：盒子、列定义、标签和当前帧的拓扑，必须是一整行。
3. 后续 N 行：逐 bead 的类型、坐标和参考力。

最小列定义是 `Properties=species:S:1:pos:R:3:force:R:3`，
即每行 `类型 x y z Fx Fy Fz`。完整数值例子直接看 [train.xyz](train/train.xyz)。

| header 字段 | 含义 |
| --- | --- |
| `pbc="T T T"` | 三维周期盒 |
| `Lattice="ax ay az bx by bz cx cy cz"` | 三个盒向量，单位 angstrom；本例为边长 20 的正交盒 |
| `energy=...` | 帧总能量，单位 eV |
| `virial="..."` | 帧总 virial，9 数值按 `xx xy xz yx yy yz zx zy zz` 排列，单位 eV |
| `cg_topology_version=1` | 本功能的拓扑格式版本 |
| `cg_bonds="i,j,type;..."` | 每个 bond 的两端与参数编号 |
| `cg_angles="i,j,k,type;..."` | 每个 angle，j 是中心 |
| `cg_dihedrals="i,j,k,l,type;..."` | 有顺序的 proper dihedral |

这里的 `...` 仅用于说明，不能原样写进输入。实际 4 bead 链的拓扑字段为：

```text
cg_topology_version=1 cg_bonds="0,1,0;1,2,1;2,3,0" cg_angles="0,1,2,0;1,2,3,0" cg_dihedrals="0,1,2,3,0;0,1,2,3,1"
```

**原子编号和参数编号都从 0 开始。**原子编号仅对当前帧有效。
三类列表必须都写；没有某一项时写 `"none"`，如 6 bead 帧的 `cg_dihedrals="none"`。
列表内不能有空格，用分号分隔 interaction；同一四元组可以有多个 dihedral 项。
不同帧可有不同 N、分子数和连接关系，不能靠 `mol_id` 或 bead 距离自动推断连接。

标签填写 **总 E/F/virial，不能提前扣除 bonded**。
若没有 virial 标签，可省略 `virial` 并设 `lambda_v 0`。
仅用力训练可设 `lambda_e 0`、`lambda_v 0`；当前读取器仍需要 `energy` 字段，
此时可填占位值，但它不是物理能量，不能据此评价能量误差。
坐标应 wrap 到原始周期盒内；bonded 向量使用连续最小镜像，跨边界的正常短键可处理。

## 6. bonded 参数格式

文件完整内容见 [bonded.in](train/bonded.in)。去掉注释后为：

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

每节标题后的数字是参数条数；该节数据行从 0 编号，每一类单独编号。
比如 `bond type=1` 用 `r0=1.4, k=3.0`，`dihedral type=1` 用 `k=0.07, n=1, phase=-0.3`。

| 类型 | 列含义 | 势能形式 |
| --- | --- | --- |
| bond | r0（angstrom）、k（eV/angstrom²） | `k(r-r0)²/2` |
| angle | theta0（rad）、k（eV/rad²） | `k(theta-theta0)²/2` |
| dihedral | k（eV）、正整数 n、phase（rad） | `k[1+cos(n*phi-phase)]` |

三个参数节必须按顺序出现；某类没有参数时写零计数。允许空行和 `#` 注释。
参数强度非负；从其他软件搬参数时注意 `/2` 约定、角度单位和编号转换。

## 7. 怎么跑 MD

MD 目录有 `model.xyz`、`topology.in`、`run.in`，模型包放在相邻 `package/`。
在该 MD 目录运行 `/absolute/path/to/build-cg/gpumd`。

接着上面的手动训练/预测步骤，在同一终端执行：

```bash
mkdir /tmp/cg-manual/package
cp /tmp/cg-manual/train/{nep.txt,cg_model.json,cg_model.bonded.in} \
  /tmp/cg-manual/package/
cp -R "$CG_EXAMPLE/md4" "$CG_EXAMPLE/md6" /tmp/cg-manual/
cd /tmp/cg-manual/md4
"$CG_BUILD/gpumd" > gpumd.log 2>&1
cd /tmp/cg-manual/md6
"$CG_BUILD/gpumd" > gpumd.log 2>&1
```

`model.xyz` 的格式见 [md4/model.xyz](md4/model.xyz)。header 的
`Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3` 表示每行：

```text
类型 x y z 质量 vx vy vz
```

坐标/盒长为 angstrom，质量为 amu，速度为 angstrom/fs；本例显式给零速度，
不需要 `velocity` 指令。自己的 CG bead 质量应取映射后质量。

`topology.in` 与训练 XYZ 中的拓扑描述同一件事，只是单独存成文件：

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

每节数字是 interaction 数，行末为参数编号，原子顺序与 `model.xyz` 一致。
空节写 `dihedrals 0` 等，不能省略；原子数必须与 model.xyz 相符。

`run.in`：

```text
cg_model ../package/cg_model.json topology.in
time_step 0.05
ensemble nve
dump_thermo 20
dump_xyz 20 movie.xyz force potential virial precision double
dump_restart 1000
run 1000
```

这段执行 1,000 步、每步 0.05 fs，共 50 fs；每 20 步写热力学量和轨迹。
此步长只用于例子，应为自己的体系另行检查稳定性。
`cg_model` 一次加载 residual 和 bonded，**不要再写 `potential` 或 `molecular_force`**。
同一包可用于 md6，换该体系的坐标、质量和拓扑即可。

如果只想给已有普通势额外添加固定 bonded，也有较低层的入口：

```text
potential existing_potential.txt
molecular_force bonded.in topology.in
```

这里 `existing_potential.txt` 必须是原本不含这些 bonded 项的势，否则会重复计入。
训练导出的完整 CG 模型仍应使用上面的 `cg_model` 入口。
旧的单文件 `molecular_force combined.in` 也兼容，详见
[molecular_force 格式文档](../../doc/gpumd/input_parameters/molecular_force.rst)。

输出 `thermo.out`、`movie.xyz`（坐标、力、每 bead 势能/virial）及 `restart.xyz`。
MD 继续运行时，将 restart 作为新目录的 `model.xyz`，同时提供对应拓扑和完整模型声明。
若使用 `replicate`，须放在 `cg_model` 前，且给出复制后全体系拓扑。

## 8. 换成自己的 CG 数据

先替换共享 bead 类型和 bonded 参数，再准备每帧总量标签与拓扑。
当前 bead 标签沿用 GPUMD 的元素符号体系，不直接支持随意命名的 `B1/B2`；
给不同 bead 类型分配不同支持的符号，并在训练和部署中保持同一映射/顺序。
MD 质量显式写出，不根据占位符号猜测。

把示例的 40 代、小网络、cutoff 和 timestep 换成适合你的任务的设置。
若目录中有旧 `nep.restart`，训练可能继续旧状态；新任务使用新目录。
NEP 的普通描述符不读取连接表；bonded 系数也不参与优化。

目前支持普通 NEP4/NEP4-ZBL、三维周期体系，未支持 GNEP、improper、约束、
virtual site、拓扑非键排除和 1–4 scaling。
训练数组仍含 float，强 bonded 叠加很小 residual 时要检查精度。
软件验收只检查计算正确性，不以能否拟合出好 CG 模型作为通过条件。

更多约定见 [完整使用指南](../../developers/multibead_cg_guide_zh.md)，
独立计算验证见 [阶段 F 记录](../../developers/cg_stage_f/STATUS.md)。
