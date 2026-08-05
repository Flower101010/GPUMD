# GPUMD molecular force 第二版输入格式说明

> 文档状态：开发版
>
> 格式版本：2
>
> 当前支持：harmonic bond、harmonic angle、periodic proper dihedral

## 1. 版本目标与兼容性

版本 2 在版本 1 的 harmonic bond 基础上增加简谐键角和周期型 proper dihedral。版本 1
文件的语法和含义保持不变，读取器同时接受版本 1 和版本 2；新文件若需要 angle 或
dihedral，必须使用：

```text
gpumd_molecular_force 2
```

这些相互作用由显式原子索引表定义，每一步直接遍历固定拓扑，不构建也不修改 neighbor
list。它们的能量、力和 virial 会在普通 GPUMD potential 计算完成后叠加。

## 2. 完整示例

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

各节顺序固定，不能交换。记录数可以为 0，但对应节本身仍必须出现。

## 3. 通用规则

- 原子编号从 `0` 开始，必须位于 `0` 到 `number_of_atoms - 1`。
- 每类 interaction 的 `type` 分别索引本类参数表，不跨表共享编号。
- 空行和 `#` 后的注释会被忽略。
- 所有数值必须完整合法，不能附带单位字符串。
- 长度单位为 Å，能量单位为 eV，所有角度和相位单位为弧度。
- 同一组有序原子可以出现多条记录，用于叠加多个势能项；输入生成工具应检查非预期重复。
- `number_of_atoms` 必须与加载时的 `model.xyz` 一致；若使用 `replicate`，仍须先复制体系，
  再加载与复制后编号匹配的 molecular force 文件。

## 4. Harmonic bond

参数节和 interaction 节为：

```text
harmonic_bond_parameters M
r0 k_bond

bonds B
atom_i atom_j type
```

势能为：

```text
U(r) = 1/2 * k_bond * (r - r0)^2
```

`r0` 必须为有限正数，单位 Å；`k_bond` 必须为有限正数，单位 eV/Å²。详细的版本 1
说明仍见 [molecular_force_input_format_v1_zh.md](molecular_force_input_format_v1_zh.md)。

## 5. Harmonic angle

### 5.1 参数表

```text
harmonic_angle_parameters A
theta0 k_angle
```

参数行顺序就是 type 编号。势能定义为：

```text
U(theta) = 1/2 * k_angle * (theta - theta0)^2
```

| 参数 | 单位 | 约束 | 含义 |
| --- | --- | --- | --- |
| `theta0` | rad | 有限，`0 < theta0 <= π` | 平衡键角 |
| `k_angle` | eV/rad² | 有限正数 | 键角力常数 |

输入其他软件的角度参数时必须先把 degree 转成 rad，并核对对方公式是否包含 `1/2`。

### 5.2 Angle 表

```text
angles N
atom_i atom_j atom_k type
```

`atom_j` 是中心原子，角度定义为 `i-j-k`。三个原子必须互不相同。两条从中心原子出发的
向量分别使用 minimum-image convention，因此跨周期边界的 angle 不依赖 neighbor list。

每条 angle 的能量平均分给三个原子。总力严格按 `F_j = -(F_i + F_k)` 构造；完整
interaction virial 平均分给三个原子后累加到 GPUMD 的 9 分量 per-atom virial。

## 6. Periodic proper dihedral

### 6.1 参数表

```text
periodic_dihedral_parameters D
k_dihedral multiplicity phase
```

势能定义为：

```text
U(phi) = k_dihedral * [1 + cos(multiplicity * phi - phase)]
```

| 参数 | 单位 | 约束 | 含义 |
| --- | --- | --- | --- |
| `k_dihedral` | eV | 有限非负数 | 周期势振幅 |
| `multiplicity` | 无量纲整数 | 正整数 | 周期重数 |
| `phase` | rad | 有限实数 | 相位 |

相位允许超出 `[-π, π]`，计算时由三角函数自然周期化。不同软件可能使用
`cos(n phi - phase)`、`cos(n phi + phase)` 或不同的 phi 符号，导入前必须核对，不能只做单位转换。

### 6.2 Dihedral 表与符号约定

```text
dihedrals N
atom_i atom_j atom_k atom_l type
```

这是有序的 proper torsion `i-j-k-l`。对连续 minimum-image 键向量定义：

```text
b1 = r_j - r_i
b2 = r_k - r_j
b3 = r_l - r_k
n1 = b1 × b2
n2 = b2 × b3
phi = atan2(|b2| * dot(b1, n2), dot(n1, n2))
```

因此 `phi` 位于 `[-π, π]`。反转或重排原子顺序可能改变符号，不应把 interaction
记录当作无序四元组。

每条 dihedral 的能量平均分给四个原子。力由势能对四个笛卡尔坐标的解析梯度计算，
interaction virial 平均分给四个原子。

## 7. 周期边界和退化几何

angle 和 dihedral 都直接使用显式拓扑索引：

```text
Topology interaction table -> consecutive MIC vectors -> bonded CUDA kernel
```

不会访问或修改普通势函数的 neighbor list。

以下几何没有良好定义的解析方向：

- angle 任一臂长度为 0；
- angle 精确为 0 或 π，导致 `sin(theta) = 0`；
- dihedral 中心键长度为 0；
- dihedral 任一相邻三原子共线，导致平面法向量长度为 0。

当前 GPU kernel 对这些精确退化 interaction 不累加能量、力或 virial，以避免 NaN。它是数值保护，
不是合理的物理模型；输入构型和积分步长必须避免到达这些状态。

## 8. 当前边界

版本 2 仍不包含：

- improper dihedral；
- constraint、SHAKE/RATTLE、rigid water、virtual site；
- topology-aware nonbonded exclusions 和 special 1-4 scaling；
- 通用固定电荷 Coulomb；
- molecule template、residue 和外部力场文件直接导入；
- 对 topology 做自动 replicate。

普通 GPUMD potential 仍不知道哪些原子通过 bond/angle/dihedral 相连。因此版本 2 适合验证
不需要拓扑非键排除的 bonded/NEP 或简化粗粒化模型，不能直接等同于完整的 OPLS、CHARMM
或 AMBER 全原子力场支持。

## 9. 推荐验证

每种新参数至少应检查：

1. 已知几何的解析能量；
2. 解析力与能量中心有限差分一致；
3. 四舍五入误差范围内总力为 0；
4. 整体平移后能量不变；
5. 跨正交或 triclinic 周期边界结果一致；
6. virial 与有限应变导数一致；
7. 短时间 NVE 能量守恒；
8. 与目标力场软件核对 dihedral 符号、相位和参数单位。
