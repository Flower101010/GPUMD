# GPUMD molecular force 第一版输入格式说明

> 文档状态：开发版
>
> 格式版本：1
>
> 当前支持的拓扑相互作用：harmonic bond（简谐键）

## 1. 文件的作用

`molecular_force` 文件描述当前模拟体系的固定拓扑和对应参数。第一版只描述简谐键，目标是先完成从“读取拓扑”到“GPU 计算能量、力和 virial”的最小闭环。

它不是 GROMACS `.top`、LAMMPS data 或 AMBER topology 的兼容格式。后续可以增加转换工具或扩展新版本，但版本 1 的含义应保持不变。

在 `run.in` 中使用：

```text
potential potential.txt
molecular_force molecular_force.in
```

当前实现仍需保留 GPUMD 原有的 `potential` 命令。`molecular_force` 计算的简谐键贡献会叠加到原有势函数产生的能量、力和 virial 上。

## 2. 完整示例

下面的文件描述 3 个粒子、2 类简谐键参数和 2 条键：

```text
# 文件类型和格式版本
gpumd_molecular_force 1

# 必须与 model.xyz 的原子数一致
number_of_atoms 3

# 参数表包含两类简谐键
harmonic_bond_parameters 2
1.50 20.0
1.30 30.0

# 键表包含两条键
bonds 2
0 1 0
1 2 1
```

其含义是：

- `model.xyz` 中第 0、1 个粒子之间使用第 0 类参数：`r0 = 1.50 Å`、`k = 20.0 eV/Å²`；
- 第 1、2 个粒子之间使用第 1 类参数：`r0 = 1.30 Å`、`k = 30.0 eV/Å²`。

## 3. 各字段的含义

### 3.1 文件头

```text
gpumd_molecular_force 1
```

| 字段 | 含义 |
| --- | --- |
| `gpumd_molecular_force` | 文件类型标识，用于防止把其他输入文件误当作拓扑文件读取。 |
| `1` | 格式版本。当前只接受版本 1。 |

### 3.2 原子数

```text
number_of_atoms N
```

`N` 是拓扑对应的总原子数或粗粒化粒子数，必须是正整数，并且必须等于当前 `model.xyz` 中的粒子数。

拓扑中的编号直接对应 `model.xyz` 的粒子行顺序。编号从 `0` 开始，因此合法范围是 `0` 到 `N - 1`。

如果 `run.in` 使用 `replicate`，必须先执行 `replicate`，再加载与复制后体系匹配的 `molecular_force` 文件。例如：

```text
potential potential.txt
replicate 2 2 1
molecular_force replicated_molecular_force.in
```

### 3.3 简谐键参数表

```text
harmonic_bond_parameters M
r0_0 k_0
r0_1 k_1
...
```

`M` 是参数类型的数量，可以为 0。接下来必须紧跟 `M` 行参数。参数行自身没有显式编号，其行序号就是参数类型编号：第一行是类型 `0`，第二行是类型 `1`，依此类推。

每行包含：

| 参数 | 单位 | 约束 | 含义 |
| --- | --- | --- | --- |
| `r0` / `equilibrium_distance` | Å | 有限正数 | 键的平衡长度。 |
| `k` / `force_constant` | eV/Å² | 有限正数 | 简谐键的力常数。 |

势能定义为：

```text
U(r) = 1/2 * k * (r - r0)^2
```

这里的 `k` 已经对应带有 `1/2` 的表达式。若从其他软件或力场导入参数，必须先确认对方采用的公式和单位，不能只按字段名称直接复制数值。

### 3.4 键表

```text
bonds B
atom_i atom_j type
...
```

`B` 是键的数量，可以为 0。接下来必须紧跟 `B` 行键记录。

| 参数 | 约束 | 含义 |
| --- | --- | --- |
| `atom_i` | `0 <= atom_i < N` | 第一个粒子的全局编号。 |
| `atom_j` | `0 <= atom_j < N`，且不能等于 `atom_i` | 第二个粒子的全局编号。 |
| `type` | `0 <= type < M` | 简谐键参数表的索引。 |

键使用最小镜像约定计算跨周期边界的距离。每条键的能量平均分配给两个粒子，两个粒子获得大小相等、方向相反的力；相应的 per-atom virial 也会被累加。

当前验证允许同一对粒子出现多条键，因为某些力场可能对同一组粒子叠加多个参数项。普通输入中若意外重复，能量和力也会被重复计算，因此生成拓扑时仍应主动检查重复项。

## 4. 语法规则

- 各节顺序固定：文件头、`number_of_atoms`、`harmonic_bond_parameters`、参数记录、`bonds`、键记录。
- 字段以空白字符分隔；可以使用多个空格或制表符。
- 空行会被忽略。
- `#` 后面的内容是注释，会被忽略；注释可以单独占一行，也可以写在数据后面。
- 声明的记录数必须与后续实际记录数一致。
- `bonds` 节之后不允许出现未知内容。
- 整数和实数必须完整合法，例如整数位置不能写 `1.5`，数值后不能附带单位字符串。

## 5. 不含简谐键的合法文件

下面的文件语法合法，可用于检查空拓扑路径；它不会产生任何 molecular force：

```text
gpumd_molecular_force 1
number_of_atoms 3
harmonic_bond_parameters 0
bonds 0
```

## 6. 常见错误

| 错误 | 结果或处理方法 |
| --- | --- |
| 把第一个粒子写成编号 `1` | 本格式从 `0` 开始编号；第一个粒子应写成 `0`。 |
| `number_of_atoms` 与 `model.xyz` 不一致 | GPUMD 在加载时拒绝运行。检查模型是否经过 `replicate`，以及拓扑是否对应同一模型。 |
| `type` 等于参数类型数量 `M` | 最大合法类型是 `M - 1`。 |
| 从 GROMACS 直接复制力常数 | 先核对势能公式，并把长度和能量单位换算为 Å 和 eV。 |
| 在 `molecular_force` 后执行 `replicate` | 当前实现会拒绝；复制会改变原子数和编号，必须先复制再加载匹配的拓扑。 |
| 同一对粒子意外写了两次 | 两条记录都会参与计算，导致该键贡献重复。 |
| 只写 `molecular_force`，不写 `potential` | 当前版本仍需要一个 GPUMD 原有势函数，bonded contribution 是叠加项。 |

## 7. 第一版的边界

版本 1 当前不包含：

- angle、proper/improper dihedral；
- constraint、rigid water、virtual site；
- 原子电荷和 Coulomb；
- 拓扑排除和特殊 1–4 相互作用；
- bonded pair 的 Lennard-Jones 自动排除；
- molecule type、residue、分子实例等高层组织信息；
- GROMACS、LAMMPS、AMBER 或 CHARMM 文件的直接导入。

尤其需要注意：原有 GPUMD 势函数并不知道哪些粒子已经通过键连接，因此第一版不会自动排除 bonded pair 的非键相互作用。当前格式适合验证 harmonic bond 最小闭环，还不能代表完整的通用全原子力场支持。

## 8. 推荐的手工检查顺序

在运行新体系前，建议依次检查：

1. `number_of_atoms` 是否等于最终 `model.xyz` 粒子数；
2. 所有粒子编号是否位于 `0` 到 `N - 1`；
3. 所有 `type` 是否能在参数表中找到；
4. `r0` 和 `k` 的公式、单位是否已经换算；
5. 是否存在意外的重复键；
6. 周期盒是否足够大，键长是否适合最小镜像约定；
7. 原有势函数是否会对 bonded pair 产生不希望的额外非键作用。
