# Harmonic bond 三粒子示例

这个示例用三个 Ar 粒子构造一个最简单的线性“分子”：

```text
0 -------- 1 -------- 2
     bond 0     bond 1
```

选择 Ar 只是为了复用 GPUMD 内置的原子质量。`zero_lj.txt` 中的 Lennard-Jones `epsilon` 为 0，因此原有势函数不产生能量或力；体系的全部势能和力都来自新实现的 harmonic bond。

## 模型和解析结果

两条键使用同一组参数：

```text
r0 = 1.5 Å
k  = 5.0 eV/Å²
U(r) = 1/2 * k * (r - r0)²
```

初始键长分别为 `1.8 Å` 和 `2.1 Å`，所以初始势能应为：

```text
U01 = 1/2 * 5.0 * (1.8 - 1.5)² = 0.225 eV
U12 = 1/2 * 5.0 * (2.1 - 1.5)² = 0.900 eV
Utotal = 1.125 eV
```

初始 x 方向力应为：

```text
F0 = +1.5 eV/Å
F1 = +1.5 eV/Å
F2 = -3.0 eV/Å
F0 + F1 + F2 = 0
```

中间粒子同时属于两条键，因此这个体系也检查了多条 bond 对同一粒子的力累加。

## 运行方法

在仓库根目录运行：

```bash
./examples/gpumd_harmonic_trimer/run_example.sh
```

也可以指定其他 GPUMD 可执行文件：

```bash
./examples/gpumd_harmonic_trimer/run_example.sh /path/to/gpumd
```

脚本会运行 4000 步 NVE，然后检查：

- `thermo.out` 中至少有两条有效数据；
- 动能和势能在振动过程中发生交换；
- 总能量的相对变化范围不超过 `1e-3`；
- `movie.xyz` 中两条键的长度确实随时间变化。

输出文件保留在本目录：

- `gpumd.log`：GPUMD 标准输出；
- `thermo.out`：温度、动能、势能和压力；
- `movie.xyz`：粒子位置轨迹。

## 这个示例能说明什么

通过该示例可以检查 harmonic bond 是否真正参与 MD，并验证短时间 NVE 中能量与力的基本一致性。

它还不能验证完整高分子力场，因为当前没有 angle、dihedral、非键排除、Coulomb 和 1–4 相互作用。后续仍应使用 GROMACS 或其他独立实现进行逐项交叉对照。
