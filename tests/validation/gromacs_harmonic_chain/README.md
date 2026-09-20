# GPUMD–GROMACS harmonic chain 交叉验证

这个测试使用由 8 个 Ar 质量粗粒子组成的线性链，对 GPUMD 的 harmonic bond MD 路径做独立交叉验证。
两套程序使用相同的初始坐标、零初速度、粒子质量、周期盒、7 条 harmonic bond 和
velocity-Verlet NVE 积分；Lennard-Jones 和电荷相互作用均为零。

势能定义和参数为：

```text
U(r) = 1/2 k (r - r0)^2
r0 = 1.5 A
k = 5.0 eV/A^2 = 48242.66606165 kJ mol^-1 nm^-2
dt = 0.05 fs
steps = 2000
```

测试逐帧比较全部笛卡尔坐标、7 个键长、28 个粒子对距离、回转半径、势能和总能。
相邻的 angle 和 full-bonded 用例复用同一运行器，并在 interaction 实际存在时额外比较
6 个键角和 5 个有符号二面角。
成对距离与回转半径不受整体平移或旋转影响，比只比较最终 XYZ 更适合判断链构象是否一致。

在仓库根目录运行：

```bash
cmake -S . -B build
cmake --build build --target gpumd -j2
ctest --test-dir build -R validation.gromacs_harmonic_chain --output-on-failure
```

CMake 会依次查找 `gmx`、`gmx_mpi`，并额外检查 `$HOME/gromacs/bin`。也可以显式指定：

```bash
cmake -S . -B build \
  -DGPUMD_GROMACS_EXECUTABLE=/path/to/gmx_mpi
```

GROMACS 对 `comm-mode = none` 给出一条已知警告；这是为了不在参考轨迹中引入 GPUMD
没有启用的周期性质心速度修正。运行脚本只为这一条警告设置 `-maxwarn 1`。体系初始总动量为零，
所有 bonded force 的总和也为零。

需要保留临时目录中的轨迹、能量和日志时：

```bash
GPUMD_KEEP_VALIDATION_DIR=1 \
ctest --test-dir build -R validation.gromacs_harmonic_chain -V
```

这是 bond-only 的确定性验证，不代表完整经典高分子力场。它没有 angle、dihedral、improper、
拓扑非键排除、special 1–4、Coulomb 或 constraints；这些能力应在各自完成后分层增加交叉验证。

当前目录中的 bond-only 基线与另外两个用例构成定位链：

```text
gromacs_harmonic_chain
    -> gromacs_harmonic_angle_chain
        -> gromacs_full_bonded_chain
```
