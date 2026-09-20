# GPUMD–GROMACS 完整 bonded chain 交叉验证

本用例复用相邻 `gromacs_harmonic_chain` 中的 8 粒子初始构型和 NVE 设置，在 bond-only
基线之上加入 6 个 harmonic angles 和 5 个 periodic proper dihedrals：

```text
bond:     U = 1/2 k_b (r-r0)^2
          r0 = 1.5 A, k_b = 5.0 eV/A^2

angle:    U = 1/2 k_a (theta-theta0)^2
          theta0 = 110 deg, k_a = 2.0 eV/rad^2

dihedral: U = k_d [1 + cos(n phi - phase)]
          k_d = 0.2 eV, n = 3, phase = 0.4 rad
```

对应 GROMACS 参数换算为：

```text
k_b = 48242.66606165 kJ mol^-1 nm^-2
k_a = 192.9706642466 kJ mol^-1 rad^-2
k_d = 19.29706642466 kJ mol^-1
phase = 22.9183118052 deg
```

运行：

```bash
ctest --test-dir build -R validation.gromacs_full_bonded_chain --output-on-failure
```

应先保留并运行 bond-only 基线。完整用例若发生差异，前者可以排除 harmonic bond 和通用
积分/轨迹转换路径；再结合 `gromacs_harmonic_angle_chain` 中间层，可以把问题进一步缩小到
angle 或 dihedral。
