# GPUMD–GROMACS harmonic angle chain 交叉验证

这是三层交叉验证的中间层。在 bond-only 链上增加 6 个 harmonic angles，但不加入
dihedral，用于把 angle 错误与 dihedral 错误分开定位。

```text
theta0 = 110 deg = 1.9198621771937625 rad
k_angle = 2.0 eV/rad^2 = 192.9706642466 kJ mol^-1 rad^-2
```

运行：

```bash
ctest --test-dir build -R validation.gromacs_harmonic_angle_chain --output-on-failure
```
