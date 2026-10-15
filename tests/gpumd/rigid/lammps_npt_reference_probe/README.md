# LAMMPS 刚体 NPT 参考行为核验

这是直接运行 **LAMMPS 可执行程序** 的开发诊断，不运行 GPUMD，也不修改其积分算法。
用于在移植 NPT 盒映射之前确认参考程序的 bead 坐标行为。
工具仅采集观察结果；成功退出不表示某个 NPT 算法通过物理验收。

## 最小体系

- 三个点粒子组成一个非线性三角刚体，初始 bead 1–2 距离为 1 Å。
- `units metal`，质量均为 10，30 Å 全周期正交盒。
- `pair_style zero`：模型力和模型 virial 均为零。
- 默认 bead 初始速度为零，目标温度为 300 K。
- 积分一步，步长 0.001 ps（1 fs），温度/压力耦合时间均为 1 ps。
- 主压力目标 10000 bar（1 GPa），另设 1 bar 和无实际盒变化的对照。

这是隔离几何更新的短测试。零初始速度不能用于证明恒温器的采样能力，
本测试也不以温度平衡、密度或涨落为验收目标。额外的 moving 对照给出
非零统一平动速度，检查观察结果是否只出现在完全静止的初态。

## 重现

提供一个启用 RIGID 包的 LAMMPS 可执行文件：

```bash
python3 tests/gpumd/rigid/lammps_npt_reference_probe/run.py \
  --lammps /path/to/lammps-build/lmp \
  --source /path/to/lammps-source \
  --out-dir tests/gpumd/rigid/lammps_npt_reference_probe/results/my-version
```

`--source` 可选，用于记录完整提交号、源码是否有修改及关键源文件 SHA256。
每次输出包含生成的输入、原始 stdout、LAMMPS dump/log 和 `reference_probe.json`。
JSON 同时记录可执行文件 SHA256、盒长、三条内部距离、几何偏差，以及是否匹配
半次/整次盒长缩放。原始结果目录被 Git 忽略。

需要自行构建参考时，应在独立目录构建，避免改动现有 LAMMPS checkout：

```bash
cmake -S /path/to/lammps-source/cmake -B /tmp/lammps-reference-build \
  -D BUILD_MPI=OFF -D BUILD_OMP=OFF -D PKG_RIGID=ON \
  -D CMAKE_BUILD_TYPE=Release -D DOWNLOAD_POTENTIALS=OFF
cmake --build /tmp/lammps-reference-build -j4
```

## 检查的参考路径

1. `rigid/nvt/small`：固定盒基线。
2. `rigid/npt/small`：目标 10000 bar、1 bar、0 bar；另有非零初始平动速度对照。
3. `rigid/npt`：检查非 small 版本是否有同样行为。
4. `rigid/nph/small`：排除物理恒温链是唯一成因。
5. `rigid/nve/small` 与空组上的 `fix npt`：检查文档推荐的另一条组合路径，
   同时记录两种 fix 定义顺序是否被接受。

[LAMMPS 官方文档](https://docs.lammps.org/fix_rigid.html)明确介绍了空组上的
恒压器与刚体积分器组合。这里仅检查该组合的几何行为；不将其短时结果
解释为长时间系综正确性的证明。

## 观察结果

具体版本和数值记录在下方。所有比较均为实际编译的、未修改的参考源码。

| 参考版本 | 提交 | NVT 的 r12 / Å | NPT（10000 bar）的 r12 / Å | NPT（1 bar）的 r12 / Å |
| --- | --- | --- | --- | --- |
| local-dev | `c71ef819ab` | 1 | 0.9998189423194948 | 0.9999999818925946 |
| stable_22Jul2025 | `c7ae612a94` | 1 | 0.9998189423194948 | 0.9999999818925946 |
| stable_30Sep2026 | `8de817dd79` | 1 | 0.9998189423194948 | 0.9999999818925946 |

初始 r12 均为 1 Å。三种坐标读取方式均复现同一结果：直接用 x/y/z、用
xs/ys/zs 和**当帧盒原点/盒长**重建，以及直接用 xu/yu/zu。所有 image flags
均为零，无跨周期边界；坐标重建误差最大为 1.78e-15 Å。输出保留 17 位有效数字。
这里没有将分数坐标的差直接当作真实距离，也没有用旧盒长换算新帧。

在三种版本中还观察到：

- 无实际盒变化的 NPT 对照保持 r12=1 Å；NPH 也复现缩放。
- `rigid/npt` 与 `rigid/npt/small` 有同样变化；非零平动速度的对照也复现。
- 刚体 NPT 的内部距离比例与 `sqrt(L_new/L_old)` 相符，误差约 1e-15 Å。
  结合源码的 `remap -> set_xv -> remap`，支持末次原子映射导致该现象的解释。
- `rigid/nve/small` 后定义空组 `fix npt`，r12 变为 0.9991854985561108 Å，
  对应整次盒长缩放；反向定义顺序被程序拒绝。

原始帧和日志见本机 `results/<版本>/<case>/`；完整观察快照见
[observed_reference_behavior.json](observed_reference_behavior.json)。

## 对本项目的决策

用户已明确：**以 GPUMD 算法的物理正确性为验收标准，不要求逐步复制 LAMMPS。**
本次未修改 GPUMD 生产算法，不把上述参考行为移植进来。
本观察只说明指定测试中的参考坐标行为，不推断所有 LAMMPS 模式均有问题，
也不直接证明任一实现的长期 NPT 分布正确或错误。

本项目后续应验证约束、步长收敛和系综统计。现有解析子步与短测试通过，
并不能自动证明完整耦合积分及长期采样正确。

