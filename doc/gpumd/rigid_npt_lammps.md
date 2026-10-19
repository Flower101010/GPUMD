# rigid_npt_lammps：LAMMPS 刚体 NPT 兼容模式

```text
time_step 0.5
ensemble rigid_npt_lammps 300 300 200 0.0001 2000 0
```

参数与 `rigid_npt` 相同：温度 K、压力 GPa、两个耦合参数为步数。
上例对应 300 K、1 bar、温度耦合 100 fs、压力耦合 1 ps。
仅支持三维全周期正交盒、恒定目标温度/压力、非线性刚体；其他输出与 action
限制沿用[刚体使用说明](rigid_body_guide_zh.md)。不记录 image 历史，不支持 unwrapped 输出。

## 两个 NPT 模式的区别

默认 `rigid_npt` 继续保持现有的质心变盒和机械约束压力。
本模式显式选择 LAMMPS `FixRigidNHSmall` 的顺序：

1. 初始速度投影和离散约束 virial 估计，准备初始活塞半步。
2. 旧力半步 kick、旧速率缩放、参考 COM 漂移系数和 Miller 转动。
3. 推进物理链与活塞链，然后执行 `remap -> set_xv -> remap`。
4. 在末次仿射映射后的 bead 坐标计算相互作用力；用实际缩放力臂汇总力矩。
5. 新速率缩放、新力半步 kick，`set_v` 更新速度；两次速度变化共同构造
   `fc=m*(v_after-v_before)/(dt/2)-f_model` 的离散约束 virial，再更新活塞。

**末次 remap 有意改变步末输出和计算模型力时的内部距离。**
这是兼容目标，不是几何保持模式。体内参考几何仍固定，每步重建后再被末次映射缩放，
不是持续修改参考边长。若要求每个完整步的 bead 内部距离不变，请继续使用 `rigid_npt`。

动量缩放、链和 MTK 复用已经对照 LAMMPS 的 `RigidNH`。NHC/活塞冲量下合约束力
不一定为零，因此离散 virial 保留绝对坐标，而非只使用相对质心坐标。
GPUMD 的盒原点仍为零，计算兼容 virial 时平移到初始盒中心固定的 LAMMPS 坐标约定；
比较绝对轨迹坐标时也需扣除 LAMMPS 当帧 `boxlo`。

## 对照范围与实跑结果

参考为未修改 LAMMPS `stable_30Sep2026`，提交
`8de817dd79bfe4525d5d39246a212d833e6dee07`。
边长 3 Å 的等边三角形、每 bead 12 u、30 Å 周期盒、步长 0.5 fs。
温度 300 K，tau_T=100 fs，tau_P=1000 fs；除 static_1GPa 外目标压力为 1 bar。
静止单步用零初速；moving 使用共同生成的刚体平动及转动速度。
interacting 对照使用两个不同取向的三角形与相同的 LJ 参数。

| 对照 | 步数 | LAMMPS Δr12 / Å | rigid_npt_lammps Δr12 / Å |
| --- | --- | --- | --- |
| static_1bar | 1 | -1.3580555346948131e-08 | -1.3580553570591292e-08 |
| static_1GPa | 1 | -0.0001358024812585512 | -0.00013580243887467702 |
| moving_1bar | 1 | 4.891938720774647e-08 | 4.8919452932949525e-08 |
| moving_100steps | 100 | 2.7994185538560146e-06 | 2.7993979569984617e-06 |
| interacting_two_bodies | 10 | 2.470077857807951e-07 | 2.470079323302343e-07 |

这些对照还检查末态盒长、温度与压力，并以几何保持的 `rigid_npt` 为控制组。
GPUMD/LAMMPS 的单位换算常量及 LJ 计算精度不同，不要求逐位轨迹相同。
直接对照覆盖纯刚体；若混有自由粒子，沿用本项目全体系平动链/自由度扩展，
不声称等同于 LAMMPS 中任意刚体 fix + 溶剂 fix 的组合。
对照也不代表已完成该兼容模式的长期 NPT 统计验收。

轨迹、输入、日志和 JSON 位于本机：
`tests/gpumd/rigid/lammps_npt_reference_probe/results/lammps_compat_pressure/<case>/<engine>/`。
LAMMPS 使用 `state.dump`；GPUMD 使用 `trajectory.xyz`。原始运行目录被 Git 忽略。
固定参考值与版本/源码 SHA256 则跟踪在 `tests/fixtures/rigid_npt_lammps_run_in.json`。

## 重现

在仓库根目录执行：

```bash
python3 tests/test_rigid_npt_lammps_run_in.py --gpumd build/gpumd \
  --lammps /path/to/lammps-build/lmp --lammps-source /path/to/lammps-source \
  --out-dir /tmp/rigid-npt-lammps-comparison
```

省略 `--lammps` / `--lammps-source` 则使用固定参考值运行自动回归。
`ctest --test-dir build -R 'integration.rigid_npt_lammps_run_in' --output-on-failure`
执行相同回归。只在主动更新参考时使用 `--write-reference`；它只保存真实 LAMMPS 输出，
不会把 GPUMD 的输出提升为参考。
