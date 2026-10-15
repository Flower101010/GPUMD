# 刚体模拟使用说明与注意事项

本仓库支持由多个 bead 组成的非线性刚体，并允许与自由粒子混合。
刚体在原子势（包括温度无关的 NEP）提供的 bead 受力下平动、转动，
内部几何保持不变。不同刚体可以有不同数量的 bead。

适用大量小刚体的体系，如果刚体中的bead过多会出现较为明显的性能问题。（没有测试，应该会有）

## 准备分子分组

复用了GPUMD原有的分组系统来表示刚体。

在 `model.xyz` 的 `Properties` 中提供 `group:I:1`，用同一个非负标签
表示同一分子的所有 bead。例如：

```text
4
Lattice="30 0 0 0 30 0 0 0 30" pbc="T T T" Properties=species:S:1:pos:R:3:group:I:1
C 10.0 10.0 10.0 10
C 11.0 10.0 10.0 10
C 10.3 10.8 10.0 10
C 15.0 15.0 15.0 -1
```

前三个 bead 构成一个三成员刚体，最后一个为自由粒子。
每个自由粒子必须有独立标签，否则同标签的多个粒子会被视作一个刚体。
-1 标签表示自由粒子不参考刚体的构成。

分组方法索引从 0 开始；上例只有一种分组方法，因此选择 `0`。
请选择表示分子归属的分组，不要误用物种或温控区域分组。
示例中的元素、盒子和坐标只用于说明格式，实际体系需要匹配的势文件。

初始化时，刚体应足够紧凑，在每个周期方向上的一致展开跨度小于半盒长。
当前不支持线性、近线性、重合或其他退化刚体。
初始 bead 速度会投影为刚体平动和转动，不会自动重标定到目标温度。

## 选择系综

| 系综 | 命令 | 盒子与温度要求 |
| --- | --- | --- |
| NVE | `ensemble rigid_nve grouping_method` | 固定盒，支持现有周期盒几何 |
| NVT | `ensemble rigid_nvt T_initial T_final T_coupling grouping_method` | 固定盒；允许目标温度线性变化 |
| NPT | `ensemble rigid_npt T_initial T_final T_coupling P P_coupling grouping_method` | 三维全周期正交盒，各向同性伸缩；两项温度必须相同 |

温度单位为 K，压力单位为 GPa。两个 coupling 均以积分步数计，
必须为有限值且至少为 1；实际耦合时间为 `coupling * time_step`。
NPT 使用恒定目标压力，不需要输入弹性模量。

NVT 使用独立的平动链和转动链，采用 LAMMPS 刚体恒温算法，并与 NPT 共用
链推进实现。平动链同时控制自由粒子和刚体质心，转动链控制刚体旋转；
没有刚体时只使用平动链。每条链默认 10 节，温度斜坡期间链质量随目标温度更新。
这替换了早期 NVT 的平动/转动共用单链实现，输入语法不变，轨迹会发生变化。

以下为 NPT 输入示例：

```text
potential nep.txt
time_step 0.1
ensemble rigid_npt 300 300 1000 0.0001 10000 0
dump_thermo 10
dump_xyz 10 trajectory.xyz precision double velocity force group_labels
run 1000
```

这里 `time_step` 为 0.1 fs，温度和压力耦合时间分别为 100 fs 和 1000 fs。
这些参数仅演示语法，步长、耦合时间和运行长度需要按实际体系验证。
使用 NVT 时，将系综命令替换为 `ensemble rigid_nvt 300 300 1000 0`；
使用 NVE 时替换为 `ensemble rigid_nve 0`。
详细参数说明见[ensemble 命令](input_parameters/ensemble_standard.rst)。

## 输出与运行注意事项

- `thermo.out` 的温度和动能使用 `3*N_free + 6*N_body` 个自由度，
  不自动扣除全体系质心运动自由度。
- 目前明确不支持线形刚体和近线形的刚体。
- 全局压力包含刚体约束贡献；XYZ 等输出中的 per-atom virial 仍只含相互作用。
  不要直接用这些 per-atom virial 重建完整约束压力。
- NPT 伸缩盒子、刚体质心和自由粒子坐标，保持刚体内部距离不变。
  NVT/NPT 的约束压力使用当前状态的瞬时机械约束力估计。
- 使用正的固定时间步长。不支持 `fix`、`move`、外部 `deform`、自适应步长、
  `correct_velocity`、unwrapped 坐标存储和温度相关势。
- 当前开放的输出/测量包括 `dump_thermo`、wrapped `dump_xyz`、
  wrapped `dump_netcdf`（需要编译支持）、`dump_restart`、`compute_rdf`、
  `compute_adf`、`compute_angular_rdf` 和 `compute_orientorder`。
  其他 action 尚未适配。
- 相邻 `run` 保留系综选择，但每次重新构建刚体；NVT 重置两条恒温链，
  NPT 重置三条链和活塞。下一次 `run` 可以修改步长，输出命令需按
  GPUMD 的常规测量生命周期重新设置。
- bead restart 文件不保存完整刚体和扩展系统状态，不能作为精确动力学续跑检查点。

## 验证状态

NVE 已有测试和运行验证。新版双链 NVT 的长期统计需要重新验证，
旧版共用单链的长时间测试结果不能作为新版的验收结果。
NPT 已通过短时约束、压力公式、输入/输出、NEP 调用及既有功能回归检查；
长时间稳定性、平衡密度和 NPT 系综统计仍待验收。
正式采样前，应检查平衡后温度、压力、密度/体积的时间序列，并验证步长收敛。
