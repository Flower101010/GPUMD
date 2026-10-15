# 刚体 NPT 上层模块

`Ensemble_Rigid_NPT` 已将 `RigidNH`、GPU 刚体积分、正交变盒和 thermo 接入
`Integrate`。主要参考本地 LAMMPS `FixRigidNHSmall::setup`、
`initial_integrate`、`final_integrate`；标量链、活塞、MTK 系数保持前一阶段的
LAMMPS 原函数对照。当前范围是三维全周期正交盒、各向同性、恒定 T/P。

## 输入与单位

```text
potential nep.txt
time_step 0.1
ensemble rigid_npt 300 300 1000 0.0001 10000 0
dump_thermo 10
dump_xyz 10 trajectory.xyz precision double mass velocity force group_labels
run 1000
```

完整语法：

```text
ensemble rigid_npt T_initial T_final T_coupling P P_coupling grouping_method
```

- 温度 K，两项必须相同；压力 GPa，有限值即可，允许零/负目标。
- 两个 coupling 是步数，均至少 1；实际 tau=dt*coupling。
- 示例只说明语法；步长/耦合时间需要按体系选择。
- 最后一项为分子 grouping method，单成员组转为自由粒子。
- 内部构造接口接收 eV/Å³；仅输入解析时将 GPa 除以
  `PRESSURE_UNIT_CONVERSION`，连续 run 不重复换算。
- 使用 type=13，独立于已有 Berendsen/SCR 的弹性模量与耦合参数。

物理平动自由度 `f_t=3*(N_free+N_body)`，转动自由度 `f_r=3*N_body`。
自由粒子的动能与自由度成对纳入平动链，也纳入活塞 MTK 的 g。
这扩展了 LAMMPS 单个 rigid fix 只管理刚体的作用范围。
没有全局 COM 自由度扣除；没有刚体时跳过转动链。

## 调用顺序

初始化只投影 bead 速度、创建刚体、统计缓冲及零速率的扩展系统。
首次 `compute1` 才能读取有效初始力：先汇总载荷，计算初始动能/压力，
再执行一次 `advance_piston(dt/2)`，对应 LAMMPS setup 的活塞准备。

每步：

1. `compute1` 用旧链和活塞速率计算两个 `dt/2` 动量因子。
2. 旧力/力矩 kick 半步，然后缩放平动速度和 Miller p。
3. 恒定 nu 下伸缩盒子/质心/自由粒子坐标，推进平动与 Miller 姿态，
   包装质心并重建 bead 坐标/速度。刚体内部位移不缩放。
4. 采样中间动能，两条物理链推进 dt，活塞链推进 dt。
5. 驱动层在新盒子、新坐标下重新计算模型力/virial。
6. `compute2` 用更新后的链速率，先缩放动量，再用新力/力矩 kick 半步。
7. 重建 bead 速度，计算最终动能和含机械约束贡献的压力。
8. 活塞推进 dt/2，准备下一步；thermo 对应这一步的最终物理状态。

链推进过程中不再调用旧 NVT 的 `nose_hoover_chain::propagate`，也没有额外
两个恒温半步。三个链默认各 10 节、一次子循环、Yoshida order=3。
调用次序区分与 LAMMPS 一致的前半步 kick->scale、后半步 scale->kick。

## 盒伸缩和参考实现的边界

使用已有 `isotropic_drift_factors` 精确求解固定 nu、v 子步：

```text
s = exp(nu*dt)
phi = expm1(nu*dt)/nu        (nu=0 时为 dt)
L_new = s*L_old
R_new = s*R_old + phi*v_half
```

实际调用为一次 `rescale_box(s)` 后 `drift_translation(phi)`。
GPUMD 的盒以原点为缩放中心；LAMMPS 的盒围绕中心缩放，且源码将漂移和
两次半盒 remap 分开。这里明确采用上述仿射流，不声称这两种上层编排
逐步轨迹一致，也不把 LAMMPS 的原子仿射映射直接用于刚体内部几何。
每步都重新构建 bead，避免伸缩内部键长。标量模块与 LAMMPS 的数值对照
不等于整个 NPT 算法已通过独立物理验证。

## 压力定义

已有 NVE/NVT 用两个离散约束冲量统计 virial；NPT 的变盒/动量缩放不能
直接塞进这个窗口，否则会把控制器冲量算进压力。
`RigidThermo::sample_constraint_virial` 改从当前状态计算物理约束力：

```text
omega_body = I^-1 L_body
alpha_body = I^-1 (torque_body - omega_body × L_body)
a_cm = F_cm/M
a_i = a_cm + alpha_space × r_i + omega_space × (omega_space × r_i)
f_constraint_i = m_i*a_i - f_model_i
W_constraint = sum_i sym(r_i ⊗ f_constraint_i)
```

`r_i` 是空间坐标中的相对质心位移，无需跨盒 image 历史。
由 `sum m_i*r_i=0`，得到可独立核对的迹：

```text
tr(W_constraint) = -2K_rotation - sum_members(r_i dot f_model_i)
3*P*V = 2K_translation + tr(W_model) - sum_members(r_i dot f_model_i)
```

纯转动不产生理想气体的分子平动压力；刚体内部力的模型 virial 也会
被相应约束项抵消。自由粒子只贡献自身动能和模型 virial。
thermo 在初始有效力和最终物理状态上重新计算，每次只加一次约束贡献。
模型的 force/virial 数组不被改写；输出的 per-atom virial 仍仅为相互作用。
此处与 LAMMPS 的离散冲量估计有有限步长差异，采用的是连续机械约束力。

## 生命周期和验证

相邻 run 保留 ensemble 选择，但重建刚体并重置三条链和活塞。
可以在下一 run 修改步长；单个 run 内必须固定步长和 T/P。
bead restart 没有扩展系统状态，不能视作精确 NPT checkpoint。
不支持 fix/move、外部 deform、非正交/非全周期盒、线性刚体、
自适应步长、correct_velocity 或 unwrapped 存储；其余 action 限制沿用刚体路径。

短测试包括：

- 全自由粒子 100 步 HOST 仿射轨迹和活塞/链调用次序核对。
- 混合体系 100 步：非零力矩、键长和切向速度、动能/DOF、机械压力迹、
  原子模型力/virial 不变、阶段/重置/盒范围检查。
- 实际 run.in：混合和全自由体系、GPa 转换及初始活塞半步、连续 run、
  NPT/NVE/NVT 切换、非零 NEP 力，以及不支持输入的错误路径。
- 既有 RigidNH 标量 LAMMPS 对照和 NVE/NVT 回归。

```bash
cmake --build build --target gpumd test_ensemble_rigid_npt -j4
ctest --test-dir build -R '^(unit.test_ensemble_rigid_npt|integration.rigid_npt_run_in)$' --output-on-failure
```

这些测试核对软件连接、解析关系和短时约束；平衡密度、温度/压力分布、
步长收敛和长时间稳定性仍需后续验收。
