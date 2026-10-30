# RigidNH：各向同性标量 MTK、恒温链与历史参考模式

实现位于 `src/integrate/rigid_nh.cuh/.cu`，上层为
[`Ensemble_Rigid_NPT`](rigid_npt.md)。平动/转动恒温链也用于 NVT。
实现最初参考 LAMMPS `FixRigidNHSmall`；标准路径现在按下述方程核对，
不以旧源码一致性作为物理正确性的依据。

## 变量与质量

三维全周期正交盒，恒定目标 T/P。使用内部时间、能量和压力单位；
`kT=K_B*T`，压力为 eV/Å³，`temperature_time/pressure_time` 是实际耦合时间，
输入 coupling 为步数时需先乘 dt。链变量是摩擦率 xi，不是 pi=Q*xi。
令 g=f_t+f_r，kt=2K_t，kr=2K_r，nu=d(log L)/dt，则 dV/dt=3nu V。

```text
Q_t0 = f_t kT tau_T²       Q_r0 = f_r kT tau_T²
Q_tj = Q_rj = kT tau_T²    (j>0)
W    = (g+3) kT tau_P²
Q_b0 = 9 kT tau_P²         Q_bj = kT tau_P² (j>0)
```

标准路径的 W 是一个标量活塞质量，活塞能量为 W*nu²/2。
Q_b0 的 9 只决定耦合时间尺度，不是九个自由度；活塞链只控制一个自由度。
所有质量均须有限且正。默认链长 10、一个子循环、Yoshida 三系数组合；
支持五系数组合和单节链。f_r=0 时跳过转动链，输入 kr 必须为零。

## 标准路径的运动方程

```text
p_dot = F - [xi_t0 + (1+3/g)*nu] p
P_dot = F_quaternion - [xi_r0 + 3*nu/g] P
nu_dot = 3*[V*(P_mechanical-P0)+(kt+kr)/g]/W - xi_b0*nu

xi_t0_dot = (kt-f_t*kT)/Q_t0 - xi_t1*xi_t0
xi_r0_dot = (kr-f_r*kT)/Q_r0 - xi_r1*xi_r0
xi_b0_dot = (W*nu²-kT)/Q_b0 - xi_b1*xi_b0
xi_j_dot  = (Q_(j-1)*xi_(j-1)²-kT)/Q_j - xi_(j+1)*xi_j
eta_j_dot = xi_j
```

最后一节链没有后一节阻尼。三条链都使用完整的 `-xi_next*xi` 阻尼。
固定驱动力与后一节速率的半步由指数和 sinh(x)/x 的八阶 Maclaurin 近似推进；
链的正反 sweep、坐标更新和 Yoshida 权重关于时间对称。
一次链调用内部输入动能固定，链本身不再缩放物理动量。

转动缩放系数是 `mtk_term2=3nu/g`，不是三倍 `mtk_term2`。
压力驱动力中的 3 与 `nu=d(log L)/dt`、标量能量 W*nu²/2 成对使用；
只改变质量名称而不调整驱动力会破坏能量收支。

机械压力由刚体质心平动和相互作用定义：

```text
3*P_mechanical*V = kt + tr(W_model) - sum_members(r_relative dot f_model)
```

它不把转动动能或恒温/控压冲量再次计入压力。
相应扩展守恒量为模型势能、(kt+kr)/2、P0V、Wnu²/2 以及各链的能量之和。
每条物理链的势项为 kT*(f*eta_0+sum_(j>0)eta_j)，活塞链为 kT*sum_j eta_j；
每节链还包含 Q_j*xi_j²/2。用这些方程求时间导数，各项相消。
这提供独立于旧 LAMMPS 输出的数学核对。

## 每步的调用顺序

初始化仅创建零速率扩展变量，不推进活塞。每一步：

1. 有效当前模型力、动能、机械压力 -> `advance_piston(dt/2, ..., initial)`：
   压力 kick 后阻尼。
2. 力/力矩 kick 半步 -> 旧链/当前 nu 的动量缩放半步 -> 平动/转动及变盒。
3. 中间动能 -> `advance_thermostats(dt, kt, kr)` 和
   `advance_piston_thermostat(dt)`。
4. 新力 -> 新链速率的动量缩放半步 -> 力/力矩 kick 半步。
5. 新动能/压力 -> `advance_piston(dt/2, ..., final)`：阻尼后压力 kick。

初始和末端的活塞调用各推进半步，必须每步成对执行。
`initial(h)` 和 `final(-h)` 互为逆；单独的 `initial(-h)` 不是
`initial(h)` 的逆。动量因子只施加一次，不叠加旧 NHC 的返回因子。

## 显式 LAMMPS 兼容模式

仅 `rigid_npt_lammps` 设置 `Parameters::lammps_compatibility=true`。
保留固定旧参考的数值行为，包括每轴 M 的压力驱动力、不带 3 的 kick、
转动 `9nu/g`、活塞链半强度阻尼，以及 setup 一次初始半步后仅末端半步的顺序。
该模式用于历史轨迹对照，不作为标准 MTK 物理方程的正确性依据。
几何/冲量 virial 的兼容差别见
[兼容入口说明](rigid_npt_lammps.md)。

## 验证

`tests/test_rigid_nh.cpp` 分开检查：

- 标准模式：单节链解析解、标量理想气体/球形转子正反积分、
  扩展能量的二阶步长收敛及暖态的 dH/dt=0。
- 显式兼容模式：固定 LAMMPS 原函数体，36 组配置各连续 100 次更新。
- NVT：温度斜坡链质量/状态的参考对照。
- 零自由度、无效输入和溢出；失败推进不提交候选状态。

固定标量参考为 `c71ef819ab3785bba81c31c0ef5fce84603608d3`。
`tests/fixtures/rigid_nh_lammps_functions.h` 保留原算法、版权和源码 SHA256，
不参与生产程序编译。更新参考必须显式运行：

```bash
python3 tests/fixtures/generate_rigid_nh_reference.py /path/to/lammps
cmake --build build --target test_rigid_nh -j4
ctest --test-dir build -R '^unit.test_rigid_nh$' --output-on-failure
```

GPU 测试另检查混合刚体/自由粒子的完整步时间反演、几何、机械压力以及真实
run.in/NEP 接入。上述有限系统验证不等于生产体系的长期稳定性、密度或体积分布验收。
