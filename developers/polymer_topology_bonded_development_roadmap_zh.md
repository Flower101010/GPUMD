# GPUMD 高分子拓扑与 Bonded Interaction 功能开发路径（调研草案）

> 文档状态：调研阶段草案  
> 版本：v0.1  
> 首个候选验证体系：PNIPAM（poly(N-isopropylacrylamide)）  
> 目标用途：用于与导师讨论项目范围、科学目标、实现路径和验收标准  

## 1. 项目定位

本项目拟为 GPUMD 增加一套通用的分子拓扑与经典力场计算框架，使 GPUMD 在保留现有 NEP、Tersoff、EAM、LJ 等势函数能力的基础上，能够描述具有固定化学连接关系的高分子体系。

项目不应被定位为“PNIPAM 专用功能”，也不应在第一阶段直接定位为“重新实现一个完整的 GROMACS 或 LAMMPS”。更合理的定位是：

> 开发一个通用、可扩展、GPU 原生的固定拓扑框架，支持高分子常用的 bonded interactions、拓扑相关的非键排除及特殊相互作用，并能够与 GPUMD 现有势函数或后续机器学习势协同工作。以 PNIPAM 和简单 bead–spring 高分子作为首批验证体系。

通用性主要通过以下方式实现：

- 拓扑、参数与具体化合物分离；
- interaction style 可扩展；
- 内部使用统一的 GPUMD topology 格式；
- 通过外部转换工具导入 GROMACS、LAMMPS 等格式；
- 第一阶段完整支持一类明确的力场语义，而不是零散支持很多力场。

## 2. 科学目标与开发目标的区分

### 2.1 科学目标

候选科学问题包括：

- PNIPAM 单链的 coil–globule 转变；
- PNIPAM 的温敏行为和 LCST 相关现象；
- tacticity、链长和端基对构象的影响；
- PNIPAM 水化层及氢键结构；
- 多链聚集、polymer brush、nanogel 和交联网络；
- 其他高分子的构象、扩散、缠结、相行为和力学行为；
- 经典力场与 NEP/机器学习势结合的高分子模拟。

不同科学问题需要的模型分辨率不同，应在正式开发前明确选择：

- 全原子模型；
- 联合原子模型；
- 粗粒化模型；
- 经典力场与机器学习势的混合模型。

### 2.2 软件开发目标

软件开发的最低目标不是保证某一 PNIPAM 模型必然复现实验 LCST，而是：

1. 正确表达目标力场的拓扑、参数和相互作用规则；
2. 正确计算能量、力和 virial；
3. 与参考软件在同一模型下达到规定的数值一致性；
4. 在标准 MD 中保持合理的能量守恒；
5. 为增加其他高分子和其他 interaction style 提供稳定接口。

力场参数是否能准确预测 PNIPAM 的实验行为，属于模型选择和参数化问题，需要在程序正确性验证之后单独评估。

## 3. 为什么 PNIPAM 不只是 Bond、Angle 和 Dihedral

PNIPAM 是温敏水溶性高分子。若研究其 coil–globule 转变、LCST、水化或压力响应，结果会明显依赖：

- PNIPAM 的原子电荷和 Lennard-Jones 参数；
- 聚合物—水相互作用；
- 水模型；
- tacticity；
- 链长和端基；
- 温度、压力及采样时间；
- 是否正确处理长程静电；
- 是否正确处理 1–2、1–3 排除和 1–4 缩放。

因此，全原子 PNIPAM 至少涉及：

- atom type、mass、charge；
- harmonic bond；
- harmonic angle；
- proper dihedral；
- improper dihedral；
- Lennard-Jones；
- Coulomb 和长程静电；
- topology exclusions；
- special 1–4 interactions；
- constraints；
- rigid water 和 virtual site（若采用 TIP4P 类水模型）；
- NVT/NPT；
- 正确的 per-atom energy、force 和 virial。

只实现 bond、angle 和 dihedral，可以构成拓扑框架的第一步，但不能直接声称已经支持完整的 PNIPAM 全原子水溶液模拟。

## 4. 建议的总体架构

### 4.1 模块关系

建议将 bonded 模块视为现有势函数之外的附加贡献，而不是直接放入当前多个 `Potential` 的平均逻辑中。

推荐计算顺序：

```text
清零 per-atom energy / force / virial
    ↓
计算现有 GPUMD potential
    ↓
若使用 multiple NEP，完成 observe/average 逻辑
    ↓
计算一次 topology-based interactions
    ↓
计算 topology-aware nonbonded/special pair contributions
    ↓
进入 HNEMD、积分、测量和输出流程
```

这样可保证多个 NEP 做平均时，bonded contribution 不会被重复计算或错误平均。

### 4.2 内部数据模型

建议建立统一拓扑数据模型：

```text
Topology
├── AtomType
│   ├── mass
│   ├── charge
│   ├── sigma / epsilon 或其他非键参数
│   └── force-field type name
├── MoleculeType
│   ├── atoms
│   ├── bonds
│   ├── angles
│   ├── proper dihedrals
│   ├── improper dihedrals
│   ├── constraints
│   ├── virtual sites
│   └── explicit exclusions
├── NonbondedRules
│   ├── mixing rule
│   ├── 1–2 scaling
│   ├── 1–3 scaling
│   ├── 1–4 LJ scaling
│   └── 1–4 Coulomb scaling
└── MoleculeInstances
    └── molecule type × count
```

拓扑实例与参数类型应分开存储，避免同一参数在大量 interaction 中重复。

### 4.3 Interaction style 扩展机制

每一种 interaction 均包含 style 和 type：

```text
bond styles:
    harmonic
    fene
    morse

angle styles:
    harmonic
    cosine

proper dihedral styles:
    fourier
    opls
    ryckaert_bellemans

improper styles:
    harmonic
    periodic
```

加载拓扑后，按照 interaction style 对数据分组，每种 style 启动独立 GPU kernel。这样能够：

- 避免一个大 kernel 中包含大量分支；
- 便于单独测试每种势函数；
- 后续添加新 style 时减少对已有代码的修改；
- 保持 GPU 数据为结构化数组（SoA）。

### 4.4 建议的代码位置

结合当前 GPUMD 仓库结构，初步建议：

```text
src/
├── model/
│   ├── topology.cuh
│   └── topology.cu
├── force/
│   ├── bonded.cuh
│   ├── bonded.cu
│   ├── special_pairs.cuh
│   └── special_pairs.cu
└── main_gpumd/
    └── 在 Run 中增加 topology 解析入口

tools/
└── topology/
    ├── gromacs_to_gpumd.py
    ├── lammps_to_gpumd.py
    └── topology_validator.py
```

是否将拓扑完全放在 `Atom`、`Force` 或独立的 `Topology` 类中，需要在详细设计阶段结合 replicate、restart 和 molecule template 的行为进一步确定。当前更倾向于独立 `Topology` 类，由 `Run` 持有，并传递给 `Force`、replicate 和 restart 相关模块。

## 5. 输入与格式策略

### 5.1 GPUMD 内部统一格式

建议 `run.in` 只增加一个入口，例如：

```text
potential nep.txt
topology topology.gpumd
```

`topology.gpumd` 使用经过展开、无宏、无 include 的规范格式。至少记录：

- 格式版本；
- 单位系统；
- 原子编号基准；
- 原子数量；
- atom types；
- molecule types；
- bond/angle/dihedral/improper types；
- interaction 实例；
- constraints；
- virtual sites；
- exclusions；
- special pair scaling；
- molecule instances。

### 5.2 外部格式转换

不建议第一阶段在 GPUMD C++ 代码中直接实现完整 GROMACS `.top/.itp` 预处理器。

推荐流程：

```text
GROMACS .top/.itp ─┐
LAMMPS data/in     ├─→ 离线转换和检查 ─→ topology.gpumd
其他建模工具       ┘
```

转换工具负责：

- 展开 molecule template；
- 参数匹配；
- 单位转换；
- 原子编号转换；
- exclusions 和 special pairs 生成；
- 缺失参数检查；
- 输出规范化拓扑；
- 输出转换报告和警告。

这样可以降低 C++ 核心复杂度，并且转换结果更容易审核和版本控制。

## 6. GPU 计算方案

### 6.1 第一版并行方式

第一版建议：

```text
one thread → one bond
one thread → one angle
one thread → one dihedral
one thread → one improper
```

每个线程计算一个完整 interaction，并使用原子加法累加到参与原子的：

- force；
- potential energy；
- virial。

该方案实现简单，适合低连接度的分子拓扑。性能不足时，再评估：

- interaction coloring；
- atom-centric gather；
- 按 molecule 或 interaction block 分组；
- 减少 atomic contention；
- kernel fusion。

### 6.2 数值稳定性

必须明确处理：

- bond 长度接近零；
- angle 接近 0 或 180 度；
- `acos` 输入越过 `[-1, 1]`；
- dihedral 中三个连续原子近似共线；
- 二面角正负号和原子顺序；
- triclinic PBC；
- 分子跨越周期边界；
- double precision atomic accumulation；
- 非确定性累加造成的末位差异。

二面角建议使用带符号的 `atan2` 定义，而不是仅使用 `acos`。

### 6.3 Per-atom energy 和 virial

第一版可将 interaction energy 平均分配：

- bond：两个原子各 `E/2`；
- angle：三个原子各 `E/3`；
- dihedral/improper：四个原子各 `E/4`。

该分配可保证总能量正确，但 per-atom energy 分解不是唯一的。对分组能量、局域热流和 HNEMD 的影响需要单独研究。

Virial 必须作为一期核心验收项，因为它直接影响：

- 压力；
- NPT；
- stress；
- elastic 相关计算；
- 热输运相关功能。

在未完成额外理论和数值验证前，不应宣称新增 bonded 模块已兼容 GPUMD 的所有热输运功能。

## 7. 两条模型路线

### 7.1 全原子路线

适合研究：

- PNIPAM 水化和氢键；
- coil–globule 微观机制；
- tacticity；
- 离子、共溶剂和端基效应；
- 聚合物—表面相互作用。

建议首先完整支持一个 OPLS 类力场语义：

- harmonic bond；
- harmonic angle；
- OPLS/Fourier proper dihedral；
- improper；
- LJ mixing；
- charges；
- 1–2、1–3 exclusions；
- 1–4 LJ/Coulomb scaling；
- Ewald/PPPM；
- constraints；
- 一种目标水模型。

若采用 TIP4P/2005、TIP4P/Ice 或 TIP4P-Ew，需要支持刚性水、virtual site 和相应的长程静电处理。

### 7.2 粗粒化路线

适合研究：

- 长链和多链；
- 熔体；
- 聚合物 brush；
- 微凝胶和交联网络；
- 大尺度聚集与相分离；
- 更长时间尺度的构象动力学。

建议支持：

- harmonic bond；
- FENE bond；
- harmonic/cosine angle；
- Fourier dihedral；
- LJ/WCA/Mie；
- 显式或隐式溶剂；
- 温度依赖参数；
- molecule/chain ID；
- 交联拓扑。

粗粒化路线的主要性能收益来自减少自由度和消除部分高频运动，而不只是来自 bonded kernel 的 GPU 化。

## 8. 分阶段开发计划

### Phase 0：科学目标和基准体系冻结

目标：在正式编码前明确首个可复现的科研基准。

需要确定：

- PNIPAM 使用全原子还是粗粒化；
- 单链、多链、brush 还是 nanogel；
- 是否以 LCST/coil–globule 为主要现象；
- tacticity；
- 链长和端基；
- polymer force field；
- 水模型；
- 温度和压力范围；
- 参考文献；
- 参考软件；
- 允许的数值误差；
- 目标 GPU 和体系规模。

交付物：

- 一份冻结的 Phase 1 技术规格；
- 一个在 GROMACS 或 LAMMPS 中可运行的参考体系；
- 单点 energy、force、virial 参考数据；
- 短 MD 参考轨迹；
- 主要科学观测量定义。

预计工作量：约 1–2 周，取决于参考体系是否已经存在。

### Phase 1：通用拓扑核心和 CPU 参考实现

目标：建立与具体化合物无关的拓扑表示和验证基础。

开发内容：

- `Topology` 数据结构；
- atom/molecule/interaction type；
- topology parser；
- 参数和编号检查；
- 单位转换；
- bond graph；
- exclusions 和 special pair 生成；
- CPU reference evaluator；
- topology validator；
- 最小输入示例。

暂不追求高性能，优先确保数据语义明确、错误信息完整。

预计工作量：约 2–4 周。

### Phase 2：GPU Bonded MVP

目标：支持常见固定拓扑 bonded terms，并进入标准 MD 流程。

开发内容：

- harmonic bond；
- harmonic angle；
- Fourier periodic proper dihedral；
- harmonic/periodic improper；
- PBC 和 triclinic box；
- per-atom energy；
- force；
- virial；
- 与现有 potential 的正确叠加；
- 与 multiple NEP average 的正确协同。

验收体系：

- 双原子；
- 非线性三原子；
- 多种正负 torsion 的四原子；
- H₂O 几何测试；
- butane；
- 跨周期边界的分子。

预计工作量：约 4–8 周。

### Phase 3：粗粒化高分子可用版本

目标：首先形成一个可以开展高分子模拟的生产级子集。

开发内容：

- FENE bond；
- LJ/WCA；
- topology-aware exclusions；
- configurable special pair scaling；
- molecule/chain ID；
- NVT/NPT 验证；
- bead–spring chain 构型；
- 简单链熔体 benchmark；
- 性能分析。

科学验证：

- bond length distribution；
- angle distribution；
- `R_g`；
- end-to-end distance；
- RDF；
- NVE energy drift；
- melt density 和压力；
- 与 LAMMPS bead–spring 模型对比。

预计工作量：约 4–8 周。

### Phase 4：PNIPAM 全原子能力

目标：完整表达选定的 PNIPAM 全原子力场和水模型。

开发内容：

- OPLS torsion；
- 选定的 improper 类型；
- atom charge；
- LJ mixing rule；
- exclusions 和 OPLS 1–4 scaling；
- 长程静电与特殊 pair 协同；
- constraints；
- rigid water；
- virtual site（若需要）；
- GROMACS-to-GPUMD 转换；
- restart 中拓扑一致性检查。

科学验证：

- PNIPAM 20-mer 或 30-mer；
- atactic/isotactic/syndiotactic 中选定一种；
- 显式水；
- 多个温度点；
- `R_g`；
- end-to-end distance；
- SASA；
- polymer–water hydrogen bonds；
- hydration number；
- torsion distribution；
- 密度和压力；
- coil/globule 状态分布。

预计工作量：约 2–4 个月，具体取决于 constraints、virtual sites 和现有长程静电代码的可复用程度。

### Phase 5：扩展和优化

候选内容：

- Morse bond；
- Ryckaert–Bellemans dihedral；
- CHARMM/AMBER 风格项；
- class-II cross terms；
- SHAKE/RATTLE/SETTLE 优化；
- 多 GPU topology partition；
- 动态成键；
- 交联网络生成；
- 温度依赖 CG 势；
- 拓扑复制和 molecule template；
- 更完整的 restart；
- 热流和 HNEMD 理论兼容；
- 更丰富的外部格式转换器。

该阶段按科研需求逐项推进，不建议在一期全部承诺。

## 9. 验证与验收标准

### 9.1 单元级数值验证

每一种 interaction style 至少验证：

- 已知几何的解析能量；
- 解析力与坐标有限差分一致；
- 总力接近零；
- 总力矩接近零；
- 整体平移后能量不变；
- 整体旋转后能量不变；
- 正交和 triclinic PBC；
- 分子跨边界；
- virial 与有限应变导数一致；
- CPU 与 GPU 对拍；
- 不同 block size 下结果一致到规定容差。

### 9.2 软件级验证

- 与 LAMMPS/GROMACS 单点能量比较；
- 比较每一个能量分项；
- 比较逐原子力；
- 比较总 virial/stress；
- 短时间 NVE 能量守恒；
- NVT 温度分布；
- NPT 密度和压力；
- restart 前后连续性；
- replicate 和 topology 的一致性；
- multiple potential 模式行为。

### 9.3 科学级验证

对 PNIPAM 等体系，程序数值一致之后再验证：

- 构象统计；
- 水化结构；
- 温度响应；
- 转变区间；
- tacticity 和链长趋势；
- 与参考轨迹、文献和实验的差异。

科学级偏差不能简单归因于代码，需要区分：

- force field 参数；
- 水模型；
- 采样长度；
- 初始构型；
- thermostat/barostat；
- 有限尺寸；
- 程序实现。

## 10. 性能预期

### 10.1 不应提前承诺的内容

第一版不应承诺：

- 全原子 PNIPAM 比 GROMACS 更快；
- 所有高分子都能直接运行；
- 所有经典力场完全兼容；
- NEP 与 bonded 任意相加都会更准确；
- bonded GPU 化必然显著提高总性能。

在显式水全原子体系中，主要耗时通常来自：

- 短程非键作用；
- neighbor list；
- 长程静电；
- constraints；
- GPU/CPU 数据流；
- 多 GPU 通信。

Bonded terms 通常不是唯一或最大的瓶颈。

### 10.2 可能形成优势的方向

GPUMD 扩展可能在以下场景形成价值：

- NEP 与显式拓扑结合；
- 聚合物—无机界面 hybrid potential；
- 单 GPU 全驻留；
- 针对特定 CG 模型优化；
- 超大链或多链粗粒化体系；
- 与 GPUMD 已有材料势和分析模块协同；
- bonded baseline 加 residual NEP。

若采用混合模型，应明确避免重复计能。推荐概念模型：

```text
E_total = E_classical_baseline + E_NEP_residual
```

其中 NEP 的训练目标应与 baseline 定义协同设计，而不是将任意 NEP 与任意 bonded 参数直接相加。

## 11. 主要风险

### 11.1 需求范围失控

“支持各种高分子”如果没有指定力场层次，容易扩展为完整分子模拟软件。控制方法：

- 先支持一个力场族；
- 每个新增 style 必须由具体科学需求驱动；
- 输入转换与计算核心分离；
- 明确一期不支持列表。

### 11.2 力场语义不一致

不同软件对 harmonic 常数、dihedral 符号、phase、单位和 1–4 规则可能采用不同约定。控制方法：

- 每个 style 给出精确数学公式；
- 明确角度单位；
- 明确 atom ordering；
- 转换器进行显式换算；
- 按能量分项对比。

### 11.3 PNIPAM 模型选择不当

PNIPAM 温敏行为对水模型和参数敏感。控制方法：

- 选择一篇目标参考工作；
- 先在原软件中复现；
- 不混用未经验证的 polymer 和 water 参数；
- 将程序验证和力场验证分开。

### 11.4 采样不足

高分子松弛和 coil–globule 转变可能需要较长采样。控制方法：

- 多条独立轨迹；
- 多温度点；
- 必要时增强采样；
- 同时监测多个构象指标；
- 避免仅凭一次塌缩事件判断转变温度。

### 11.5 GPU 写冲突和可扩展性

interaction-centric kernel 需要向多个原子累加。控制方法：

- MVP 使用 atomic accumulation；
- benchmark contention；
- 后续评估 coloring/gather；
- 将正确性和优化分阶段处理。

### 11.6 与 GPUMD 特殊功能的兼容

潜在受影响功能：

- multiple NEP；
- replicate；
- restart；
- minimize；
- PIMD；
- MC 类型交换；
- HNEMD；
- heat current；
- multi-GPU NEP。

一期应提供明确的支持矩阵，对未验证组合主动报错，而不是静默运行。

## 12. 建议的一期支持矩阵

| 功能 | 一期建议 |
|---|---|
| 单 GPU | 支持 |
| CUDA | 支持 |
| HIP | 设计时保留，按 GPUMD 宏接口实现 |
| 固定 topology | 支持 |
| Orthogonal/triclinic PBC | 支持 |
| Harmonic bond/angle | 支持 |
| Fourier proper dihedral | 支持 |
| Improper | 支持一种基础形式 |
| FENE | 粗粒化阶段支持 |
| LJ/WCA | 支持 |
| Exclusions/1–4 | 支持 |
| NVE/NVT/NPT | 支持并验证 |
| Minimize | 力路径复用后验证 |
| Replicate | 初期要求加载 topology 前完成，或使用最终体系拓扑 |
| Restart | 重启时重新加载同一 topology，并检查一致性 |
| PIMD | 暂不承诺 |
| MC 类型交换 | 暂不承诺 |
| HNEMD/heat current | 暂不承诺 |
| Multi-GPU bonded partition | 暂不承诺 |
| 原生 GROMACS parser | 暂不实现，使用转换器 |
| 原生 AMBER/CHARMM parser | 暂不实现 |

## 13. 与导师讨论时需要明确的问题

### 13.1 科学问题

1. PNIPAM 首先研究什么：单链转变、多链聚集、brush、nanogel、界面还是力学？
2. 是否必须研究实验 LCST，还是先研究一般构象动力学？
3. 目标是全原子、联合原子还是粗粒化？
4. 是否需要显式水、离子或共溶剂？
5. tacticity、链长、端基和交联方式是什么？
6. 预期的体系规模和物理时间是多少？

### 13.2 力场和参考

7. 是否已有指定 PNIPAM 力场参数或参考论文？
8. 是否已有可以运行的 GROMACS/LAMMPS 输入文件？
9. 水模型是否已经确定？
10. 是否接受先完整支持 OPLS 类力场，再扩展其他力场？
11. 是否需要与已有文献做到定量复现？
12. 参考软件优先使用 GROMACS 还是 LAMMPS？

### 13.3 项目定位

13. 项目首要贡献是“支持高分子”，还是“实现 NEP–bonded hybrid”？
14. 预期成果是内部科研工具、论文方法，还是希望向 GPUMD 上游贡献？
15. 是否要求 CUDA 和 HIP 同时支持？
16. 是否要求多 GPU？
17. 是否要求第一版就支持热输运和 HNEMD？
18. 项目可接受的开发周期和人力是多少？

### 13.4 验收标准

19. 单点能量、力和 virial 的数值容差是多少？
20. 需要复现哪些统计性质？
21. 性能目标相对于什么软件、硬件和体系定义？
22. 第一篇或第一个项目计划以什么体系产生科研结果？

## 14. 下次讨论后需要形成的文档

与导师确认后，建议将本草案收敛为以下三个文件：

1. **科学需求说明**
   - 研究体系；
   - 目标观测量；
   - 模型分辨率；
   - 参考论文和实验。

2. **Phase 1 技术规格**
   - 明确支持的 interaction styles；
   - 拓扑格式；
   - 数学公式；
   - 单位；
   - 支持矩阵；
   - 验收容差；
   - 不支持范围。

3. **验证计划**
   - 最小单元测试；
   - LAMMPS/GROMACS 参考体系；
   - PNIPAM benchmark；
   - 性能测试；
   - 科学观测量。

## 15. 初步结论

1. PNIPAM 适合作为首个真实验证体系，但不应成为内部架构的硬编码对象。
2. 高分子模拟需要的是通用拓扑框架，而不只是三个 bonded kernels。
3. 第一阶段不必实现完整 GROMACS/AMBER/CHARMM 文件兼容，可使用统一内部格式和外部转换器。
4. 若研究 PNIPAM LCST，全原子模型必须重视水模型、长程静电、exclusions、1–4 scaling、constraints 和采样。
5. 若主要目标是更大的链长和更长时间尺度，应同时建设粗粒化路线。
6. 全原子经典 MD 的第一目标应是正确性和可复现性，不能预设一定快于 GROMACS。
7. GPUMD 最有特色的潜在方向是通用拓扑、粗粒化高分子，以及经典 bonded baseline 与 NEP residual 的混合模拟。

## 16. 调研参考

- GPUMD documentation: <https://gpumd.org/>
- LAMMPS bond style documentation: <https://docs.lammps.org/bond_style.html>
- LAMMPS FENE bond: <https://docs.lammps.org/bond_fene.html>
- LAMMPS special bonds: <https://docs.lammps.org/special_bonds.html>
- GROMACS bonded interactions: <https://manual.gromacs.org/documentation/2025.3/reference-manual/functions/bonded-interactions.html>
- GROMACS topology format: <https://manual.gromacs.org/documentation/2025.1/reference-manual/topologies/topology-file-formats.html>
- Abbott and Stevens, temperature-dependent coarse-grained PNIPAM model: <https://pubmed.ncbi.nlm.nih.gov/26723705/>
- PNIPAM explicit-solvent coarse-grained model: <https://pubs.rsc.org/en/content/articlelanding/2020/cp/d0cp03101a>
- PNIPAM pressure-induced coil–globule study: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8247264/>
- PNIPAM water-model comparison: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9150113/>

