# GPUMD 多 bead CG 功能合并与补全规划

日期：2026-10-09。状态：阶段 A/B 完成；训练 baseline 和逐帧拓扑尚未实现。
当前结果见 [阶段 B 状态](cg_stage_b/STATUS.md)，历史基线见 [阶段 A 状态](cg_stage_a/STATUS.md)，接口草案见 [输入说明](cg_stage_a/README.md)。

## 1. 目标和范围

用户确认的目标：一个分子可由多个 CG bead 表示；训练数据允许不同 bead 数、不同分子数和不同连接拓扑，共享 bead 类型定义与 bonded 参数表。每次 MD 运行的连接拓扑固定。

最终交付应包含：同一数据与参数约定下的训练、预测、MD、例子、文档和自动验证。固定拓扑小体系是中间检查点，不是最终功能范围。

第一版采用：

- harmonic bond、harmonic angle、periodic proper dihedral；同一四元组可叠加多个 dihedral 项。
- 固定 bonded 参数 + 可训练 NEP 残差势；参数来源由用户提供，可在外部拟合。
- 普通势能型 `nep` 训练与预测，以及 `gpumd` MD。
- 多种 bead 类型、多分子混合体系、不同大小/拓扑的训练帧。
- 空 interaction 列表、一 bead 分子与多 bead 分子共存。
- 以三维周期体系为首批闭环用例；部分周期和非周期的支持边界必须明确，不允许训练与 MD 静默采用不同边界。

后续候选：`gnep` 梯度训练、bonded 参数联合优化、其他 bond/angle 势型、improper、自动拓扑复制、拓扑相关非键排除、拓扑感知 NEP 描述符、运行时反应。第一版不自动扩展到这些功能。

## 2. 总模型和监督约定

总模型为：

```text
U_total = U_bond + U_angle + U_dihedral + U_NEP_residual
F_total = F_bonded + F_NEP_residual
W_total = W_bonded + W_NEP_residual
```

`NEP_residual` 可以包含分子间相互作用与尚未被固定 bonded 项描述的分子内贡献，不能直接等同于纯非键势。

训练输入沿用参考标签的总量含义：力必需；能量、virial 按原训练器支持的标签机制使用。标签应与目标 CG 模型对应，不能自动把任意 AA 瞬时能量当作 CG 有效势能标签。本项目负责消费标签，不在本轮重设计标签生成方法。

训练器在预测后加一次 bonded baseline，再计算 loss；预测文件默认输出总模型结果。内部保留原始参考标签，避免用户离线减 bonded 后训练器再次相加所造成的含义混乱。文档须说明应输入 total 标签。

固定 baseline 的坐标导数参与总力、总 virial；其参数不进入 NEP 优化变量。缺少某类参考标签时，不能凭 baseline 自动制造该类监督。

NEP 的普通描述符只看到几何和 bead 类型。逐帧拓扑只改变显式 bonded 项，不能保证几何和类型相同而连接不同的体系具有可区分的 residual。此限制需写入使用说明。

## 3. 合并原则

以 `GPUMD` 现有实现为基础，保留：

- `Topology`、`ForceFieldParameters` 与校验；
- 分离的 GPU data 与三个计算模块；
- `MolecularForce` 以及已有 `molecular_force` MD 接口；
- v1/v2 输入兼容性与已有测试。

从 `GPUMD-5.6_bondangledihedral_need_test` 吸收：

- CPU/GPU 共用 evaluator 的思路；
- interaction 能量、力、virial 公共累加的思路；
- 固定 bonded baseline 接入 NEP 训练与预测；
- 逐帧连接表和共享参数表；
- 有限差分与应变导数测试思路。

按功能迁移并接入当前分支，不整文件覆盖较新的训练器，也不留下两套长期并行的 bonded 内核或内部拓扑模型。

## 4. 数值与格式约定

内部继续采用当前实现：

```text
bond:     U = 1/2 k_b (r-r0)^2
angle:    U = 1/2 k_a (theta-theta0)^2
dihedral: U = K [1+cos(n phi-delta)]
```

内部单位为 eV、Å、rad，原子与 interaction type 索引从 0 开始。bond/angle 从对方格式导入时 `k=2K`；输入角度从 degree 转 rad；外部 1-based 索引转 0-based。转换必须有明确的源格式或版本，不能靠数值猜单位。

bead type 与 bond/angle/dihedral type 是不同命名空间。bead type 用于 NEP 与质量定义，interaction type 索引对应参数表；不能由元素或 bead type 自动推断连接关系。

MD 继续兼容 v1/v2。新增训练参数文件独立版本化，共享参数文件 + 每帧连接表是主路径；旧完整拓扑文件可作为固定拓扑简写。具体关键字在接口阶段确定，训练与 MD 的新输入尽量使用同一参数来源。

逐帧拓扑使用该帧局部编号。所有有 bonded 参数的训练帧都必须显式给出三类列表；空列表明确声明，缺失列表报错。检查越界、重复原子、缺失/重复系数、非法类型、非有限数值和计数不符。

重复 interaction 不能一律禁止：同一四元组的多个 dihedral 项是合法用法；区分 interaction 内重复原子与多个合法叠加项，避免去重工具删掉物理项。

## 5. 阶段与验收

| 阶段 | 工作 | 验收条件 |
| --- | --- | --- |
| A：冻结约定与建立基线 | 记录两个源目录版本/文件差异；明确模型、单位、标签、逐帧编号、边界条件和输入草案；在有工具链环境运行已有测试 | 新旧格式转换例子可核对；测试 PASS/SKIP/未运行分别记录 |
| B：补全计算核心 | 共用 host/device evaluator；统一退化检测；抽取累加工具；保持 GPU SoA 数据；删除重复校验路径 | 三类 E/F/W 有限差分与不变性通过，v1/v2 行为兼容，普通构型不回归 |
| C：逐帧拓扑与训练 baseline | 参数读取与拓扑读取分离；Structure 持有帧拓扑；Dataset 正确打包 baseline；精度策略和 PBC 与 MD 对齐 | 混合 bead 数和拓扑可打包；shuffle、batch、train/test、设备分配不会错配；CPU baseline 对齐 GPU bonded 输出 |
| D：NEP 训练/预测闭环 | 在普通 NEP 总预测路径加 baseline；检查 loss、prediction、训练输出与可选加速路径 | 标签为总量；每次评估仅加一次；总预测=残差预测+baseline；关闭功能后回归通过 |
| E：模型交付与 MD | 输出伴随参数/约定清单；MD 加载共享参数与对应体系拓扑；记录 bead 类型和质量；检查误用及 restart/replicate 顺序 | 同构型的 NEP prediction 与 gpumd 单点 E/F/W 对齐；多分子多类型体系可运行；训练与部署参数来源可追溯 |
| F：验证与发布准备 | 合成训练、小型多分子 MD、跨软件验证、较大体系运行与性能记录；补文档 | 功能与验收矩阵完整；不能以低训练误差替代 MD 和物理验证 |

### B：退化几何的具体决策

不能继续使用“几何异常时静默跳过 interaction”。应区分：

- 重合原子/零长度键：对有非零强度的有效 interaction 视为非法几何，报错并定位帧或原子；不能漏掉 harmonic 势能后继续运行。
- 普通近共线 angle：使用更稳定的角度计算与相对尺度检测；在可定义范围内保持 E/F 一致。
- 严格共线 angle：先分析平衡角为端点时的可定义极限；非平衡共线情形缺少唯一梯度方向，不统一假装力为零。第一版允许对未支持情况明确报错。
- dihedral 的法向量退化：二面角未定义，明确报错；近退化状态不得仅以固定长度阈值静默删除。
- 零强度 interaction：明确是否允许；如允许，应在求几何前当作关闭项处理。

CPU 返回可检查状态；GPU 用错误标记/索引回传宿主，避免每条 interaction 的常规路径增加同步。实际代价须测量，不把规划中的优化当作性能结论。

### C：精度和边界

baseline evaluator 和 CPU 累加优先 double；写入现有 float 训练数组时再转换，评估强 bonded 项加小 residual 的误差。训练坐标本身目前是 float，不能只改 evaluator 就声称全部训练输入具有 double 精度。

共享几何约定；显式管理周期方向与原始盒。若现有 NEP 训练流程限制了边界条件，先一致地拒绝未支持的输入，并在文档说明。比较正交、三斜、跨边界和半盒边界用例，不能只验证单一非周期几何。

训练用 6 分量、MD 用 9 分量 virial，转换、符号、单位和总量/每 bead 归一化必须按既有接口逐项验证。第一版重点使用总 virial 监督；逐 bead virial 依赖分配约定，需要独立定义后才能支持。

### D/E：避免模型部署遗漏

`nep.txt` 表示残差模型时，完整模型交付至少包含：NEP 文件、bonded 参数文件、bead 类型对应关系、单位/势函数版本和加载说明。每次 MD 另提供该体系连接表及质量。

规划增加伴随清单和一致性检查，避免只加载 residual 文件而漏掉 bonded。清单格式与是否扩展模型头部在 A 阶段决定；保持原有普通 NEP 的兼容性。

需要检查现有 NEP 编译/加速推理路径是否绕过统一预测入口。不能只在一个常规 kernel 末尾接入后就宣称所有训练与预测路径完成。

## 6. 最小验证矩阵

1. 两 bead bond、三 bead angle、四 bead dihedral：已知能量、坐标有限差分力、全张量应变导数、总力/总力矩、平移/旋转不变性。
2. 端点角、近共线、零长度、零强度和非法输入：得到约定的结果或可定位错误；不产生静默丢项。
3. 正交/三斜 PBC、跨盒连接、长链展开：CPU/GPU 一致。声明 MIC 假设，连接向量必须能由最小像唯一恢复；不把任意长键自动当作正确。
4. 可手工计算的两帧：不同 bead 数、不同 interaction 表、不同 type、多项 dihedral；检查打包 offset、输出顺序与 shuffle 后结果。
5. baseline 恒等式：同一冻结 NEP 和构型，有/无 bonded 的预测差应恰为独立 baseline，逐帧核对 E/F/W。
6. 合成参考集：先使用纯 bonded，再使用已知 residual + bonded；检查训练能否恢复 residual 和总模型。不要要求随机训练精确零误差，阈值由精度基线确定。
7. 模型部署一致性：相同构型、参数和 NEP 文件，在训练 prediction 与 gpumd 单点计算对齐 E/F/W。
8. 小型多分子液体/链体系：NVE 随时间步减小的漂移趋势；NVT 稳定性；键长/键角/二面角分布。需要压力用途时增加 virial/压力与 NPT 检查。
9. 保留既有 GROMACS 小链交叉验证；按 bond、bond+angle、完整 bonded 顺序定位差异。
10. 更大 bead 数与不同连接度：记录耗时、显存、原子写入竞争、异常几何数量与重复运行差异。通过正确性后再优化。
11. 不开启 bonded 功能：原训练、prediction 和 MD 回归；GPU 不可用的 SKIP 不算通过。

## 7. 完成标准和下一入口

完成标准：变大小/变拓扑数据可训练并预测；完整模型可在固定拓扑的多 bead、多分子体系中运行；CPU/GPU、训练/MD 的 E/F/W 一致；异常输入可定位；原功能保持兼容；交付例子能复现全流程。

阶段 A/B/C 已完成：输入草案、两帧最小数据、基线、容器测试修正、共用 evaluator/累加、可定位退化错误，以及共享参数/逐帧拓扑读取、Structure/Dataset baseline 打包和验收。阶段 C 记录见 `cg_stage_c/STATUS.md`；训练关键字尚未开放。下一步是阶段 D：接入 residual prediction、loss 和输出，并验证 baseline 仅相加一次。测试随各阶段进入，不等合并完成后集中补。

当前证据边界：host 检查通过；CUDA 构建成功；CTest 14/14 通过；两帧 GPU E/F/W 对照、50 fs 时间步收敛检查及 Compute Sanitizer 通过。GROMACS、NEP 训练、新草案读取器和长时间/大体系验证尚未完成。详细数值和提交边界见阶段 B 文档。
