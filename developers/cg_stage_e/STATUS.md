# 阶段 E：完整 CG 模型交付与 MD 加载

2026-10-09 完成；实现提交 `362a0f8f`。分支 `codex/multibead-cg-stage-b`。
冻结摘要见 `evidence/validation_20261009.json`，可运行的合成模型包见 `example/`。

## 新接口与交付

训练入口沿用阶段 D：`molecular_force bonded.in per_frame`。每次写出当前模型或
checkpoint 时，NEP 自动写出伴随清单及 **内存中实际训练参数** 的 double 精度快照：

```text
nep.txt
cg_model.json
cg_model.bonded.in
```

Checkpoint 的后两者为 `<nep_filename>.cg_model.json` 与
`<nep_filename>.cg_model.bonded.in`。拷贝这三个文件即可搬移模型，不依赖原训练目录
或原始参数文件。清单记录 SHA-256、bead 类型顺序、eV/angstrom/radian、harmonic k/2、
proper periodic dihedral、consecutive MIC 和格式版本。成员必须位于清单同一目录。
拓扑和质量属于具体 MD 系统，不固定在共享模型中。

MD 推荐一次加载完整模型：

```text
cg_model ../package/cg_model.json topology.in
```

也提供底层分离入口：

```text
potential residual_without_companion.txt
molecular_force bonded_parameters.in topology.in
```

旧 `molecular_force combined.in` 的 v1/v2 仍保留。独立拓扑使用 `gpumd_topology 1`，
包含 `number_of_atoms` 及 bonds/angles/dihedrals 三节，参数与原子编号均从零开始。
详细格式见 `doc/gpumd/input_parameters/cg_model.rst` 与 `molecular_force.rst`。

完整加载检查文件哈希、约定、类型顺序、原子数、索引和三维 PBC。只能声明一个 CG
模型，不能混用其他 potential/molecular_force。replicate 必须先执行，随后提供完整
复制体系的拓扑；restart XYZ 仍需重新声明模型和对应拓扑。质量从 model.xyz 读取。
初始化、dump_xyz 的模型类型检查均支持新入口。CG 加载保留完整共享类型表，不压缩
本体系中缺失的类型。

已知 companion 存在且指向该 residual 时，普通 `potential` 会拒绝 residual 单文件
加载。**删除 companion 或单独改名后无法自动识别 residual**；尚未在 NEP 文件头新增
标记，不能宣称任何遗漏均会被检出。哈希用于防止模型包误配，不是数字签名。

## 验收

- CUDA 构建成功；19/19 CTest PASS、0 skipped，包括 manifest/SHA-256/独立拓扑单元测试。
- SHA-256 对照三个标准输入，并由 production exporter 与 Python hashlib 独立交叉检查。
  当前模型和 checkpoint 均导出完整清单；原输入参数损坏后，复制出的包仍可加载。
- 4/6 bead 两个不同拓扑、共享 C/O 和参数表：同一非零 frozen residual，训练预测与 MD
  使用相同 float 可表示坐标。帧总 E 最大差 `1.62e-7 eV`、F 最大差
  `3.28e-6 eV/angstrom`、总 W 最大差 `1.85e-6 eV`；训练文本为原有六位有效数字，
  训练数组为 float、MD 累加为 double，阈值预设为 `3e-5`。
- 完整包、split 和 legacy v2 的 MD E/F/9 分量 W 在 `2e-12` 阈值内一致。
  legacy v1 兼容另由原有输入测试覆盖。
- 删除体系中的 C 类型、全部采用 O，完整类型表路径与普通 NEP 自动压缩路径一致。
- 两种体系都成功 dump/reload restart；实际 `replicate 2 1 1` 使用复制后拓扑，能量和
  virial 为原体系两倍。restart 对照容差 `3e-5`，不声称长期轨迹逐位续接。
- 输入保护覆盖缺拓扑、重复/混合声明及两种顺序、模型后 replicate、只加载 residual、
  原子数不符、非三维 PBC、修改参数后哈希不符；host 测试还覆盖错误清单单位、版本、
  图像约定、类型顺序、路径、重复字段、尾部内容以及越界拓扑。
- 每个体系 NVE 50 fs，dt=0.05/0.025 fs。最大能量偏差分别：

| 体系 | dt=0.05 / eV | dt=0.025 / eV |
| --- | ---: | ---: |
| 6 bead | 7.44e-7 | 1.99e-7 |
| 4 bead | 1.27e-6 | 3.25e-7 |

步长减半后偏差约降至四分之一。全部相对偏差低于 `3.41e-6`。
普通 NEP 不启用 bonded 的一代训练与 fresh train/test prediction smoke 通过；
这不是原版二进制的逐位回归。Compute Sanitizer memcheck：两个示例各 1000 步，均为 0 errors；新模型模块的 C++14 独立编译通过（不等同完整 make 构建）。

## 重现

```bash
/tmp/gpumd-stage-a-tools/bin/cmake --build /tmp/gpumd-stage-a-build --parallel 2 --target gpumd nep check
python3 developers/cg_stage_e/run_md_validation.py \
  /tmp/gpumd-stage-a-build/gpumd /tmp/gpumd-stage-a-build/nep \
  /tmp/gpumd-stage-a-build/tests/test_bonded_nep
python3 developers/cg_stage_c/run_nep_smoke.py /tmp/gpumd-stage-a-build/nep \
  --output developers/cg_stage_e/ordinary_nep_baseline.json
```

若临时工具环境被清理，使用正常 CMake/CUDA 环境构建。harness 在临时目录训练一代、
导出并核对两个清单，重新 prediction，再检查 MD 与输入保护和 NVE；临时数据自动清理。
使用 `--example-directory <不存在的目录>` 可导出类似 example 的包，拒绝覆盖已有目录。
生成 JSON/log/XML 被忽略，冻结 evidence 和例子输入被跟踪。

## 证据边界与下一入口：阶段 F

仅一张 RTX 3060、两个合成小帧和 50 fs NVE；train/test 同帧用于实现检查，不是泛化
验收。还没有真实 CG 数据、独立 holdout、长时间/NVT/NPT 分布、大体系/多 GPU 性能或
新增跨软件对照。既有 CTest 本次未注册 GROMACS 外部验证，不能算通过。

阶段 D 量化的强 bonded + 小 residual 的 float 精度限制仍存在；没有修改训练标签、
总预测和 loss 的精度。GNEP、bonded 联合拟合和拓扑感知的 residual 描述符仍未支持。
下一阶段先建立更完整的合成/跨软件验证矩阵，再检查目标数据量级下的小 residual
精度；真实数据、长期动力学和性能各自给出独立结论，不能从低训练误差推断物理正确。
