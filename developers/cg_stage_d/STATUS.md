# 阶段 D：NEP residual prediction / loss / 输出

2026-10-09 完成。接入提交 `743ea5ab`；冻结验收见 `evidence/validation_20261009.json`。
当前分支 `codex/multibead-cg-stage-b`，已具备普通 NEP 的固定 bonded + residual 训练/预测入口。

## 实现与使用

在 `nep.in` 添加：

```text
molecular_force bonded_parameters.in per_frame
```

Parameters 读取一次共享系数，Fitness 给 train/test 读取器显式传递参数；每个 Dataset 保存对应帧的 baseline。NEP 的 generic/specialized/ZBL 共用 forward 结束位置，在新鲜 E/F/virial 预测上加一次 baseline，再进入 loss、report 或 prediction 输出。重复调用由原 ANN/force 内核重新生成 residual，不在旧 total 上累计。

输入与输出含义：

- XYZ energy、virial 是 **总模型的帧总量**；force 是总模型逐 bead 力。不得预先减 bonded。
- Dataset reference 标签保持不变。prediction 中 E/W 仍遵循 GPUMD 原有按 bead 归一化规则，F 不除以 N。
- `nep.txt` 是 residual 模型。本阶段快照没有完整 CG 包；后续阶段 E 已实现伴随清单/参数快照、完整 MD 加载和独立参数/拓扑入口，见 `../cg_stage_e/STATUS.md`。
- 普通 potential NEP 支持本入口；charge/vdw/temperature/dipole/polarizability/atomic tensor 拒绝。GNEP 尚未接入。
- 类型和连接表格式、单位及精度边界详见 `doc/nep/input_parameters/molecular_force.rst`。共享表仍采用 k/2 的 harmonic bond/angle 和 radians。

Potential-model `prediction 1` 现在同时重新计算 train 和存在的 test 数据，覆盖其 E/F/W/stress 文件。原路径只重新计算 train；这次补齐 test 输出（charge/BEC 条件输出也沿用对应模式）。

**阶段 C 的证据修正：** 当时 smoke 复用了训练工作目录，没有先删除旧输出，因此不能证明 test 文件由 prediction 本次重新生成。本阶段已修复该路径，修改 smoke 在预测前删除所有 `.out`，并在全新临时目录逐路径检验六个 E/F/W 输出。阶段 C 冻结记录保留历史快照，不能据此声称当时已实现 test prediction。

## 验收

- gpumd、nep、全部测试构建成功；**18/18 CTest PASS，0 skipped**，包含新 generic 与实际 specialized NEP forward 测试。测试显式检查 specialized library 已启用，fallback 不能当作该路径通过。
- 冻结非零 residual 权重：有/无 baseline 的逐 bead GPU E/F/六分量 W 差符合 `float(residual + baseline)`，容差 `2e-6*(1+|expected|)`；重复 3 次 forward 不累计；4/6 bead 借用 batch 通过。
- 零 residual 直接 forward 的纯 bonded E/F/W loss < `2e-6`。
- 生产二进制冻结模型：清空旧输出后，train/test 的 E/F/W 行数分别 2/10/2、列数 2/6/12；reference 列不变。total - residual - independent baseline 文本最大差 `7.39e-6`，阈值 `2e-5`，主要受原 `%g` 六位有效数字输出限制。
- batch=1/2、stream=0/1 的 frozen output 最大差 0；两种大小和拓扑使用同一参数表。
- 默认自动 scaler 的 CG 一代训练及预测均有限；拒绝重复关键字、缺参/非法模式、所有上述不支持模型和 atomic tensor labels。
- Compute Sanitizer memcheck，**specialized** NEP 新测试：0 errors。
- 不启用功能：原普通 NEP 一代训练与 fresh train/test prediction smoke 通过；不是旧二进制逐位回归。

合成训练数据只有两个小帧，train 与 test 使用同样的帧，用于检验实现和不同读取/输出路径，不是独立泛化测试。已知 residual 由冻结 teacher NEP 生成；学生从 teacher 输出权重的 0.6 倍及 sigma=0.003 开始，描述符参数使用相同起点。三条路径各 SNES 200 代、population=30：

| 检查项 | 初始 test RMSE | 训练后 test RMSE |
| --- | --- | --- |
| E / eV per bead | 1.91831e-3 | 5.58614e-6 |
| F / eV per angstrom | 1.90384e-4 | 1.55284e-5 |
| W / eV per bead | 3.86146e-5 | 6.05831e-6 |

resident generic、stream generic、stream specialized 三条路径本次结果相同。验收要求三类 RMSE 均低于初始的 35%，且低于 `3e-4`；这些是预设的软件检查阈值，不是实际 CG 数据集的物理精度承诺。

纯 bonded 合成数据用零 residual、sigma=1e-4 训练 10 代，F RMSE `4.58621e-7`、W `2.93556e-7`、文本 E RMSE 为 0。此测试证明可维持零 residual 并正确计算 total loss，不证明随机初始化能恢复真实模型。

## 精度与尚未验收的范围

将同一 baseline 放大 1e7、保持小 teacher residual，float total 相减恢复 residual 的最大力误差 **1.13261e-3 eV/angstrom**，可丢失整个小信号。本阶段量化了限制，未改成 double 标签/总量/loss；只用 double 算 baseline 不足以保证强 bonded、小 residual 的训练精度。实际使用前需在目标数据的量级上检查误差。

坐标/盒仍是 float，使用 wrap 后坐标及既有 MD 最小镜像约定。普通 NEP 描述符不读拓扑；同几何/类型但异拓扑的 residual 不可自动区分。

只在 RTX 3060 的 device 0 验收，没有多 GPU 硬件、独立 holdout 构型、真实训练数据、长时间动力学或大体系性能验收。未新增 GROMACS 对照。未接入 GNEP 或 bonded 参数联合训练。

## 重现

工具沿用阶段 A/B 的临时 CMake/Ninja 环境（如被清理，使用正常 CMake/CUDA 环境）。

```bash
export PATH="/tmp/gpumd-stage-a-tools/bin:$PATH"
cmake --build /tmp/gpumd-stage-a-build --parallel 2 --target nep check
python3 developers/cg_stage_d/run_training_validation.py \
  /tmp/gpumd-stage-a-build/nep /tmp/gpumd-stage-a-build/tests/test_bonded_nep
python3 developers/cg_stage_c/run_nep_smoke.py /tmp/gpumd-stage-a-build/nep \
  --output developers/cg_stage_d/ordinary_nep_baseline.json
CUDACXX=/usr/local/cuda/bin/nvcc /usr/local/cuda/bin/compute-sanitizer \
  --tool memcheck --error-exitcode 1 /tmp/gpumd-stage-a-build/tests/test_bonded_nep --compiled
```

新训练 harness 自动导出 teacher/initial/zero 模型、构造合成 total 标签、调用 production NEP、检查输出，并清理临时目录；训练使用固定 restart 和既有 CUDA RNG seed。JSON/log 属于可重生成报告，已忽略；`evidence/` 为冻结摘要。

## 下一入口：阶段 E

1. 将独立 shared parameters + frame/system topology 读取接到 MD；保留既有 combined v1/v2 输入兼容。
2. 实现完整 CG 模型伴随清单，记录 residual NEP、共享参数、bead 类型、单位/势型/版本/图像约定，避免漏加载 bonded。文件路径和一致性检查要明确。
3. 用相同冻结模型和相同 float 可表示构型对照训练预测与 MD 总 E/F/W；区分训练 float 与 MD double 的正常误差。
4. 在至少两种 bead 数/拓扑下完成 combined NEP + bonded 短 MD/NVE；再评估真实数据、小 residual 精度、规模和性能。
