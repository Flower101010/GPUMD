# 阶段 B：计算核心与退化几何处理

日期：2026-10-09。状态：完成。下一阶段是逐帧拓扑与 NEP 训练 baseline 接入；当前仍只有旧 v1/v2 MD 输入可实际加载。

## 改动

- 修正 GPU_Vector 和 bond data 的容量复用测试，明确 `resize(0)` 不释放 allocation、`clear()` 释放，并验证重新上传后的内容。
- 给 GPU 测试程序增加纯设备可用性 `--probe`；集成测试不再把完整容器测试当作设备探测。
- 三类势共用可由普通 C++ 或 CUDA 编译的 evaluator；GPU 数据保留 SoA。公共 interaction 累加统一能量均分、力和 GPUMD 9 分量 virial。
- angle 使用归一化叉积与 atan2；近端点无需从 `1-cos²` 恢复小角度。平衡角允许闭区间 [0,π]，准确共线的平衡端点按零梯度极限处理；非平衡共线报错。
- dihedral 的近退化阈值为归一化叉积平方 <=1e-24，与长度尺度无关。
- 非零强度项的零键长、未定义几何及非有限结果报错，报告类别和 0-based interaction 编号。零强度项直接关闭。
- 三类 kernel 共用错误状态；每次 MolecularForce 调用一次读回检查，缓冲区复用。独立计算器调用各自检查。发生错误后的部分输出不能继续用于积分。
- MolecularForce 初始化只做一次完整参数校验；独立 data upload 仍保留校验，绕过重复校验的入口为私有 friend 方法。

内部势函数、单位、索引及旧 v1/v2 语法保持兼容；参数范围扩展为允许零强度和 0 平衡角。行为上，原来静默丢掉异常 interaction 的路径现在停止运行并报错。

## 验证

固定证据：[validation_20261009.json](evidence/validation_20261009.json)。

| 检查 | 结果 |
| --- | --- |
| 容器测试修正后的原基线 | 13/13 PASS |
| 补全核心后的 CTest | 14/14 PASS，无 SKIP |
| 普通 C++ host 基线与 fixture | 6/6 PASS |
| 两帧旧 v2 MD E/F/W 对照 | 2/2 PASS |
| 50 fs bonded-only NVE，0.05/0.025 fs 两个时间步 | 两帧均 PASS；减半 dt 后最大能量误差约降为 1/4 |
| 新核心测试 Compute Sanitizer memcheck | PASS，0 errors |

新增 GPU 测试逐类和组合检查：坐标有限差分力、9 分量应变导数、正交/三斜 PBC、平移与旋转、净力和净力矩、同四元组多项 dihedral、零长度/NaN/共线错误、关闭项、错误状态重置、角度端点和 1e-14 近共线角。

host 测试还覆盖二面角在 1e-100 和 1e100 长度缩放下的能量与逆尺度力，验证退化判据不依赖长度单位。

NVE 数值结果：

| 帧 | dt/fs | 50 fs 内最大总能量变化/eV | 相对变化 |
| --- | --- | --- | --- |
| frame4 | 0.05 | 1.268e-6 | 1.541e-6 |
| frame4 | 0.025 | 3.169e-7 | 3.853e-7 |
| frame6 | 0.05 | 7.397e-7 | 3.724e-6 |
| frame6 | 0.025 | 1.849e-7 | 9.310e-7 |

这些是小体系软件回归，不是 CG 物理模型已验证的证据。没有运行 GROMACS、长时间稳定性或大体系性能比较。新增的一次同步读回有成本，后续需要在目标体系量化，不据此声称性能提升。

## 重现

环境与临时构建工具同阶段 A。完整构建/CTest 及近静态对照：

```bash
export PATH="/tmp/gpumd-stage-a-tools/bin:$PATH"
CG_CUDA_ARCH=86 bash developers/cg_stage_a/run_gpu_baseline.sh /tmp/gpumd-stage-a-build
python3 developers/cg_stage_b/run_nve_baseline.py /tmp/gpumd-stage-a-build/gpumd
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /tmp/gpumd-stage-a-build/tests/test_bonded_core
```

工具或构建临时目录若已清理，使用已有 CMake/CUDA 环境重新构建即可。常规生成报告已忽略；`evidence/` 是显式冻结的版本记录。

## 提交边界与下一入口

- `7f91c454`：规划、输入草案、阶段 A 固定证据和 .gitignore。
- `f48a7d6a`：容器测试契约和纯设备 probe。
- `7fc93dac`：共用计算核心、异常处理、验证和参数文档。

后续进入阶段 C：实现共享参数/逐帧拓扑读取，并在 Structure/Dataset 中建立 baseline；先验收不同 bead 数、空列表、多类型与多项 dihedral，再接训练 prediction/loss。继续复用这里的 evaluator，避免另写第二套角度与二面角公式。
