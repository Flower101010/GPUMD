# 阶段 A 结果与下一入口

日期：2026-10-09。输入约定和共享参数的两帧 fixture 已具体化；已建立可运行基线并记录失败。生产实现和既有测试均未修改。

## 当前产物

- [输入草案说明](README.md)：新参数/拓扑文件、训练/MD 接口建议、总量标签、编号与单位约定。
- `input_draft/`：共享参数、4/6 bead 两帧 XYZ、分离拓扑草案、兼容当前 v2 的完整拓扑文件及 model.xyz。
- [host 基线](baseline.json)：本仓库 4 项现有 host 测试、对方 CPU 测试、fixture 检查均 PASS；含源文件 SHA256。
- [CTest 原始结果](gpu_tests.xml)：10 项通过、3 项失败，无跳过。
- [GPU fixture 对照](fixture_md_baseline.json)：两帧均通过，保留日志、输出和 binary SHA256。
- [当前工具链与源版本记录](gpu_baseline.json)：构建参数、失败分类、未运行项目、补充源文件 SHA256。

## 环境与构建

本仓库 HEAD：`bee7f3abeedac1c009f4de1b6a8da099efd6e01c`（master）。
GPU：RTX 3060；驱动 615.71.09；CUDA 13.4.92；GNU 16.2.1；CMake 4.4.4。
Release 构建，CUDA 架构 86，既有 unit tests 保留 assert。

构建目录 `/tmp/gpumd-stage-a-build`，工具环境 `/tmp/gpumd-stage-a-tools`，都是临时目录，不能作为长期交付路径。重现命令见 README 和 `run_gpu_baseline.sh`。

## 测试结果与失败原因

| 检查 | 结果 | 证据范围 |
| --- | --- | --- |
| host 基线 | 6/6 PASS | 4 项现有测试 + 对方 CPU 测试 + fixture |
| gpumd/测试 CUDA 编译链接 | PASS | 未改生产源码，按上述工具链构建 |
| CTest | 10/13 PASS，3 FAIL | 不是全部通过 |
| 三类 bonded GPU 计算测试 | PASS | 现有测试覆盖的非退化几何与 PBC |
| 独立 run.in 集成检查 | PASS | 不使用失败的 test_gpu_vector 作为前置 probe，实际运行 gpumd |
| 两帧 v2 MD E/F/W 对照 | 2/2 PASS | 零速度、1e-8 fs 一步，近静态对照 |
| GROMACS 交叉验证 | NOT_RUN | 当前未找到可执行文件，CTest 未注册该类检查 |
| 新接口训练、prediction/MD 闭环 | NOT_RUN | 草案未实现；没有训练已完成的含义 |

失败 1：`unit.test_gpu_vector` 在 `resize(0)` 后断言 `data()==nullptr`。
失败 2：`unit.test_harmonic_bond_data` 在重新上传空列表后做相同断言。
源码 `GPU_Vector::resize(0)` 明确保留 allocation 以复用容量，因此这是旧测试与当前容器契约不一致。相关历史提交 `a431c7f9`（Merge reusable GPU vector storage）。后续优先保持容量复用，修改测试检查逻辑长度、复用和 clear 释放；不能只把失败断言删除，也不应为了测试把生产容器改回反复分配。

失败 3：`integration.molecular_force_run_in` 使用完整 `test_gpu_vector` 作为 probe；其失败导致脚本提前退出，实际 MD 未开始。独立执行不带 probe 的同一脚本后通过，势能为 1.3121444959 eV。后续应明确区分 GPU 设备可用性探测和完整容器功能测试。

两帧对照比较所有 bead 力、总能量，以及从每 bead 9 分量 virial 求和的总张量，容差 1e-8：

| 帧 | 能量绝对误差 eV | 最大力误差 eV/Å | 最大 virial 误差 eV |
| --- | --- | --- | --- |
| frame4 | 8.88e-16 | 4.22e-15 | 3.33e-15 |
| frame6 | 6.94e-16 | 3.39e-15 | 2.62e-15 |

host fixture 与有限差分最大力/virial 误差分别约 5.48e-10 和 6.56e-10。这些结论只覆盖当前测试几何，尚不能替代退化、三斜盒、长时间或大体系验证。fixture 共享当前 angle/dihedral 公式，有限差分是独立的导数检查；GPU 对照不是独立软件交叉验证。

## 下一步：阶段 B

1. 修正容器测试契约与集成 probe；先恢复完整基线，并保留本次原始失败记录。
2. 建立共用 host/device evaluator，保留内部参数与 GPU SoA 约定。
3. 对零键长、共线与二面角退化返回可定位状态，避免静默漏能量；专门分析 harmonic angle 端点的可定义极限。
4. 补 E/F/W、退化与三斜盒验证，再进行逐帧拓扑和训练 baseline 接入。

本阶段没有实际解析 `gpumd_bonded_parameters` / `gpumd_topology`，也没有在 `nep.in` 实现 `molecular_force`；所有 `.draft` 文件均为建议接口。
