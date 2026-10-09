# 多 bead CG 交接：阶段 E 完成

日期：2026-10-09。此文档记录同步前的工作状态，不启动阶段 F 实验。

## 路径与分支

- 工作仓库：`/home/flos/Documents/Project/GPUMDs/GPUMD`
- 对方实现参考目录：`/home/flos/Documents/Project/GPUMDs/GPUMD-5.6_bondangledihedral_need_test`
- 分支：`codex/multibead-cg-stage-b`
- GitHub origin：`git@github.com:Flower101010/GPUMD.git`
- 本次同步推送上述开发分支；主分支为 `master`。
- 当前交付入口：[中文指南](multibead_cg_guide_zh.md)。

## 已完成与提交边界

目标保持：不同 bead 数/分子数/拓扑的训练帧，共享 bead 类型及 bonded 参数；
固定 bonded + 学习 residual NEP，训练标签为总量，每次 MD 使用固定体系拓扑。

| 阶段 | 状态与入口 |
| --- | --- |
| A/B | 输入/单位约定、共用几何 evaluator、退化处理与 GPU 累加；[B 状态](cg_stage_b/STATUS.md) |
| C | 共享参数、逐帧连接表及 Dataset baseline 打包；`c60dc833`、`188cda44` |
| D | total prediction/loss/输出、fresh test prediction、合成训练；`743ea5ab`、`12695f7d` |
| E | 完整模型包、MD split/清单加载、检查与可运行例子；`362a0f8f`、`85e3f40a` |

本次新增统一指南、交接文档及导航，另作 docs 提交。Git commit 历史是版本依据。
原有 `.clangd` 配置提交 `d7ccec25` 保留；本次文档整理没有修改实现或运行新实验。

## 证据与复现环境

- 最终阶段 E：19/19 CTest PASS，0 skipped；两帧训练/MD 总 E/F/W 对齐。
- 完整包与 split/legacy 一致；类型子集、restart、实际复制体系及输入保护通过。
- 两种体系 50 fs NVE，0.05/0.025 fs 步长收敛；两个示例各 1000 步 memcheck 0 errors。
- 普通 NEP 一代训练及 fresh train/test prediction smoke 通过。
- 冻结数值、二进制/源码哈希：[E evidence](cg_stage_e/evidence/validation_20261009.json)。
- 详细边界与命令：[E 状态](cg_stage_e/STATUS.md)、[D 状态](cg_stage_d/STATUS.md)。

本地临时工具：`/tmp/gpumd-stage-a-tools/bin/`；构建：`/tmp/gpumd-stage-a-build/`。
CUDA：`/usr/local/cuda/`；GPU RTX 3060（sm_86）。工具目录可能被清理，不能作为其他
机器的安装依赖。通用构建入口见中文指南，集群安装惯例见仓库 `INSTALL_CLUSTER.md`；
其中历史集群安装不代表已部署这次 CG 改动。

生成日志、JSON/XML、构建产物和示例运行输出已忽略。冻结 evidence、源码、文档和
示例输入被跟踪。模型包含 residual、清单与参数快照三文件，必须整体交付。

## 尚未完成和续接入口

- 强 bonded + 小 residual 的 float 精度限制仍存在；本阶段没有改成 double total/loss。
- 只有单 GPU、两个合成小帧、train/test 同帧和短 NVE，不能声称泛化或物理验收。
- 未新增 GROMACS 对照；真实 CG 数据、长期/NVT/NPT 分布、大体系与多 GPU 尚未验收。
- GNEP、bonded 联合拟合、拓扑感知 residual、improper/约束/排除/1–4 scaling 未实现。
- 删除/单独改名 companion 后无法识别 residual-only 误用。

下一阶段为 F。先读 [总体规划](multibead_cg_merge_plan_20261009_zh.md) 与 E 状态，
建立独立验证矩阵，补合成/跨软件对照并检查目标量级的精度，然后分别开展真实数据、
长期动力学和规模验收。不要仅依赖旧日志或低训练误差判断当前模型有效。

同步后可用 `git fetch origin` 与 `git rev-parse HEAD origin/codex/multibead-cg-stage-b`
核对当前分支是否一致；不能把本交接中的历史通过记录视为以后新改动已通过。

## 2026-10-09 验证规划补充

阶段 F 的来源调查与明日执行方案已写入
[独立参考与公开数据验证计划](cg_stage_f/VALIDATION_PLAN_20261010_zh.md)。
首选 GROMACS 独立 bonded 核对与 CGnet alanine 公开坐标—力数据；该计划尚未执行，
没有新增测试通过结论。明天先完成来源/单位、force-only 及参考单点检查。
