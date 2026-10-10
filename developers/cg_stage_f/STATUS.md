# 阶段 F：计算正确性验证

2026-10-10，分支 `codex/multibead-cg-stage-b`。
按用户修订，只验收计算与软件接口；不把 NEP 拟合质量、泛化或物理分布作为通过条件。
冻结结果见 `evidence/validation_20261010.json`；原始再生报告和日志由 .gitignore 排除。

## 环境与复现

GROMACS：`/home/flos/opt/gromacs-2026.3-double/bin/gmx_d`，2026.3，已核实 double。
GPU 为 RTX 3060，CUDA Toolkit 13.4；使用已有 Release、sm86 构建。
Python 依赖由本目录 `pyproject.toml` 和 `uv.lock` 管理，虚拟环境 `.venv` 不提交。

在仓库根目录：

```bash
uv sync --project developers/cg_stage_f --locked
uv run --project developers/cg_stage_f --locked python \
  developers/cg_stage_f/run_independent_validation.py /path/to/build/gpumd \
  --public-coordinates /path/to/ala2_coordinates.npy
uv run --project developers/cg_stage_f --locked python \
  developers/cg_stage_f/run_nep_interface_validation.py \
  /path/to/build/gpumd /path/to/build/nep /path/to/build/tests/test_bonded_nep
cmake -S . -B build \
  -DGPUMD_GROMACS_EXECUTABLE=/home/flos/opt/gromacs-2026.3-double/bin/gmx_d
ctest --test-dir build --output-on-failure
```

以上 CMake 命令需在已完成 CUDA 工具链配置和编译的 build 上执行；首次构建参考阶段 A。
公开坐标本地保存在仓库外 `../cg_validation_data/cgnet-a3e0e8dd/`，不提交大数据。
来源固定为 CGnet commit `a3e0e8ddc06f4b6a9f48f4886b73b4cf372ff481` 的
`examples/data/ala2_coordinates.npy`，SHA-256：
`00f1f6b70fbc9473157511d53a73b6f629d284d3e08e79155b9d2bf546d6dc81`。
不传公开坐标参数时只运行构造几何的矩阵；这不等于复现全部 21 个用例。

## 覆盖与结果

- 22/22 CTest 通过、无跳过，含三个 GROMACS 8 bead NVE 对照，
  dt=0.05 fs、2,000 步：bond、bond+angle、完整 bonded。
  已更新旧 `dump_position` 输入为 `dump_xyz`，并支持查找 double 可执行文件。
- OpenMM 8.6.1 原生 HarmonicBond/Angle/PeriodicTorsion，Reference CPU 后端作为
  独立 E/F 参考；W 使用其标量能量的九分量应变中心差分，两个步长检查收敛。
  不导入 GPUMD 的 bonded 公式或 evaluator 生成参考答案。
- 21 用例各运行 3 次：四种 interaction 分层、32/256/2,048/8,192 bead，
  混合链/支化/含环拓扑、共享参数、多重 torsion、三斜盒跨边界、原子重编号，
  以及 8 个公开 alanine 数值构型；E/F/W 和差分收敛均通过脚本中的预设阈值。
  最大误差分别为 E/N `1.78e-15 eV/bead`、F `1.41e-12 eV/angstrom`、
  W/N `4.53e-10 eV/bead`。这些是本测试矩阵的误差，不是通用精度保证。
- zero 与非零 frozen residual 各检查 resident/stream、batch 2/1，大小为
  32/256/2,048 bead；train/test 全量预测输出和 MD 加法均对照独立 baseline。
  冻结模型不进行优化。预测和 MD 直接比较最大差：E/N `1.21e-6 eV/bead`、
  F `5.34e-6 eV/angstrom`、W/N `9.19e-7 eV/bead`，均通过
  `atol=rtol=1e-5`。容差包含训练 float 与六位有效数字文本输出。
- 专用训练内核由 `unit.test_bonded_nep_compiled` 检查实际启用。
  生产 `prediction 1` 当前会关闭专用内核，不能把预测模式记为专用内核覆盖。
- 无 virial 标签、energy 常数占位 0/1、lambda_e=lambda_v=0 的预测不变性通过。
  此项只检验预测接口，不单独证明优化 fitness 对占位值的处理。
- 8,192 bead 混合拓扑、三斜盒跨边界体系运行一步，Compute Sanitizer memcheck
  报告 `ERROR SUMMARY: 0 errors`。它检查这个 bonded MD 用例，不代表所有路径都做过内存检查。

内存检查输入可用独立脚本的 `--sizes 8192 --export-case /tmp/cg-memcheck`
导出（目标目录必须尚不存在），随后在该目录执行
`compute-sanitizer --tool memcheck --error-exitcode 1 /path/to/build/gpumd`。

## 证据边界

公开坐标数值在两引擎中同样解释为 angstrom，并采用本测试定义的参数。
这仅提供外部几何来源，不声称还原作者 CG Hamiltonian、原始单位或 PMF 力标签。
GROMACS 验证为小体系轨迹对照；大体系 E/F/W 参考来自 OpenMM。
进程耗时含启动开销，不作为性能基准。

保留阶段 D 的已知限制：训练坐标/盒子及预测数组含 float，极强 bonded 与很小 residual
叠加存在精度损失。当前通过不等于任意参数尺度、多 GPU、长期动力学或所有拓扑均已验证。
