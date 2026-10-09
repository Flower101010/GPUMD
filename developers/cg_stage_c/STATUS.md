# 阶段 C：逐帧拓扑与训练 baseline

2026-10-09 完成。实现提交 `c60dc833`；补充验收记录见同目录 `evidence/validation_20261009.json`。当前分支沿用 `codex/multibead-cg-stage-b`。

## 已实现

- `read_bonded_parameters()` 读取 `gpumd_bonded_parameters 1`，三类系数表与 v1/v2 MD 文件共用读取和语义校验。参数文件不包含 atom count 或连接表。单位、势函数和索引沿用阶段 A/B。
- `read_frame_topology()` 读取原始 XYZ header 的版本与三类连接表；缺失、重复、未知 `cg_` 字段、非整数/越界/重复原子/非法类型均拒绝。列表必须使用双引号；空列表为 `"none"`，tuple 内不允许空白。允许同一 quadruple 的多个 dihedral 项。显式 PBC 必须为 T T T；缺失 PBC 沿用普通 NEP 的三维周期假设。
- `Structure` 保存该帧 Topology、double baseline 和存在标志。读取坐标后，用原始盒、阶段 B evaluator 和 MD `apply_mic` 生成逐 bead E/F/W；baseline 与 reference 标签分开存储。
- 数据重排移动完整 Structure，保留新增字段，并保留原来容易在逐字段复制中漏掉的其他元数据。
- `Dataset` 的普通复制及 borrowed/stream 数据路径均使用帧自身 baseline 和正确 prefix offset 打包。写入 float 时才转换；GPU virial baseline 顺序为 xx yy zz xy yz zx。GPU MD 九分量到训练六分量映射为 0/1/2/3/5/7。
- 有 baseline 的 Dataset 不允许混入无 baseline 帧；清除功能后释放对应 GPU 缓冲。不启用时不分配 baseline GPU 数组。
- baseline 分配与 MD 一致：interaction energy/virial 平均分给参与 bead，force 按解析梯度累加。baseline 不按帧 atom count 再除；现有 reference E/W 仍由原加载器按 N 归一化，F 不变。
- 拒绝退化的有效相互作用、非有限坐标、非正/奇异训练晶胞及 float 无法表示的 baseline。k=0 项跳过几何。错误含 XYZ 文件和 header 行号；几何错误另含 interaction family 与 0-based index。

内部入口：

```cpp
auto parameters = read_bonded_parameters("bonded_parameters.in");
read_structures(true, para, train, &parameters);
read_structures(false, para, test, &parameters);
```

**阶段 C 尚未开放 `nep.in molecular_force`。** production 调用暂不传第四个参数；遇到 CG metadata 明确拒绝，避免在 baseline 未加到预测的情况下误训练。阶段 D 将让 Parameters 持有共享参数，并把训练/测试读取接到上述入口，再完成预测/loss 的相加。现阶段不能直接开始 CG residual 训练。

## 验收与证据边界

- gpumd、nep、全部测试构建成功；CTest **16/16 PASS，0 skipped**，其中新增 GPU/host Dataset 测试各一项。
- 严格 parser 的合法/非法输入、4/6 bead、不同分子数和拓扑、两种 bond 类型、同一 quadruple 的两项 dihedral、空 dihedral、整帧空连接表。
- train 与 test 加载；按能量重排；普通 batch；非零起点的 borrowed batch；同一 Dataset 在 4/6/10 bead 间重建及清除 baseline；检查 GPU 缓冲中的实际 E/F/W 与帧一致、reference 标签未减 baseline。
- 将周期盒中的构型平移并 wrap，让相互作用跨边界：两帧 × 正交/三斜盒，与 GPU MolecularForce 的逐 bead E/F/W 对照通过，相对容差 `2e-12 * (1 + |reference|)`。
- 退化 bond/dihedral、disabled dihedral、重复/缺失/非法字段以及零体积晶胞拒绝。
- Compute Sanitizer memcheck：新 Dataset 测试 **0 errors**。
- 普通 NEP 一代训练 + train/test 预测均正常退出，六类 E/F/W 输出有限且行数正确；未启用功能的 CG 输入拒绝。此项是冒烟测试，没有进行旧二进制逐位一致性比较，也不是 CG 训练验证。

环境沿用阶段 B：RTX 3060 单 GPU、CUDA 13.4.92、GCC 16.2.1、Release、CUDA arch 86。设备分配只实测 device 0，没有多 GPU 硬件验收。

精度边界：坐标与盒来自现有 float Structure；这里只保证 evaluator 与累加用 double、最终打包转 float，不能称为全 double 训练。强 bonded + 小 residual 的损失/抵消误差将在阶段 D 单独量化。沿用 MD 最小镜像约定，输入使用原始周期盒中 wrap 后的坐标；没有新增任意多周期 unwrapped 坐标处理。连续链的每段最小镜像不能消除长键/半盒处的图像歧义。

未进行 CG residual 训练、GROMACS 新对照、大体系性能、长时间物理稳定性或多个设备验收。阶段 A `.draft` 中的独立 `gpumd_topology 1` 文件读取及 MD 双文件入口仍属于阶段 E。

## 重现

```bash
# 若该临时工具环境还存在；否则使用自己的 cmake/ninja。
export PATH="/tmp/gpumd-stage-a-tools/bin:$PATH"
cmake --build /tmp/gpumd-stage-a-build --parallel 2 --target nep check
ctest --test-dir /tmp/gpumd-stage-a-build --output-on-failure \
  --output-junit "$PWD/developers/cg_stage_c/gpu_tests.xml"
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /tmp/gpumd-stage-a-build/tests/test_bonded_dataset
python3 developers/cg_stage_c/run_nep_smoke.py /tmp/gpumd-stage-a-build/nep
```

CTest 创建独立临时输入并在成功后删除；未完成测试可能留下 `/tmp/gpumd-bonded-dataset-*` 供诊断。常规 XML/JSON/log 已加入 `.gitignore`；`evidence/` 保存显式冻结的摘要。

## 下一入口：阶段 D

1. 普通 potential NEP 的 Parameters 解析新关键字，加载一次共享参数；train/test 显式传入参数。拒绝 temperature/dipole/polarizability/charge/vdw、atomic virial；保留普通训练默认路径。
2. 检查 generic 与 specialized forward 路径，在每次 residual prediction 结束后统一加 baseline 一次，再进入 loss 或 prediction 输出。各 batch、stream、device 的 baseline 必须随 Dataset；不改 reference 标签。
3. 用冻结 NEP 验证有/无 baseline 的 E/F/W 差恰好等于独立 baseline，包含重排、batch 与 train/test 输出；重复预测不得累计两次。
4. 纯 bonded 与已知 residual + bonded 合成集小规模训练；定义符合 float 精度的阈值，并量化强 bonded、小 residual 的精度问题。不要把普通训练 smoke 当作 CG 训练验收。
5. MD 完整模型加载、伴随模型清单及参数/拓扑分离接口留到阶段 E。
