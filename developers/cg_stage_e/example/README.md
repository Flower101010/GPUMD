# 可运行的 CG 模型包示例

这是两帧合成数据的一代训练输出，用于软件验收，不是可用于实际材料的 CG 力场。
C/O 是共享 bead 类型的占位标签，质量在各自 model.xyz 中显式给出。

- `cg_model.json`、`nep.txt`、`cg_model.bonded.in` 必须一起保存和复制。
- `frame4`：4 bead 链，含 bond、angle 和两项 proper dihedral。
- `frame6`：两个 3 bead 片段，含 bond 和 angle；共享同一参数表与残差 NEP。
- 每个目录的 `topology.in` 固定连接关系；`model.xyz` 给出盒、类型、质量与零初速度。

在对应目录运行新的 GPUMD 二进制，例如：

```bash
cd developers/cg_stage_e/example/frame4
/tmp/gpumd-stage-a-build/gpumd > gpumd.log 2>&1
```

换到 `frame6` 可运行第二个体系。输入运行 1000 步、时间步 0.05 fs，即 50 fs NVE；
`thermo.out` 和日志已加入 `.gitignore`。上述 `/tmp` 二进制仅存在于本次本地验收环境，
也可替换为正常构建得到的绝对路径。

NEP 是 residual，完整势为固定 bonded + NEP；在 run.in 使用：

```text
cg_model ../cg_model.json topology.in
```

不要单独复制/加载 residual。SHA-256 会拒绝被修改的模型或参数；要部署重新训练的模型，
应使用该次训练导出的完整包。示例生成与验收命令见上一级 `STATUS.md`。
