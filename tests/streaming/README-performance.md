# NEP streaming performance checks

`run_nep_performance.sbatch` creates a new run directory, symlinks the input
datasets, samples the Slurm-assigned GPU, and records GPUMD phase timings. It
refuses to overwrite an existing run. Supply paths through environment
variables, for example:

```bash
NEP_BINARY=/path/to/nep \
BENCHMARK_ROOT=/path/to/new-benchmark-root \
TRAIN_XYZ=/path/to/train.xyz \
TEST_XYZ=/path/to/test.xyz \
BATCH_SIZE=1000 GENERATIONS=100 OUTPUT_INTERVAL=100 VARIANT=optimized \
sbatch --export=ALL tests/streaming/run_nep_performance.sbatch
```

Summarize completed runs with:

```bash
python3 tests/streaming/analyze_nep_performance.py \
  /path/to/new-benchmark-root/baseline/batch-1000 \
  /path/to/new-benchmark-root/optimized/batch-1000
```

For deterministic numerical checks, build with `-DNEP_TEST_DIAGNOSTICS=ON`
and a Debug configuration. `run_nep_compile_equivalence.sbatch` compares the
generic and runtime-specialized kernels. `run_nep_prediction_equivalence.sbatch`
evaluates one checkpoint with the baseline and optimized executables.

On 2026-09-09, the 100,800 x 1,000-bead dataset with the production
`cutoff 8 4`, 42-90-1 network, population 50, and batch 1000 gave:

| Metric | bc8024fb | optimized |
|---|---:|---:|
| Total, 100 generations | 578.12 s | 445.00 s |
| GPUMD initialization | 237.19 s | 108.47 s |
| GPUMD training section | 339.36 s | 334.52 s |
| Optimized core training | - | 3.194 s/generation |
| Optimized core throughput | - | 312.5 configurations/s |
| GPU peak | 10,131 MiB | 10,131 MiB |
| Active-sample GPU utilization | 97.9% | 98.1% |

The end-to-end improvement is 1.30x and initialization is 54.3% shorter.
Steady-state improvement is small because Dataset loading is about 2% of a
generation and population fitness computation already keeps the A100 busy.
