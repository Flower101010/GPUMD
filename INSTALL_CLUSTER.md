# Cluster installation

Production-tested installation:

```text
/home/m9n0o/opt/gpumd-5.8.1-stream-cross-cutoff-3865a370
```

Add the executables to the environment with:

```bash
module load gcc/11.2.0
export GPUMD_HOME=/home/m9n0o/opt/gpumd-5.8.1-stream-cross-cutoff-3865a370
export CUDA_HOME=/home/m9n0o/opt/cuda-12.6.0
export PATH="$GPUMD_HOME/bin:$PATH"
export GPUMD_SRC="$GPUMD_HOME/src"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:/home/software/GCC/gcc-11.2.0/lib64:${LD_LIBRARY_PATH:-}"
```

Pair-specific cutoff syntax in `nep.in`:

```text
type         2 O C
cutoff       8 4
cross_cutoff 0 1 <radial_cutoff> <angular_cutoff>
stream_train 1
batch        1000
```

Type indices are zero-based and follow the order of the `type` line. Unspecified pairs use the
arithmetic mean of their per-type cutoffs. The production acceptance test used `cross_cutoff 0 1
6 4` only to exercise the feature; choose the scientifically intended O--C values for production.
