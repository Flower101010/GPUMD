# Cluster installation

## Build a cluster-compatible package

The GitHub-hosted Ubuntu build is a CI artifact only; it is linked against a newer glibc
than the CentOS 7 cluster provides. Build on the cluster with its native toolchain instead:

    module load gcc/11.2.0
    export CUDA_HOME=/home/m9n0o/opt/cuda-12.6.0
    export GPUMD_CUDA_ARCHITECTURES='70;80'
    scripts/build_cluster.sh
    source build/gpumd-cluster/env.sh

The default package contains both sm_70 (V100) and sm_80 (A100) code in one gpumd
and one nep executable. To produce a smaller single-architecture package, use
GPUMD_CUDA_ARCHITECTURES=70 or GPUMD_CUDA_ARCHITECTURES=80.

The package includes a matching src/ tree so runtime nep_compile 1 can specialize kernels
on either GPU generation.

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
