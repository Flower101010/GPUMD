#!/usr/bin/env bash
# Production-scale acceptance test for streaming NEP pair cutoffs.
#SBATCH --job-name=nep-cross-accept
#SBATCH --partition=gpu4-80GB
#SBATCH --account=sait
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/m9n0o/GPUMD-cross-cutoff-3865a370-test/acceptance-real-120800/slurm-%j.out
#SBATCH --error=/home/m9n0o/GPUMD-cross-cutoff-3865a370-test/acceptance-real-120800/slurm-%j.err
set -euo pipefail

SOURCE_ROOT=/home/m9n0o/GPUMD-cross-cutoff-3865a370-test
DATA_ROOT=/home/m9n0o/CG-NEP-21comp-20260908/datasets/train-100800-plus-enhanced-20000-v1
RUN_ROOT="$SOURCE_ROOT/acceptance-real-120800"
NEP_BIN="$SOURCE_ROOT/build/a100-v2/nep"
CUDA_ROOT=/home/m9n0o/opt/cuda-12.6.0

module load gcc/11.2.0 >/dev/null 2>&1 || true
export CUDA_HOME="$CUDA_ROOT"
export GPUMD_SRC="$SOURCE_ROOT/src"
export LD_LIBRARY_PATH="$CUDA_ROOT/targets/x86_64-linux/lib:$CUDA_ROOT/lib64:/home/software/GCC/gcc-11.2.0/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$RUN_ROOT"
cd "$RUN_ROOT"
test -s "$DATA_ROOT/DATASET_COMPLETE" || test -e "$DATA_ROOT/DATASET_COMPLETE"
test -s "$DATA_ROOT/train.xyz"
test -s "$DATA_ROOT/test.xyz"
test -x "$NEP_BIN"
ln -sfn "$DATA_ROOT/train.xyz" train.xyz
ln -sfn "$DATA_ROOT/test.xyz" test.xyz

cat > nep.in <<'EOF'
type         2 O C
version      4
cutoff       8 4
cross_cutoff 0 1 6 4
n_max        6 6
basis_size   6 6
l_max        4 1
neuron       30
lambda_e     0
lambda_f     1
lambda_v     0.3
population   50
generation   2
output_interval 1
save_potential 100 0 0
stream_train 1
batch        1000
nep_compile  1
EOF

sha256sum "$NEP_BIN" "$DATA_ROOT/train.xyz" "$DATA_ROOT/test.xyz" nep.in > INPUT_SHA256SUMS
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader > GPU_INFO
"$NEP_BIN" > run.log 2>&1

test -s nep.txt
test -s loss.out
test "$(awk 'END {print $1}' loss.out)" = 2
grep -q '^cross_cutoff 0 1 6 4$' nep.txt
grep -q 'Compile specialized NEP training kernels' run.log
grep -q 'STREAM_MEMORY component=NEP' run.log
! grep -Ein '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' loss.out nep.txt
sha256sum nep.in nep.txt loss.out run.log > OUTPUT_SHA256SUMS
touch ACCEPTANCE_PASSED
