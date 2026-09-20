#!/usr/bin/env bash
#SBATCH --job-name=nep-stream-perf
#SBATCH --partition=gpu4-80GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

source /usr/share/Modules/init/bash
module purge
module load gcc/11.2.0

: "${NEP_BINARY:?Set NEP_BINARY to the executable under test}"
: "${BENCHMARK_ROOT:?Set BENCHMARK_ROOT to a new benchmark root}"
: "${TRAIN_XYZ:?Set TRAIN_XYZ to train.xyz}"
: "${TEST_XYZ:?Set TEST_XYZ to test.xyz}"

BATCHES=(1000 4000 10000)
batch="${BATCH_SIZE:-${BATCHES[${SLURM_ARRAY_TASK_ID:-0}]}}"
generations="${GENERATIONS:-1000}"
output_interval="${OUTPUT_INTERVAL:-100}"
variant="${VARIANT:-optimized}"
run_dir="${BENCHMARK_ROOT}/${variant}/batch-${batch}"

if [[ -e "${run_dir}" ]]; then
  echo "Refusing to overwrite existing benchmark directory: ${run_dir}" >&2
  exit 2
fi
mkdir -p "${run_dir}"
ln -s "${TRAIN_XYZ}" "${run_dir}/train.xyz"
ln -s "${TEST_XYZ}" "${run_dir}/test.xyz"

cat >"${run_dir}/nep.in" <<EOF
type 2 O C
version 4
cutoff 8 4
n_max 6 6
basis_size 6 6
l_max 4 1
neuron 90
lambda_e 0
lambda_f 1
lambda_v 0.3
population 50
generation ${generations}
output_interval ${output_interval}
save_potential 50000 0 1
stream_train 1
batch ${batch}
EOF

export LD_LIBRARY_PATH=/home/software/GCC/gcc-11.2.0/lib64:/home/m9n0o/opt/cuda-12.6.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export GPUMD_PERF_TIMING=1

cd "${run_dir}"
{
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "variant=${variant}"
  echo "batch=${batch}"
  echo "generations=${generations}"
  echo "binary=${NEP_BINARY}"
  echo "start_epoch=$(date +%s)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi -L
} > metadata.txt

(
  while :; do
    date +%s
    nvidia-smi \
      --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used \
      --format=csv,noheader,nounits
    sleep 2
  done
) > gpu-samples.csv 2>&1 &
sampler_pid=$!
(
  previous_lines=0
  while :; do
    current_lines=0
    if [[ -f loss.out ]]; then
      current_lines=$(wc -l < loss.out)
    fi
    if (( current_lines != previous_lines )); then
      printf '%s %d\n' "$(date +%s)" "${current_lines}"
      previous_lines=${current_lines}
    fi
    sleep 1
  done
) > loss-progress.txt 2>&1 &
progress_pid=$!
cleanup() {
  kill "${sampler_pid}" 2>/dev/null || true
  kill "${progress_pid}" 2>/dev/null || true
  wait "${sampler_pid}" 2>/dev/null || true
  wait "${progress_pid}" 2>/dev/null || true
}
trap cleanup EXIT

set +e
/usr/bin/time -f 'elapsed_s=%e\nmax_rss_kb=%M\nexit_status=%x' \
  -o time.txt "${NEP_BINARY}" > run.log 2> run.err
status=$?
set -e

cleanup
trap - EXIT
echo "end_epoch=$(date +%s)" >> metadata.txt
echo "exit_status=${status}" >> metadata.txt
if compgen -G '*_train.out' >/dev/null; then
  stat --printf='%n %s bytes\n' *_train.out > train-output-sizes.txt
  if [[ "${CLEAN_LARGE_OUTPUTS:-1}" == 1 ]]; then
    rm -f *_train.out
  fi
fi
exit "${status}"
