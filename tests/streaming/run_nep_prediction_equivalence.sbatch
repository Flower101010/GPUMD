#!/usr/bin/env bash
#SBATCH --job-name=nep-predict-eq
#SBATCH --partition=gpu4-80GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail
source /usr/share/Modules/init/bash
module purge
module load gcc/11.2.0
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new directory}"
: "${TRAIN_XYZ:?Set TRAIN_XYZ}"
: "${NEP_TXT:?Set NEP_TXT}"
: "${BASELINE_BINARY:?Set BASELINE_BINARY}"
: "${OPTIMIZED_BINARY:?Set OPTIMIZED_BINARY}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite ${OUTPUT_ROOT}" >&2
  exit 2
fi
export LD_LIBRARY_PATH=/home/software/GCC/gcc-11.2.0/lib64:/home/m9n0o/opt/cuda-12.6.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
for variant in baseline optimized; do
  run="${OUTPUT_ROOT}/${variant}"
  mkdir -p "${run}"
  ln -s "${TRAIN_XYZ}" "${run}/train.xyz"
  cp "${NEP_TXT}" "${run}/nep.txt"
  cat > "${run}/nep.in" <<EOF
type 3 Ba Ti O
prediction 1
batch 2
stream_train 1
EOF
  binary="${BASELINE_BINARY}"
  [[ "${variant}" == optimized ]] && binary="${OPTIMIZED_BINARY}"
  (cd "${run}" && "${binary}" > run.log 2> run.err)
done
