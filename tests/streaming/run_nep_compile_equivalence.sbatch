#!/usr/bin/env bash
#SBATCH --job-name=nep-compile-eq
#SBATCH --partition=gpu4-80GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail
source /usr/share/Modules/init/bash
module purge
module load gcc/11.2.0

: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new directory}"
: "${NEP_BINARY:?Set NEP_BINARY to a deterministic DEBUG build}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite ${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
python3 "${SOURCE_ROOT}/tests/streaming/make_case.py" \
  "${OUTPUT_ROOT}/train.xyz" --frames 11 --vary-size

export LD_LIBRARY_PATH=/home/software/GCC/gcc-11.2.0/lib64:/home/m9n0o/opt/cuda-12.6.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
for compile_mode in off on; do
  run="${OUTPUT_ROOT}/${compile_mode}"
  mkdir "${run}"
  ln -s "${OUTPUT_ROOT}/train.xyz" "${run}/train.xyz"
  sed "s/nep_compile 0/nep_compile ${compile_mode}/" \
    "${SOURCE_ROOT}/tests/streaming/nep_stream1.in" > "${run}/nep.in"
  (cd "${run}" && "${NEP_BINARY}" > run.log 2> run.err)
done
