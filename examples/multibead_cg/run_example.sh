#!/usr/bin/env bash
# Run from any directory. Keep shipped inputs intact and refuse output overwrite.
set -euo pipefail
if [[ $# -gt 2 ]]; then
  echo "Usage: bash $0 [BUILD_DIRECTORY] [NEW_OUTPUT_DIRECTORY]" >&2
  exit 2
fi
example_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_directory=$(cd "${example_directory}/../.." && pwd)
build_directory=$(realpath "${1:-${repo_directory}/build-cg}")
output_directory=$(realpath -m "${2:-${example_directory}/work}")
for program in nep gpumd; do
  if [[ ! -x "${build_directory}/${program}" ]]; then
    echo "ERROR: missing executable ${build_directory}/${program}" >&2
    exit 2
  fi
done
if [[ -e "${output_directory}" || -L "${output_directory}" ]]; then
  echo "ERROR: output already exists: ${output_directory}. Choose a new output directory." >&2
  exit 2
fi
mkdir -p "$(dirname "${output_directory}")"
mkdir "${output_directory}"
mkdir "${output_directory}/train" "${output_directory}/package"
cp "${example_directory}/train/"* "${output_directory}/train/"
(
  cd "${output_directory}/train"
  echo "Training: 40 generations (software demonstration only)"
  "${build_directory}/nep" > train.log 2>&1
  cp nep.in nep.training.in
  cp nep.predict.in nep.in
  echo "Predicting train and test frames"
  "${build_directory}/nep" > predict.log 2>&1
  cp nep.training.in nep.in
)
for member in nep.txt cg_model.json cg_model.bonded.in; do
  if [[ ! -s "${output_directory}/train/${member}" ]]; then
    echo "ERROR: missing exported model member ${member}" >&2
    exit 1
  fi
  cp "${output_directory}/train/${member}" "${output_directory}/package/"
done
for system in md4 md6; do
  cp -R "${example_directory}/${system}" "${output_directory}/${system}"
  (
    cd "${output_directory}/${system}"
    echo "Running MD: ${system}"
    "${build_directory}/gpumd" > gpumd.log 2>&1
  )
done
echo "Done: ${output_directory}"
echo "Model: package/; prediction: train/*_train.out and *_test.out"
echo "MD: md4/ and md6/ contain thermo.out, movie.xyz and restart.xyz"
