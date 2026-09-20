#!/usr/bin/env bash

set -euo pipefail

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_directory}/../.." && pwd)
gpumd_binary=${1:-"${repo_root}/build/gpumd"}

if [[ ! -x "${gpumd_binary}" ]]; then
  echo "ERROR: GPUMD is not executable: ${gpumd_binary}" >&2
  echo "Build it with: cmake --build build --target gpumd -j2" >&2
  exit 2
fi

gpumd_binary=$(readlink -f "${gpumd_binary}")

cd "${script_directory}"
rm -f -- thermo.out movie.xyz gpumd.log
"${gpumd_binary}" > gpumd.log
./check_results.sh
