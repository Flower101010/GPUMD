#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_GPUMD="$HERE/../../../../../build/gpumd"
GPUMD_BIN="${GPUMD_BIN:-$DEFAULT_GPUMD}"

if [[ ! -x "$GPUMD_BIN" ]]; then
  printf 'GPUMD executable not found or not executable: %s\n' "$GPUMD_BIN" >&2
  exit 1
fi

cd "$HERE"
rm -f trajectory.xyz thermo.out gpumd.log
printf 'Running initially-wrapped rigid-body PBC test\n'
printf 'GPUMD: %s\n' "$(realpath "$GPUMD_BIN")"
"$GPUMD_BIN" 2>&1 | tee gpumd.log
python3 check.py
