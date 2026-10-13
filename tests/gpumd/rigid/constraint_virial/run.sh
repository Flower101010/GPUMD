#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_GPUMD="$SCRIPT_DIR/../../../../build/gpumd"
GPUMD_BIN="$(realpath "${GPUMD_BIN:-$DEFAULT_GPUMD}")"

if [[ ! -x "$GPUMD_BIN" ]]; then
  echo "GPUMD executable does not exist or cannot execute: $GPUMD_BIN" >&2
  exit 1
fi

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [spin_only|spin_translate|asymmetric_rotation]" >&2
  exit 2
fi

if [[ $# -eq 1 ]]; then
  cases=("$1")
else
  cases=(spin_only spin_translate asymmetric_rotation)
fi

for name in "${cases[@]}"; do
  case "$name" in
    spin_only|spin_translate|asymmetric_rotation) ;;
    *) echo "Unknown case: $name" >&2; exit 2;;
  esac
  directory="$SCRIPT_DIR/$name"
  echo "------------------------------------------------"
  echo "Running constraint virial test: $name"
  echo "GPUMD binary: $GPUMD_BIN"
  [[ -f "$directory/run.in" && -f "$directory/model.xyz" && -f "$directory/zero_lj.txt" ]] || {
    echo "Missing test input files in $directory" >&2; exit 1;
  }
  (
    cd "$directory"
    rm -f trajectory.xyz thermo.out gpumd.log
    "$GPUMD_BIN" 2>&1 | tee gpumd.log
  )
  python3 "$SCRIPT_DIR/check.py" "$directory" "$name"
done

echo "All selected constraint virial tests passed."
