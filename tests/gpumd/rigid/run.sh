#!/usr/bin/env bash
set -euo pipefail

# Resolve paths independently of the calling directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUMD_BIN="${GPUMD_BIN:-"$SCRIPT_DIR/../../../build/gpumd"}"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <test_directory>" >&2
  exit 2
fi

TEST_DIR="$SCRIPT_DIR/$1"

if [[ ! -d "$TEST_DIR" ]]; then
  echo "Test directory not found: $TEST_DIR" >&2
  exit 1
fi

if [[ ! -x "$GPUMD_BIN" ]]; then
  echo "GPUMD executable not found: $GPUMD_BIN" >&2
  exit 1
fi

cd "$TEST_DIR"

echo "Test: $(basename "$TEST_DIR")"

# Remove generated data from previous runs.
rm -f trajectory.xyz thermo.out gpumd.log

echo "Running GPUMD..."
"$GPUMD_BIN" 2>&1 | tee gpumd.log

echo "Checking results..."
python3 check.py
