#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
gpumd_binary=${1:-"${repo_root}/build/gpumd"}
gpu_probe=${2:-}

if [[ -n "${gpu_probe}" ]]; then
  if [[ ! -x "${gpu_probe}" ]]; then
    echo "ERROR: GPU probe is not executable: ${gpu_probe}" >&2
    exit 2
  fi

  probe_output=$("${gpu_probe}")
  if [[ "${probe_output}" == SKIP:* ]]; then
    echo "SKIP: no accessible GPU for molecular_force run.in integration test."
    exit 0
  fi
fi

if [[ ! -x "${gpumd_binary}" ]]; then
  echo "ERROR: GPUMD is not executable: ${gpumd_binary}" >&2
  echo "Build it with: cmake --build build --target gpumd -j2" >&2
  exit 2
fi

gpumd_binary=$(readlink -f "${gpumd_binary}")
test_directory=$(mktemp -d -t gpumd-molecular-run-in-XXXXXX)

cleanup()
{
  if [[ "${GPUMD_KEEP_TEST_DIR:-0}" == "1" ]]; then
    echo "Kept test directory: ${test_directory}"
  else
    rm -rf -- "${test_directory}"
  fi
}
trap cleanup EXIT

cat > "${test_directory}/model.xyz" <<'EOF'
2
pbc="T T T" Lattice="30 0 0 0 30 0 0 0 30" Properties=species:S:1:pos:R:3:vel:R:3
Ar 5.0 5.0 5.0 0.0 0.0 0.0
Ar 7.0 5.0 5.0 0.0 0.0 0.0
EOF

# Keep the existing LJ neighbor-list sort capacity below CUDA's 1024-thread block limit.
# Epsilon is zero, so this 9 A cutoff still contributes no LJ energy or force.
cat > "${test_directory}/zero_lj.txt" <<'EOF'
lj 1 Ar
0.0 1.0 9.0
EOF

cat > "${test_directory}/molecular_force.in" <<'EOF'
gpumd_molecular_force 1
number_of_atoms 2
harmonic_bond_parameters 1
1.0 2.0
bonds 1
0 1 0
EOF

cat > "${test_directory}/run.in" <<'EOF'
potential zero_lj.txt
molecular_force molecular_force.in
time_step 0.001
ensemble nve
dump_thermo 1
run 1
EOF

(
  cd "${test_directory}"
  "${gpumd_binary}" > gpumd.log
)

if ! grep -q "Initialized molecular force from molecular_force.in" \
  "${test_directory}/gpumd.log"; then
  echo "FAIL: run.in did not initialize the molecular force." >&2
  exit 1
fi

potential_energy=$(awk '$1 !~ /^#/ { value = $3 } END { print value }' \
  "${test_directory}/thermo.out")

if [[ -z "${potential_energy}" ]]; then
  echo "FAIL: thermo.out contains no potential energy value." >&2
  exit 1
fi

if ! awk -v value="${potential_energy}" 'BEGIN { exit !(value > 0.9 && value < 1.1) }'; then
  echo "FAIL: expected approximately 1 eV harmonic energy, got ${potential_energy} eV." >&2
  exit 1
fi

echo "PASS: molecular_force was loaded from run.in."
echo "PASS: stretched harmonic bond produced ${potential_energy} eV potential energy."
