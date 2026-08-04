#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
baseline_binary=${1:-$(command -v gpumd || true)}
candidate_binary=${2:-"${repo_root}/build/gpumd"}
absolute_tolerance=${GPUMD_ATOL:-1.0e-8}
relative_tolerance=${GPUMD_RTOL:-1.0e-10}

if [[ -z "${baseline_binary}" || ! -x "${baseline_binary}" ]]; then
  echo "ERROR: baseline GPUMD is not executable: ${baseline_binary:-<not found>}" >&2
  echo "Usage: $0 [baseline-gpumd] [candidate-gpumd]" >&2
  exit 2
fi

if [[ ! -x "${candidate_binary}" ]]; then
  echo "ERROR: candidate GPUMD is not executable: ${candidate_binary}" >&2
  echo "Build it with: cmake --build build --target gpumd -j2" >&2
  exit 2
fi

baseline_binary=$(readlink -f "${baseline_binary}")
candidate_binary=$(readlink -f "${candidate_binary}")
test_directory=$(mktemp -d -t gpumd-molecular-noop-XXXXXX)

cleanup()
{
  if [[ "${GPUMD_KEEP_TEST_DIR:-0}" == "1" ]]; then
    echo "Kept test directory: ${test_directory}"
  else
    rm -rf -- "${test_directory}"
  fi
}
trap cleanup EXIT

echo "Baseline : ${baseline_binary}"
echo "Candidate: ${candidate_binary}"
sha256sum "${baseline_binary}" "${candidate_binary}"

for variant in baseline candidate; do
  mkdir -p "${test_directory}/${variant}"
  cp "${repo_root}/tests/gpumd/carbon/model.xyz" "${test_directory}/${variant}/model.xyz"
  sed -E \
    -e "s|^[[:space:]]*potential[[:space:]].*|potential ${repo_root}/potentials/nep/C_2022_NEP4.txt|" \
    -e "s|^[[:space:]]*velocity[[:space:]].*|velocity 300 seed 42|" \
    "${repo_root}/tests/gpumd/carbon/run.in" > "${test_directory}/${variant}/run.in"
done

(
  cd "${test_directory}/baseline"
  "${baseline_binary}" > gpumd.log
)

(
  cd "${test_directory}/candidate"
  "${candidate_binary}" > gpumd.log
)

baseline_thermo="${test_directory}/baseline/thermo.out"
candidate_thermo="${test_directory}/candidate/thermo.out"

if [[ ! -s "${baseline_thermo}" || ! -s "${candidate_thermo}" ]]; then
  echo "ERROR: one of the GPUMD runs did not produce thermo.out." >&2
  exit 1
fi

if cmp -s "${baseline_thermo}" "${candidate_thermo}"; then
  echo "PASS: baseline and candidate thermo.out files are byte-for-byte identical."
  exit 0
fi

echo "INFO: byte comparison differs; checking numeric data with atol=${absolute_tolerance}, rtol=${relative_tolerance}."

if awk -v atol="${absolute_tolerance}" -v rtol="${relative_tolerance}" '
  function abs(x) { return x < 0 ? -x : x }
  function max(a, b) { return a > b ? a : b }

  FNR == NR {
    if ($1 ~ /^#/) next
    reference_rows++
    reference_columns[reference_rows] = NF
    for (column = 1; column <= NF; ++column) {
      reference[reference_rows, column] = $column
    }
    next
  }

  {
    if ($1 ~ /^#/) next
    candidate_rows++
    if (candidate_rows > reference_rows || NF != reference_columns[candidate_rows]) {
      failed = 1
      next
    }

    for (column = 1; column <= NF; ++column) {
      expected = reference[candidate_rows, column]
      actual = $column
      difference = abs(actual - expected)
      scale = max(abs(actual), abs(expected))
      relative = scale > 0.0 ? difference / scale : 0.0
      if (difference > max_difference) max_difference = difference
      if (relative > max_relative_difference) max_relative_difference = relative
      if (difference > atol + rtol * scale) {
        if (reported < 10) {
          printf("Mismatch row %d column %d: baseline=%.17g candidate=%.17g abs=%.3e rel=%.3e\n",
                 candidate_rows, column, expected, actual, difference, relative) > "/dev/stderr"
          reported++
        }
        failed = 1
      }
    }
  }

  END {
    if (candidate_rows != reference_rows) failed = 1
    printf("Compared %d thermo rows; max abs diff %.3e; max rel diff %.3e.\n",
           candidate_rows, max_difference, max_relative_difference)
    exit failed
  }
' "${baseline_thermo}" "${candidate_thermo}"; then
  echo "PASS: baseline and candidate thermo data agree within tolerance."
else
  echo "FAIL: molecular-force no-op regression changed the carbon result." >&2
  echo "Set GPUMD_KEEP_TEST_DIR=1 to preserve logs and outputs for diagnosis." >&2
  exit 1
fi
