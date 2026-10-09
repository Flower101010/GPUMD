#!/usr/bin/env bash
set -euo pipefail

task_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
task_build_dir=${1:-"${task_repo_root}/build-cg-stage-a"}
task_cuda_compiler=${CUDACXX:-/usr/local/cuda/bin/nvcc}
task_arch=${CG_CUDA_ARCH:-native}

# Put CMake/Ninja on PATH if using the temporary tools environment from this session.
cmake -S "${task_repo_root}" -B "${task_build_dir}" \
  -DCMAKE_CUDA_COMPILER="${task_cuda_compiler}" \
  -DCMAKE_CUDA_ARCHITECTURES="${task_arch}" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build "${task_build_dir}" --parallel 2 --target gpumd \
  test_topology test_force_field_parameters test_read_molecular_force test_gpu_vector \
  test_bonded_geometry test_harmonic_bond_data test_harmonic_bond test_harmonic_angle_data \
  test_harmonic_angle test_periodic_dihedral_data test_periodic_dihedral test_molecular_force \
  test_bonded_core
task_ctest_status=0
ctest --test-dir "${task_build_dir}" --output-on-failure \
  --output-junit "${task_repo_root}/developers/cg_stage_a/gpu_tests.xml" || task_ctest_status=$?
python3 "${task_repo_root}/developers/cg_stage_a/run_fixture_md.py" "${task_build_dir}/gpumd"
exit "${task_ctest_status}"
