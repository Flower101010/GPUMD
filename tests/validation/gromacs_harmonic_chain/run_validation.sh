#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 GPUMD_BINARY GROMACS_BINARY PYTHON COMMON_DIRECTORY CASE_DIRECTORY COMPARISON_SCRIPT" >&2
  exit 2
fi

gpumd_binary=$(readlink -f "$1")
gromacs_binary=$(readlink -f "$2")
python_binary=$(readlink -f "$3")
common_directory=$(readlink -f "$4")
case_directory=$(readlink -f "$5")
comparison_script=$(readlink -f "$6")

for executable in "${gpumd_binary}" "${gromacs_binary}" "${python_binary}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "ERROR: executable not found: ${executable}" >&2
    exit 2
  fi
done

work_directory=$(mktemp -d -t gpumd-gromacs-chain-XXXXXX)
cleanup()
{
  if [[ "${GPUMD_KEEP_VALIDATION_DIR:-0}" == "1" ]]; then
    echo "Kept validation directory: ${work_directory}"
  else
    rm -rf -- "${work_directory}"
  fi
}
trap cleanup EXIT

cp "${common_directory}"/{model.xyz,zero_lj.txt,run.in,conf.gro,md.mdp} "${work_directory}/"
cp "${case_directory}"/{molecular_force.in,topol.top} "${work_directory}/"

(
  cd "${work_directory}"
  "${gpumd_binary}" > gpumd.log
)

(
  cd "${work_directory}"
  GMX_MAXBACKUP=-1 "${gromacs_binary}" grompp \
    -f md.mdp -c conf.gro -p topol.top -o md.tpr -maxwarn 1 > grompp.log 2>&1
  GMX_MAXBACKUP=-1 "${gromacs_binary}" mdrun \
    -deffnm md -nb cpu -bonded cpu -update cpu -ntomp 1 > mdrun.log 2>&1
  printf 'System\n' | "${gromacs_binary}" traj \
    -f md.trr -s md.tpr -ox gromacs_positions.xvg -xvg none -nocom -nopbc \
    > traj.log 2>&1
  printf 'Potential\nKinetic-En.\nTotal-Energy\n0\n' | "${gromacs_binary}" energy \
    -f md.edr -s md.tpr -o gromacs_energy.xvg -xvg none > energy.log 2>&1
)

comparison_options=()
if grep -Eq '^angles[[:space:]]+[1-9][0-9]*' "${case_directory}/molecular_force.in"; then
  comparison_options+=(--compare-angles)
fi
if grep -Eq '^dihedrals[[:space:]]+[1-9][0-9]*' "${case_directory}/molecular_force.in"; then
  comparison_options+=(--compare-dihedrals)
fi

"${python_binary}" "${comparison_script}" \
  --gpumd-xyz "${work_directory}/movie.xyz" \
  --gromacs-positions "${work_directory}/gromacs_positions.xvg" \
  --gpumd-thermo "${work_directory}/thermo.out" \
  --gromacs-energy "${work_directory}/gromacs_energy.xvg" \
  --case-name "$(basename "${case_directory}")" \
  "${comparison_options[@]}"
