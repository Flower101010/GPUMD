#!/usr/bin/env bash

set -euo pipefail

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
thermo_file="${script_directory}/thermo.out"
trajectory_file="${script_directory}/movie.xyz"

if [[ ! -s "${thermo_file}" ]]; then
  echo "FAIL: thermo.out is missing or empty." >&2
  exit 1
fi

if [[ ! -s "${trajectory_file}" ]]; then
  echo "FAIL: movie.xyz is missing or empty." >&2
  exit 1
fi

awk '
  function abs(value) { return value < 0.0 ? -value : value }

  $1 !~ /^#/ {
    total_energy = $2 + $3
    if (samples == 0) {
      first_energy = total_energy
      min_energy = total_energy
      max_energy = total_energy
      min_potential = $3
      max_potential = $3
    }
    if (total_energy < min_energy) min_energy = total_energy
    if (total_energy > max_energy) max_energy = total_energy
    if ($3 < min_potential) min_potential = $3
    if ($3 > max_potential) max_potential = $3
    samples++
  }

  END {
    if (samples < 2) {
      print "FAIL: thermo.out has fewer than two data rows." > "/dev/stderr"
      exit 1
    }

    energy_scale = abs(first_energy)
    if (energy_scale < 1.0e-12) energy_scale = 1.0
    relative_range = (max_energy - min_energy) / energy_scale
    potential_range = max_potential - min_potential

    printf("Thermo samples: %d\n", samples)
    printf("Total energy range: %.10e to %.10e eV\n", min_energy, max_energy)
    printf("Relative total-energy range: %.3e\n", relative_range)
    printf("Potential-energy range: %.10e eV\n", potential_range)

    if (relative_range > 1.0e-3) {
      print "FAIL: relative total-energy range exceeds 1e-3." > "/dev/stderr"
      exit 1
    }
    if (potential_range < 1.0e-2) {
      print "FAIL: kinetic and potential energies did not visibly exchange." > "/dev/stderr"
      exit 1
    }
  }
' "${thermo_file}"

awk '
  function distance(ax, ay, az, bx, by, bz) {
    return sqrt((bx-ax)^2 + (by-ay)^2 + (bz-az)^2)
  }

  {
    line_in_frame = (NR - 1) % 5
    if (line_in_frame == 2) {
      x0 = $2; y0 = $3; z0 = $4
    } else if (line_in_frame == 3) {
      x1 = $2; y1 = $3; z1 = $4
    } else if (line_in_frame == 4) {
      x2 = $2; y2 = $3; z2 = $4
      r01 = distance(x0, y0, z0, x1, y1, z1)
      r12 = distance(x1, y1, z1, x2, y2, z2)
      if (frames == 0) {
        min_r01 = max_r01 = r01
        min_r12 = max_r12 = r12
      }
      if (r01 < min_r01) min_r01 = r01
      if (r01 > max_r01) max_r01 = r01
      if (r12 < min_r12) min_r12 = r12
      if (r12 > max_r12) max_r12 = r12
      frames++
    }
  }

  END {
    if (frames < 2) {
      print "FAIL: movie.xyz has fewer than two complete frames." > "/dev/stderr"
      exit 1
    }

    printf("Trajectory frames: %d\n", frames)
    printf("Bond 0-1 range: %.8f to %.8f A\n", min_r01, max_r01)
    printf("Bond 1-2 range: %.8f to %.8f A\n", min_r12, max_r12)

    if ((max_r01 - min_r01) < 1.0e-4 || (max_r12 - min_r12) < 1.0e-4) {
      print "FAIL: one or both bond lengths did not change." > "/dev/stderr"
      exit 1
    }
  }
' "${trajectory_file}"

echo "PASS: harmonic trimer oscillates while conserving total energy within tolerance."
