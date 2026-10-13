# GPUMD rigid NVE — final-scale NEP verification

This is a **numerical/software acceptance test**, not a physically calibrated water/benzene coarse-grained simulation. All beads use a `C` type to match GPUMD's bundled single-species NEP4. Free water molecules are represented by **one free bead each**, with mass 18 amu, **not three rigid atoms**. The three-bead triangular bodies use 12 amu per bead. There is no thermostat, equilibration stage, training, or temperature rescaling.

## Prerequisites

- Run from an existing GPUMD source tree; expected executable: `build/gpumd`.
- Expected standard model: `potentials/nep/C_2022_NEP4.txt` (must start `nep4 1 C`).
- Python 3 + NumPy.
- GPUMD build must include your `ensemble rigid_nve 0` implementation, compatible with global temperature/KE using rigid DOF.

## Test matrix

| Case | 3-bead rigid bodies | Singleton free beads | Total beads | Thermal DOFs | Groups |
|---|---:|---:|---:|---:|---:|
| `all_rigid_3000` | 3000 | 0 | 9000 | 18000 | 3000 |
| `mixed_2700_300` | 300 | 2700 | 3600 | 9900 | 3000 |

Both models put 3000 sites on a 15×20×10 orthorhombic lattice (spacing 4.3 Å, small jitter) and assign distinct random orientations/rigid angular velocities. Each rigid body is a noncollinear equilateral triangle (side 1.5 Å); the free-bead masses differ from the rigid-bead masses. The random seed is fixed (`20261013`), and global translational momentum is removed. All initial velocities obey rigid constraints. A periodic environment is used to exercise PBC at scale.

## Run

Extract this folder into `tests/gpumd/rigid/`, then from that directory:

```bash
./scale_validation/run.sh --smoke     # 2,000 steps at 0.05 fs = 0.1 ps, per case
./scale_validation/run.sh --long      # 200,000 steps at 0.05 fs = 10 ps, per case
```

To extend to 50 ps while retaining 0.05 fs:

```bash
./scale_validation/run.sh --steps 1000000
```

For only one case:

```bash
./scale_validation/run.sh --long --case mixed_2700_300
```

To point at a specific executable:

```bash
./scale_validation/run.sh --smoke --gpumd /path/to/GPUMD/build/gpumd
```

`--dt 0.025` and `--nep /path/to/your/C-only/NEP4.txt` are supported. Custom steps must be divisible by 200. When the timestep changes, the simulation duration changes accordingly. All generated files are under `scale_validation/results/<profile>/<case>/`. Existing results **in the same profile and case** will be cleared/replaced on rerun; results from other profiles remain intact. Generated results are excluded from Git.

The `--smoke` run checks basic correctness and NEP static references, but only **reports** energy drift (too short to judge secular drift). The `--long` run also enforces configurable energy thresholds. The script records wall time in `performance.json` and the simulation settings in `manifest.json`. Every test also writes `energy_trace.csv` and, when Matplotlib is available, `energy_drift.png` for direct inspection of long-time trends.

## Independent checks

- Confirm exact body/singleton counts, grouping, masses and temperature DOFs.
- Check each triangle's minimum-image distances and velocity constraints.
- Verify total linear momentum under PBC; **do not** use naive wrapped-coordinate global angular momentum as a conservation test under periodic boundaries.
- Cross-check bead kinetic energy, per-bead NEP potential-energy sum, and temperature against `thermo.out` (including correct rigid temperature DOFs).
- Reject NaN/Inf stresses/energies and unexpectedly zero NEP forces.
- Quantify maximum energy excursion and the least-squares secular energy slope across the run. Both are recorded *per bead* so results are comparable between 9000- and 3600-bead systems.
- Recompute NEP per-bead **forces and energies** for first and last trajectory frames using an independent, ordinary-NVE `time_step 0; run 1` GPUMD invocation. Results saved to `static_reference/`. This is an **execution-path cross-check**, not an independent force-gradient finite-difference proof.

Thermo and XYZ output are intentionally sparse: 200 thermo rows and 20 XYZ frames, regardless of `--steps`. Long runs avoid writing thousands of large snapshots. Outputs are appended by GPUMD, so the harness deletes prior output files before each run.

## Acceptance interpretation

Default numerical thresholds are listed in `check.py`, including:

- Maximum rigid pair distance error: `2e-6 Å`.
- Relative total linear momentum deviation: `2e-4` (relative to initial sum-of-mass-weighted RMS speed, not `P0`, which is near zero).
- Max total energy excursion: `0.002 eV / bead`.
- Absolute fitted energy slope: `0.0002 eV / bead / ps`.

Energy thresholds are **test-specific numerical regression criteria**, not physical thermodynamic standards or mathematical proof of stability. If the long-run energy criterion fails, first repeat at half timestep and identical total simulated time to check convergence; do not loosen the threshold blindly. You may override the energy limits with environment variables `MAX_E_EXCURSION_EV_BEAD` and `MAX_E_SLOPE_EV_BEAD_PS` after documenting your rationale.

The 10 ps default is the first final-scale validation. For long-time drift, extend to 50 or 100 ps on your GPU and inspect energy vs. time, not just the final-minus-initial difference. Because no equilibration is requested, the **kinetic and potential energies are free to vary**; only their sum should stay bounded without secular drift. No production-physics conclusions can be drawn from these artificial models.

## Files

- `generate_cases.py`: deterministic model generator
- `scale_common.py`: extxyz I/O, physical conversion constants
- `check.py`: streaming trajectory checker and report writer
- `static_compare.py`: static ordinary-NVE NEP cross-check
- `run_tests.py`: generator/simulation/validation orchestrator
- `run.sh`: command-line entry point
- `results/**/manifest.json`, `validation.json`, `performance.json`: machine-readable reports

## Fix for `numpy.bool_` serialization in `validation.json`

The report writer converts all NumPy comparison results to built-in Python
`bool` values before calling `json.dumps`. This prevents the error
`TypeError: Object of type bool is not JSON serializable`.

If only the old validator failed *after* a successful simulation, you do not
need to repeat GPU dynamics. From the `rigid` directory, after updating the
package, run:

```bash
python3 scale_validation/check.py scale_validation/results/smoke/all_rigid_3000 --dt 0.05 --steps 2000 --thermo-interval 10 --xyz-interval 100
python3 scale_validation/check.py scale_validation/results/smoke/mixed_2700_300 --dt 0.05 --steps 2000 --thermo-interval 10 --xyz-interval 100
```

Then rerun static NEP reference checks without repeating the dynamics:

```bash
cd scale_validation
python3 - <<'PY_STATIC'
from pathlib import Path
from static_compare import check_frames
from run_tests import ROOT
exe = ROOT / 'build/gpumd'
for case in ('all_rigid_3000', 'mixed_2700_300'):
    if not check_frames(Path('results/smoke') / case, exe):
        raise SystemExit(f'Static NEP reference failed: {case}')
print('Static reference checks passed for both cases.')
PY_STATIC
```

For another custom run, replace the folder and use its settings from
`manifest.json`.
