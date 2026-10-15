# Mixed rigid NPT — NEP acceptance test

Tests the current isotropic, orthorhombic, fully periodic `rigid_npt` path:
300 three-bead rigid bodies plus 2700 free particles (3600 beads, 9900 DOFs).
The synthetic fixture definition, streaming trajectory parser, rigid-constraint checks,
kinetic partitions and first/last static NEP comparisons are reused from the
sibling `scale_validation_NVT` suite. Keep the two directories together.
No changes to the production integrator are made by this suite.

This is a synthetic C-labelled software test with the bundled C NEP potential,
not a calibrated mixture with a known density or equation of state. Even a
statistical pass does not establish physical accuracy of a real liquid,
ensemble-distribution correctness, timestep convergence or agreement with LAMMPS.
The duration presets are proposed starting points, not proof of equilibrium.

## Run

From the repository root, with a **freshly built** `build/gpumd`, Python 3,
NumPy and Matplotlib:

```bash
bash tests/gpumd/rigid/scale_validation_NPT/run.sh --smoke
bash tests/gpumd/rigid/scale_validation_NPT/run.sh --long
```

No mode flag means smoke. Use `--long` with **two hyphens**.

| Setting | Smoke | Long |
| --- | --- | --- |
| Total steps | 2000 | 400000 |
| Fixed timestep | 0.05 fs | 0.5 fs |
| Total duration | 0.1 ps | 200 ps |
| Discarded initial duration | 0.02 ps | 40 ps |
| Initial / target temperature | 150 / 300 K | 300 / 300 K |
| Target pressure | 0.0001 GPa = 1 bar | same |
| Temperature coupling | 2000 steps = 0.1 ps | 200 steps = 0.1 ps |
| Piston coupling | 20000 steps = 1 ps | 2000 steps = 1 ps |
| Thermo interval | 10 steps = 0.5 fs | 20 steps = 10 fs |
| XYZ frames | 20 | 100 |
| Enforce production statistics | No | Yes |

### Use the same initial model as the current NVT comparison

By default, **NPT long copies the initial `model.xyz` from the 0.5 fs NVT run**:
`../scale_validation_NVT/results/long_from_npt_steps_100000_dt_0p5/mixed_2700_300/`.
This is that run's **input**, not its final trajectory frame. The current 0.5,
1 and 2 fs NVT comparisons all used byte-identical input models. The current
0.5 fs run passed its checks and reported a production mean of about 299.82 K.

Copying the input preserves the box (about 53.8838 x 71.8450 x 35.9225 A), all
coordinates, masses, group labels and velocities exactly. No coordinate expansion,
velocity resampling or velocity rescaling is performed. The default initial and
target temperatures are both 300 K. `--initial-temperature` must agree with the
copied model; a conflicting value is rejected rather than silently changing the
comparison. The loader checks the NVT manifest, potential, target temperature,
periodicity, rigid geometry, tangent velocities and kinetic temperature.

No preparation simulation is launched. New thermostat and piston states start
at time zero; this is a separate experiment, not checkpoint continuation. The
manifest records the source directory, model/manifest/potential hashes and the
fact that the model was copied byte-for-byte. It does not reinterpret the old
NPT experiment that originally supplied these coordinates as accepted production.

To choose another NVT input explicitly:

```bash
bash tests/gpumd/rigid/scale_validation_NPT/run.sh --long \
  --nvt-initial-source /absolute/path/to/nvt_results \
  --output /absolute/path/to/new_npt_results
```

Missing or inconsistent input stops before MD. `--equilibrated-from` explicitly
selects the older behavior of using the final frame of an accepted NVT run;
`--fresh-lattice` restores the raw fixture. These options and
`--nvt-initial-source` are mutually exclusive. Smoke still uses the original
lattice by default. The raw lattice previously showed large early heating and
should be treated as a separate stress test.

The NPT long preset uses **0.5 fs**, keeping the 200 ps total duration, 40 ps
warmup, 0.1 ps temperature coupling, 1 ps piston coupling and 10 fs thermo output.
This rerun will test whether the smaller-step temperature recovery also holds
in NPT, and whether the volume reaches a stationary production regime.

One continuous run preserves all chains and piston state. Warmup is discarded
only during offline analysis. There is no automatic cutoff search to make a run
pass. Inspect volume/density relaxation as well as temperature and pressure.
A short smoke only checks execution, constraints and output consistency.

For example, choose another duration, cutoff or pressure explicitly:

```bash
bash tests/gpumd/rigid/scale_validation_NPT/run.sh --long \
  --steps 600000 --equilibration-ps 100 \
  --temperature 300 --pressure 0.0001 \
  --coupling 200 --pressure-coupling 2000 \
  --output /absolute/path/to/new_npt_results
```

`--steps` must be divisible by 200. `--equilibration-ps` and
`--equilibration-fraction` are mutually exclusive. All couplings are in **steps**;
changing `--dt` changes their physical times. `--thermo-interval` controls the
sampling interval; it must divide both total steps and the XYZ interval
(`steps/100` long, `steps/20` smoke). Thermo sampling does not become coarser
as the run grows; the default long interval is at most 10 fs.
`--gpumd`, `--nep` and `--skip-static` work as in NVT.

Default output: `results/long_from_nvt_initial_dt_0p5fs/mixed_2700_300/` for
the new NPT long preset, next to this README. Other profiles use
`results/<profile>/mixed_2700_300/`.
Custom step counts have a separate profile. Reusing the same output directory
**replaces generated simulation data and root-level plots**; copy old results or
use a new `--output` directory to retain them. Existing offline `analysis/`
subdirectories are not refreshed by a new simulation and belong to their old run;
use a fresh output directory to avoid confusing those with the new root plots.

## Automatic plots

Figures are generated after analysis even when the statistical criteria fail:

- `npt_diagnostics.png` / `.pdf`: full history and selected production window
  side by side for temperature, hydrostatic pressure, volume and density.
- `production_fluctuations.png` / `.pdf`: **unsmoothed** temperature and pressure
  over the production window, plus a final detail (last quarter, capped at 5 ps).
  These axes use only the displayed window, so early transients cannot hide
  production fluctuations. Target and production mean are marked.
- `volume_history.png` / `.pdf`: a separate volume-versus-time figure with the
  full trajectory, the selected production window and final-window detail.
  All three traces are unsmoothed and have independent vertical scales. The
  initial discarded window is shaded. The selected production mean is marked
  in the first two panels; the final detail uses its own window mean so that
  earlier drift cannot stretch its vertical axis. Neither is a prescribed
  equilibrium volume.

All labels are English. The selected production window is a candidate equilibrium
window; the label does not certify equilibration. The instantaneous pressure can
fluctuate strongly or become negative even with positive target pressure.

Replot another window without running MD or changing original acceptance:

```bash
python3 tests/gpumd/rigid/scale_validation_NPT/npt_analysis.py \
  tests/gpumd/rigid/scale_validation_NPT/results/long_from_nvt_initial_dt_0p5fs/mixed_2700_300 \
  --start-ps 100 --end-ps 200
```

Writes figures and `window.json` to `analysis/window_100_200ps/`.
This command only changes the plot window; it does not recompute acceptance.

## Diagnose temperature bias before accepting production

A preparation sequence alone cannot validate an integrator. NVE is useful for
checking energy conservation on a relaxed configuration; running NVE on the raw
lattice does not remove its excess potential energy or establish the target
temperature. A suitable preparation route is relaxed geometry / small-step NVT,
then NPT equilibration until temperature, volume and structure stop drifting,
then production. Keep the equilibration boundary explicit instead of choosing a
late cutoff solely to get a passing result.

For implementation diagnosis, start independent NVE, NVT and NPT branches from
the **same** relaxed snapshot, with the same velocities and potential. At each
stage compare several timesteps (for example 0.5, 1 and 2 fs), equal physical
durations and equal physical thermostat/piston coupling times. NVE checks energy
drift; NVT isolates the shared thermostat/integration path; NPT adds piston,
box scaling and pressure feedback. A temperature bias shared by NVT/NPT points
to their common path or timestep sensitivity; a bias appearing only in NPT
directs the next audit to the pressure-coupled path. These are diagnostic clues,
not proofs of a particular source-code error.

The present NPT entry point already supports this duration/coupling control:

| Timestep (fs) | Steps for 200 ps | Temperature coupling (steps) | Piston coupling (steps) | Thermo interval (steps) |
| --- | --- | --- | --- | --- |
| 0.5 | 400000 | 200 | 2000 | 20 |
| 1 | 200000 | 100 | 1000 | 10 |
| 2 | 100000 | 50 | 500 | 5 |

Each row has 100 fs temperature coupling, 1000 fs piston coupling and 10 fs
thermo output. Supply `--dt`, `--steps`, `--coupling`, `--pressure-coupling` and
`--thermo-interval` together, and use separate `--output` directories. A short
diagnostic can identify immediate problems, but cannot replace converged NPT
equilibration and production. The current 5% temperature-mean tolerance is a
coarse engineering gate; a passing flag does not establish recovery to 300 K.
Inspect the measured offset and block errors explicitly. The scripts do not
automatically launch this additional comparison matrix.

## Checks and interpretation

Every run checks finite data, fixture membership, DOFs, rigid Cartesian distances
under each frame's periodic box, tangent velocities, zero total momentum,
bead/global KE and temperature consistency, potential consistency, kinetic
partitions, isotropic orthorhombic box shape, and XYZ/thermo lattice agreement.
The rigid distances must **not** expand with the box. First/last static ordinary
NVE evaluations use each selected frame's actual lattice and compare NEP forces
and energies; `--skip-static` explicitly reduces the acceptance scope.

NPT double-precision distance checks now use the stricter limit
`max(1e-11 A, 64 * float64_epsilon * maximum_box_length)` instead of the older
2e-6 A general constraint tolerance. This is an engineering precision guard,
not a rigorous bound on all accumulated arithmetic errors. Actual peak errors
are retained in `validation.json`; the maximum box length includes the initial
and all saved boxes. NVT's existing tolerance is unchanged.

To examine tiny changes without rerunning MD or rewriting acceptance reports:

```bash
python3 tests/gpumd/rigid/scale_validation_NPT/audit_rigidity.py \
  tests/gpumd/rigid/scale_validation_NPT/results/long_from_nvt_initial_dt_0p5fs/mixed_2700_300
```

Add `--allow-partial` to inspect complete saved frames from an ongoing run;
the report then explicitly identifies whether the trajectory is complete. The
tool reads a snapshot copy, compares real Cartesian pair distances with the
initial model using each frame's own minimum-image box, and writes
`analysis/rigidity_audit/rigidity_audit.json` plus `rigidity_trace.csv`.
It reports signed errors, distance ratios, box-length scale and the worst pair,
including a 50-digit recomputation of that pair's distance from the parsed
coordinates. High-precision postprocessing cannot undo errors already present
in the stored coordinates or quaternion propagation. Machine epsilon alone is
not an absolute distance tolerance for subtracting coordinates tens of A large.
Saved frames do not cover every integration substep; the existing C++ mixed-NPT
test also checks geometry after both compute1 and compute2.

Pressure is `(sxx + syy + szz)/3` from `thermo.out`, in **GPa**, using GPUMD's
positive-compression convention. The header must identify the instantaneous
mechanical rigid constraint estimator and separate translation/rotation baths.
Volume is the determinant of the dumped cell. Density uses the actual fixture
masses. This checks the output path; it is not an independent constraint-virial
or force-gradient derivation.

Long temperature checks reuse NVT's engineering criteria, including separate
free, body-translation and body-rotation temperatures. Additional NPT gates are:

| Criterion | Long requirement |
| --- | --- |
| Warmup / production | At least 10 / 100 piston coupling times |
| Production thermo samples | At least 1000 |
| 20 pressure blocks | Each spans at least 5 piston coupling times |
| Mean pressure error | At most 0.01 GPa absolute |
| Pressure block error estimate | At most half the pressure tolerance |
| First vs last production-quarter mean pressure | Difference at most the pressure tolerance |
| First vs last production-quarter mean volume | Difference at most 5% of mean volume |
| Volume block error estimate | At most 2% of mean volume |
| Box response over whole run | Relative volume range greater than 1e-8 |

`--pressure-tolerance` changes the absolute GPa engineering tolerance (recorded
in the manifest). The default 0.01 GPa = 100 bar is a coarse software acceptance
threshold, **not** proof that 1 bar has been resolved. To establish finer pressure
accuracy, preselect a smaller tolerance and collect enough data to resolve it.
No relative pressure error is used near zero pressure.

For pressure and volume, report block means and errors at 4, 8 and 20 blocks;
the gate uses the largest observed error. These estimates do not guarantee
independent samples or a rigorous confidence interval. Slow relaxation can exceed
the piston time; inspect block-size dependence and extend sampling when needed.
The suite does not prescribe an instantaneous pressure variance, a target density
or an NVE energy conservation criterion. Such tests require additional physical
reference information or extended-system state.

`validation.json` records metrics, thresholds and individual checks;
`summary.json` combines these with the static-reference result and distinguishes
`smoke_only` from `long_npt`. `manifest.json` records input settings, binary,
potential and script hashes. `npt_trace.csv` contains time, T, P, V, density,
energies and diagonal stresses; `kinetic_partitions.csv` records partition data.
Logs, raw thermo/XYZ and `performance.json` retain the runtime evidence.

Recheck existing data without simulations (overwrites validation and root plots;
does not rerun the static comparisons or update their combined summary):

```bash
python3 tests/gpumd/rigid/scale_validation_NPT/check_results.py \
  tests/gpumd/rigid/scale_validation_NPT/results/long_from_nvt_initial_dt_0p5fs/mixed_2700_300
```
