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
| Total steps | 2000 | 100000 |
| Fixed timestep | 0.05 fs | 2 fs |
| Total duration | 0.1 ps | 200 ps |
| Discarded initial duration | 0.02 ps | 40 ps |
| Initial / target temperature | 150 / 300 K | 300 / 300 K |
| Target pressure | 0.0001 GPa = 1 bar | same |
| Temperature coupling | 2000 steps = 0.1 ps | 50 steps = 0.1 ps |
| Piston coupling | 20000 steps = 1 ps | 500 steps = 1 ps |
| Thermo interval | 10 steps = 0.5 fs | 5 steps = 10 fs |
| XYZ frames | 20 | 100 |
| Enforce production statistics | No | Yes |

### Prepare a gentle initial state

By default, **NPT long starts from the final frame of an accepted 50 ps NVT run**
in `../scale_validation_NVT/results/long/mixed_2700_300/`. Run the sibling NVT
long test first if those results are unavailable. Existing accepted NVT output
can be reused; the NPT runner does not repeat the pre-equilibration simulation.
Smoke still uses the original synthetic lattice to exercise initialization.

The raw lattice was designed to exercise bookkeeping and NEP calls, not to be
near a carbon-NEP energy minimum. It showed large potential-energy release and
heating even at 0.05 fs. Merely lowering initial velocities does not remove that
stored potential energy. Reusing the relaxed positions avoids repeating this
specific transient. No rigid internal coordinates are minimized or changed.

The NPT initial-state builder:

- Requires successful long-NVT statistical/structural validation, matching target
  temperature and potential, and a complete final trajectory frame.
- Copies the final box, coordinates, group labels, masses and rigid-compatible
  velocities; removes residual COM motion and scales all velocities together to
  the requested initial temperature (default: the NPT target, 300 K).
- Checks the fixture's 1.5 A rigid sides and tangent velocities. Records source
  file hashes, source binary hash, frame time and temperature in `manifest.json`.
- Starts **new** thermostat/piston state. This is a new NPT experiment, not an NVT
  checkpoint continuation; NPT warmup/production start at zero in its own clock.

To choose another accepted NVT source explicitly:

```bash
bash tests/gpumd/rigid/scale_validation_NPT/run.sh --long \
  --equilibrated-from /absolute/path/to/accepted_nvt_results \
  --output /absolute/path/to/new_npt_results
```

A missing/rejected source stops before MD with an explanation; it does not
silently fall back to an unrelaxed lattice. `--fresh-lattice` explicitly restores
the original stress fixture (use `--initial-temperature 150` to match the original
cold initialization). It is mutually exclusive with `--equilibrated-from`.
An NVT-relaxed configuration is not necessarily equilibrated at the NPT target
pressure; the NPT warmup is still required, and 2 fs accuracy is still unverified.

The NPT long preset uses 2 fs while preserving the previous 200 ps duration,
40 ps warmup, 0.1 ps temperature coupling, 1 ps piston coupling and 10 fs
thermo sampling. Stability and timestep accuracy at 2 fs remain to be tested.

One continuous run preserves all chains and piston state. Warmup is discarded
only during offline analysis. There is no automatic cutoff search to make a run
pass. Inspect volume/density relaxation as well as temperature and pressure.
A short smoke only checks execution, constraints and output consistency.

For example, choose another duration, cutoff or pressure explicitly:

```bash
bash tests/gpumd/rigid/scale_validation_NPT/run.sh --long \
  --steps 150000 --equilibration-ps 100 \
  --temperature 300 --pressure 0.0001 \
  --coupling 50 --pressure-coupling 500 \
  --output /absolute/path/to/new_npt_results
```

`--steps` must be divisible by 200. `--equilibration-ps` and
`--equilibration-fraction` are mutually exclusive. All couplings are in **steps**;
changing `--dt` changes their physical times. `--thermo-interval` controls the
sampling interval; it must divide both total steps and the XYZ interval
(`steps/100` long, `steps/20` smoke). Thermo sampling does not become coarser
as the run grows; the default long interval is at most 10 fs.
`--gpumd`, `--nep` and `--skip-static` work as in NVT.

Default output: `results/<profile>/mixed_2700_300/` next to this README.
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

All labels are English. The selected production window is a candidate equilibrium
window; the label does not certify equilibration. The instantaneous pressure can
fluctuate strongly or become negative even with positive target pressure.

Replot another window without running MD or changing original acceptance:

```bash
python3 tests/gpumd/rigid/scale_validation_NPT/npt_analysis.py \
  tests/gpumd/rigid/scale_validation_NPT/results/long/mixed_2700_300 \
  --start-ps 100 --end-ps 200
```

Writes figures and `window.json` to `analysis/window_100_200ps/`.
This command only changes the plot window; it does not recompute acceptance.

## Checks and interpretation

Every run checks finite data, fixture membership, DOFs, rigid Cartesian distances
under each frame's periodic box, tangent velocities, zero total momentum,
bead/global KE and temperature consistency, potential consistency, kinetic
partitions, isotropic orthorhombic box shape, and XYZ/thermo lattice agreement.
The rigid distances must **not** expand with the box. First/last static ordinary
NVE evaluations use each selected frame's actual lattice and compare NEP forces
and energies; `--skip-static` explicitly reduces the acceptance scope.

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
  tests/gpumd/rigid/scale_validation_NPT/results/long/mixed_2700_300
```
