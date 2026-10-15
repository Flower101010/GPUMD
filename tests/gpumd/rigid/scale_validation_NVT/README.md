# Mixed rigid NVT — NEP acceptance test

This suite exercises fixed-box Miller rigid dynamics with separate translation
and rotation Nose-Hoover chains following LAMMPS rigid dynamics.
It is a synthetic software/thermostat acceptance fixture,
not a calibrated water/methanol model. All beads have a `C` symbol to match the
bundled `potentials/nep/C_2022_NEP4.txt` potential.

The former common-chain implementation has been replaced. Existing long-run
results belong to the binary that produced them; rerun long validation before
accepting the new two-chain implementation. The command syntax is unchanged.

Only `mixed_2700_300` is supported: **300 three-bead rigid bodies + 2700 free
single beads**, giving 3600 beads, 3000 molecule labels and 9900 thermal DOFs.
The former 3000-rigid-body case has been removed from this NVT suite. The
original sibling `scale_validation/` NVE suite is unchanged.

## Run

From the repository root:

```bash
# Requires a built build/gpumd, Python 3 and NumPy.
# Matplotlib is optional for the English diagnostic figure.
bash tests/gpumd/rigid/scale_validation_NVT/run.sh --smoke

# Long acceptance run; run this separately when ready.
bash tests/gpumd/rigid/scale_validation_NVT/run.sh --long
```

No flag means **smoke**, so a bare invocation never starts a long run.
The presets are:

| Setting | Smoke | Long |
| --- | --- | --- |
| Integration steps, including equilibration | 2000 | 200000 |
| Timestep | 0.05 fs | 0.05 fs |
| Total simulated duration | 0.1 ps | 10 ps |
| Initial / target temperature | 150 / 300 K | 150 / 300 K |
| NHC coupling | 2000 steps = 100 fs | 2000 steps = 100 fs |
| Initial discarded window | 20% = 0.02 ps | 20% = 2 ps |
| Thermo samples / XYZ frames | 200 / 20 | 1000 / 100 |
| Enforce equilibrium temperature criteria | No | Yes |

Initialization samples mass/inertia-dependent Maxwell velocities for free
particles, body translations and body rotations. It removes global COM motion,
then scales the rigid-compatible velocities to exactly the requested initial
kinetic temperature under GPUMD's `3*N_free + 6*N_body` convention. Initial
velocities deliberately start colder than the target so that a disabled
thermostat cannot be hidden by initializing at the target temperature.

Equilibration and production belong to **one continuous `run`**. The checker
discards the first 20% offline; it does not start a new `run`, which would reset
the NHC state. Only a constant target temperature is tested by this suite.

Parameters can be changed explicitly, for example:

```bash
bash tests/gpumd/rigid/scale_validation_NVT/run.sh --long \
  --steps 400000 --temperature 300 --initial-temperature 150 \
  --coupling 2000 --equilibration-fraction 0.25
```

`--dt`, `--gpumd`, `--nep` and `--case mixed_2700_300` are also supported.
`--steps` must be at least 200 and divisible by 200. A step override without
`--long` still performs smoke-only acceptance. Changing `--dt` changes the
physical coupling time as well as the trajectory duration, since coupling is
specified in steps. `--skip-static` skips the static reference and explicitly
marks the reduced acceptance scope in `summary.json`.

Instead of discarding a fraction, `--equilibration-ps <duration>` selects an
absolute warmup duration. It is mutually exclusive with
`--equilibration-fraction`. The cutoff is rounded up to the next integration
step. For example, the following **starts a new 50 ps continuous run**, with
statistics restricted to 20-50 ps:

```bash
bash tests/gpumd/rigid/scale_validation_NVT/run.sh --long \
  --steps 1000000 --equilibration-ps 20
```

Twenty ps is a proposed warmup, not a guarantee of equilibrium. Check the
temperature, kinetic partitions and drift before interpreting that window as
production. The minimum warmup measured in thermostat coupling times is only
a lower-duration guard; it does not prove that the structure or temperature
has equilibrated.

Results live under `results/<profile>/mixed_2700_300/` and are ignored by Git.
A repeated invocation of the same profile replaces its generated outputs.
Copy results elsewhere before rerunning if they need to be retained. Smoke
and long results use separate folders; custom step counts get separate names.

## Acceptance rules

### Every run, including smoke

- Confirm the actual `rigid_nvt` initialization, molecule counts, masses, group
  labels, thermal DOFs, output times and fixed periodic box.
- Check all dumped values for finiteness, rigid distances and relative-velocity
  constraints, and verify nonzero NEP bead forces.
- Independently compute bead KE and `T = 2K/(f*k_B)` and compare with thermo.
  Also verify per-bead potential energy against global potential energy.
- Decompose KE into free-particle, body-COM and body-rotation parts. Compute
  COM velocities with body masses and rotational energy from the residual bead
  velocities, without relying on the integration code's quaternions.
- Check that initial zero total momentum remains near zero. A common NHC
  rescales nonzero total momentum, so arbitrary momentum conservation is **not**
  an NVT requirement. This fixture specifically initializes zero momentum.
- Re-evaluate the first and last dumped configurations in ordinary NVE with
  `time_step 0`. Compare their coordinates and NEP forces/energies. This is an
  execution-path cross-check, not an independent force-gradient verification.

The existing geometry/output thresholds are retained: distance error < 2e-6 A,
velocity-constraint residual < 1e-7 A^2/fs, normalized zero-COM residual < 2e-4,
KE/T consistency < 1e-4, and potential consistency < 5e-5 eV/bead. The KE
conversion uses GPUMD's exact source time-unit constant squared.

### Long only: after the discarded window

The following are explicit **engineering tolerances**, fixed before running the
long test; they are not universal thermostat parameters or a proof of canonical
sampling:

| Criterion | Required value |
| --- | --- |
| Discarded equilibration duration | At least 10 thermostat coupling times |
| Production duration | At least 20 coupling times |
| Production thermo / kinetic-partition samples | At least 100 / 20 |
| Production blocks | At least 10 (20 requested), each spanning at least one coupling time |
| Mean total temperature | Within 5% of target |
| Mean free, body-translation and body-rotation temperatures | Each within 10% of target |
| Temperature drift: first vs last production quarter | Difference below 5% of target |
| Standard error estimated from block means | Below 2% of target |
| Observed temperature variance / canonical reference | Between 0.2 and 5 (broad sanity guard) |

Temperatures of the three kinetic partitions use DOFs `3*N_free`, `3*N_body`
and `3*N_body`, respectively. Checking them separately catches a cold rotational
subsystem hidden behind a correct total temperature. Blocks reduce the effect
of serial correlations, but their standard error is an observed uncertainty
estimate, not a guarantee that all samples are independent. Insufficient duration
or sample count fails long acceptance rather than silently skipping it.

The canonical reference is `Var(T_kin) = 2*T_target^2/f`. The broad ratio guard
rejects temperatures clamped to a fixed value and grossly excessive fluctuations;
it is not a formal distribution test. Correlation, finite sample count,
initialization with zero COM motion and incomplete mixing affect interpretation.
Smoke is too short to establish equilibration, equipartition or canonical
fluctuations. Mean energy and equipartition alone do not establish the correct
kinetic-energy distribution; see [Merz and Shirts (2018)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0202764).

### Removed NVE rules

Physical-system total energy `E = K + U` exchanges with the bath. Its excursion,
endpoint change and time trace are reported but **never judged by an NVE energy
conservation threshold**. A conserved extended-system energy would require
thermostat coordinates and momenta, which are not present in the current output;
this suite does not claim to reconstruct it from bead energies.

See the [LAMMPS rigid thermostat documentation](https://docs.lammps.org/fix_rigid.html)
for the distinction between rigid translational/rotational thermostatting and
bath energy exchange.

## Reports and checker tests

- `manifest.json`: exact settings, ensemble, seed, potential/binary paths and
  SHA256, plus analysis-script hashes.
- `validation.json`: structural checks, kinetic partitions, temperature statistics,
  thresholds and pass/fail; statistical checks are not enforced in smoke.
- `summary.json`: combined structural/statistical/static-reference result and
  explicit acceptance scope (`smoke_only` or `long_nvt`).
- `nvt_trace.csv`, `kinetic_partitions.csv`, `nvt_diagnostics.png` (optional):
  temperature/partition histories and physical energy exchange.
- `performance.json`, `gpumd.log`, `thermo.out`, `trajectory.xyz`: raw runtime evidence.
- `static_reference/{first,last}/`: ordinary-NVE static recalculations and comparison report.

To recheck existing results without rerunning MD:

```bash
python3 tests/gpumd/rigid/scale_validation_NVT/check.py \
  tests/gpumd/rigid/scale_validation_NVT/results/smoke/mixed_2700_300
```

To draw detailed plots and compare later windows **without MD**, use Python
with NumPy and Matplotlib:

```bash
python3 tests/gpumd/rigid/scale_validation_NVT/analyze.py \
  tests/gpumd/rigid/scale_validation_NVT/results/long/mixed_2700_300

# Explicitly reassess 8-10 ps, without changing the original reports.
python3 tests/gpumd/rigid/scale_validation_NVT/analyze.py \
  tests/gpumd/rigid/scale_validation_NVT/results/long/mixed_2700_300 \
  --start-ps 8 --end-ps 10
```

The default start is the original manifest's warmup cutoff. The output goes
to `analysis/window_<start>_<end>ps/`: `diagnostics.png`, `diagnostics.pdf`,
`window_comparison.csv` and `temperature_analysis.json`. The plots include
the full temperature trace, the final 20% in detail, free/translation/rotation
temperatures, physical energy changes, and diagnostic tail-window averages.
Those windows are comparisons, not an automatic search for a passing cutoff.
The original `validation.json`, `summary.json`, trajectory and manifest are
preserved. Successful plot generation does not imply statistical acceptance.

The same command also writes `production_temperature.png` and
`production_temperature.pdf`: the selected production window and a final-window
detail, with a temperature axis fitted to the production data. These plots show
the unsmoothed instantaneous temperature, target and production mean, so the
initial hot transient cannot obscure the fluctuations. The reported temporal
standard deviation describes fluctuations, not the uncertainty of the mean.

For the first 10 ps long run, the original 2-10 ps mean was about 672 K;
8-10 ps averaged about 401 K and the last sample was about 382 K, above the
300 K target. These data are still cooling, so discarding more of this
particular trajectory does not establish equilibrium. Longer sampling is
needed to determine whether it settles at the target; this observation by
itself does not rule out integration or thermostat issues.

To check the analysis logic on the CPU:

```bash
python3 -m unittest discover -s tests/gpumd/rigid/scale_validation_NVT \
  -p test_analysis.py -v
```

These tests include wrong total temperature, clamped temperature, hidden rotational overheating,
unresolved drift, inadequate duration, analytic kinetic-energy decomposition,
and fixture membership/initial temperature/zero momentum.
