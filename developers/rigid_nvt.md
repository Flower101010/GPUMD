# Rigid NVT implementation and acceptance

The fixed-box Miller/Nose-Hoover-chain NVT implementation is connected to the
production input and run loop:

```text
ensemble rigid_nvt T_initial T_final T_coupling grouping_method
```

The temperature/coupling arguments follow `nvt_nhc`; the last argument follows
`rigid_nve`. The public syntax, supported actions, units and restart semantics
are documented in `doc/gpumd/input_parameters/ensemble_standard.rst`.

## Code path

1. `Integrate::parse_ensemble` selects type 2 with a nonnegative rigid grouping
   index. Other ensemble commands reset that index. Existing temperature-ramp
   handling in `Integrate::compute1/compute2` applies unchanged.
2. `Run::perform_a_run` checks the shared rigid-mode restrictions, then
   `Integrate::initialize` constructs and initializes `Ensemble_Rigid_NVT`
   before the first force evaluation. It uses the timestep in effect at `run`,
   so `time_step` may appear before or after `ensemble`.
3. `Ensemble_Rigid_NVT` derives the chain masses from the projected rigid state
   and `f = 3*N_free + 6*N_body`. Each run creates a new chain, following the
   existing ordinary NHC lifecycle. A temperature ramp changes the target but
   leaves those chain masses fixed within that run.
4. `nose_hoover_chain::propagate` advances the CPU chain and returns a common
   momentum scale. `rigid_integrator::scale_momenta` applies it on the GPU to
   free velocities, body COM velocities and Miller conjugate momenta.
5. The two thermostat half-steps surround the physical NVE operations. Bead
   velocities are rebuilt after scaling. Constraint-virial sampling starts
   after the first thermostat and finishes before the second, excluding bath
   impulses. Final thermo is evaluated after the final scale; its temperature,
   kinetic energy and pressure agree with the final bead state.
6. `Dump_Thermo` labels the output `rigid_nvt` and uses the rigid thermal DOFs.

## Automated checks

`test_ensemble_rigid_nvt` checks the GPU momentum scale, splitting against the
physical NVE step, chain evolution, physical-only constraint virial and final
pressure/temperature. `test_rigid_body_operators` separately tests the scaling
invariants and bead reconstruction.

`integration.rigid_nvt_run_in` runs the actual executable in temporary folders.
It covers 17 small input scenarios: consecutive runs with a timestep change,
temperature-ramp equivalence to ordinary NHC for free particles, transitions
between rigid NVT/NVE and ordinary NHC/NVE, real nonzero NEP forces, and rejected
parameter/action combinations. It independently checks rigid distances,
velocity constraints, bead kinetic energy and thermo DOFs. Outputs are reissued
for each run, following GPUMD's measurement lifecycle. The test needs Python 3
but no additional Python packages; it skips when the GPU probe cannot find a GPU.

```bash
cmake --build build --target gpumd test_gpu_vector test_ensemble_rigid_nvt -j4
ctest --test-dir build --output-on-failure \
  -R '^(unit.test_ensemble_rigid_nvt|integration.rigid_nvt_run_in)$'
```

The input-path checks, nine relevant unit tests, and the existing molecular-force
input regression passed. The two final-scale cases `all_rigid_3000` and
`mixed_2700_300` were not run. These checks establish implementation and
execution-path acceptance; they do not establish long-time canonical sampling
or timestep/coupling choices for a production material. The chain and exact body
orientation/momenta are not persisted in bead restart output.
