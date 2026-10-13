# Initially wrapped rigid body under PBC

This is a **numerical correctness test**, not a physical model validation.

It checks the rigid NVE initialization when the three bead coordinates are
initially wrapped across **both X and Y periodic boundaries**. The bead positions
belong to a compact triangle after applying the minimum-image convention.
The triangle translates with constant COM velocity and rotates about its
principal Z axis under zero external force.

- Cell: 20 x 20 x 20 Å, `pbc="T T T"`
- Body: one non-linear 3-bead triangle; single group 0
- Coherent initial COM: `(19.4, 19.5, 10.0)` Å
- COM velocity: `(0.003, 0.002, 0.0002)` Å/fs
- Angular velocity: `(0, 0, 0.0015)` rad/fs
- Timestep 0.5 fs; 1000 steps = 500 fs
- 100 output frames; `precision double`
- Zero-force LJ potential (`epsilon=0`)

## Install and run

Unzip from `GPUMD/tests/gpumd/rigid/`, preserving the directory structure,
then run:

```bash
./run.sh periodic_boundary/initially_wrapped
```

This uses the existing common runner in `tests/gpumd/rigid/run.sh`.
Alternatively run the self-contained script:

```bash
./periodic_boundary/initially_wrapped/run.sh
```

The local script resolves `GPUMD/build/gpumd` and runs `check.py` automatically.
Override the executable with `GPUMD_BIN=/absolute/path/to/gpumd` if needed.

**Warning:** either runner deletes the generated `trajectory.xyz`, `thermo.out`
and `gpumd.log` in this test directory before each run.

## Validation criteria

`check.py` reconstructs the initially coherent triangle from its wrapped
coordinates, compares output coordinates modulo the periodic box against
an exact translated-and-rotated reference, and checks:

- velocity and COM motion;
- all internal distances and velocity constraints;
- lab-frame angular momentum, kinetic energy, temperature with 6 DOFs;
- zero force and potential;
- at least one bead's periodic cell index changes during propagation.

Note that COM and angular momentum must never be computed directly from
wrapped bead coordinates; the check reconstructs coherent images first.
