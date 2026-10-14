# Rigid NVE — independent constraint virial and global pressure tests

The tests use a **zero LJ force field**. No physical realism is implied.
The purpose is to check how `rigid_nve` combines the full bead kinetic
stress and the **constraint virial** in `thermo.out`.

## Running

Extract the `constraint_virial/` folder under `GPUMD/tests/gpumd/rigid/`.
Then execute from that directory:

```bash
./constraint_virial/run.sh
```

Or from the `constraint_virial` directory:

```bash
./run.sh spin_only
./run.sh spin_translate
./run.sh asymmetric_rotation
```

Default executable: `GPUMD/build/gpumd`. Override with the environment
variable `GPUMD_BIN=/absolute/path/to/gpumd`.

The script deletes `trajectory.xyz`, `thermo.out` and `gpumd.log` in each
case before running. It does NOT modify your other test folders.

## Cases

- `spin_only`: equilateral triangle, no COM translation, rotation about z.
  Its rotational kinetic tensor must be cancelled by the constraint virial:
  the **full global pressure tensor is zero** (up to integration error).
- `spin_translate`: the same rotor with a nonzero COM velocity. Constraint
  contributions cancel only its internal rotation, leaving exactly
  `M V_COM \otimes V_COM` as the pressure numerator.
- `asymmetric_rotation`: triangle with unequal principal moments and
  nonprincipal rotation. The checker **independently reconstructs** the
  constraint virial using rigid-body kinematics from the trajectory.

## Reference equation

The GPUMD pressure convention is:

`P = [sum_i m_i v_i ⊗ v_i + W_interaction + W_constraint] / V`.

In the present cases, `W_interaction = 0` because epsilon=0.
For an isolated rigid body, with relative bead position `rho_i`, velocity
`v_i` and angular velocity `omega`, one has in the laboratory frame:

```text
I = sum_i m_i (|rho_i|^2 1 - rho_i ⊗ rho_i)
L = sum_i m_i rho_i × (v_i - V_COM)
omega = I^-1 L
alpha = -I^-1 (omega × L)           [zero external torque]
a_i = alpha × rho_i + omega × (omega × rho_i)
W_constraint = sum_i rho_i ⊗ (m_i a_i)
```

The independent checker converts `m_i a_i` and kinetic tensors from
`amu * Å / fs²` to `eV / Å` and `amu * (Å/fs)²` to eV, respectively,
and compares the **full symmetric 3×3 pressure tensor** against `thermo.out`.
GPUMD reports its six components in order `xx yy zz yz xz xy`, in GPa.

It also compares the pressure-inferred constraint virial and checks that
`dump_xyz virial` remains zero (interaction-only), even while the global
pressure contains nonzero constraint virial.

The chosen tolerances are *regression-test tolerances* for these inputs,
not universal accuracy requirements. A failed test should be investigated
rather than being handled by indiscriminately relaxing tolerances.

## Reproducing input coordinates

All static input files are included. `generate_cases.py` can regenerate
them deterministically (only NumPy is required).
