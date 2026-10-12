# Fixed-three-bead rigid bodies: implementation guide

The initial target is a few hundred rigid bodies with exactly three point-mass
beads per body. The rotational integrator is Miller's symmetric five-stage
splitting, not a general interchangeable-integrator abstraction.

## What is implemented

The existing quaternion multiplication, conjugation, dot product, Miller
permutation, single-axis flow, and five-stage free rotational drift now live in
`src/utilities/math.cuh` as host/device inline functions. These are the original
formulas, with typed principal-axis arguments and debug precondition checks.
They can be called from another CUDA translation unit without device separate
compilation, and from ordinary C++ host tests.

The full drift is named `integrate_free_rotation`: it contains no external
torque kick. `math.cu` remains an include-only translation unit.

## What you must implement

All declarations marked TODO in `math.cuh` have no function bodies. Define them
in that header with the same host/device inline qualifier. Do not put their
bodies only in `math.cu`, and do not call them until they are implemented.

Suggested order:

1. Vec3 algebra, norms, and checked normalization.
2. Quaternion vector rotation and inverse rotation. Check known 90-degree
   rotations, round trips, and equivalence of q and -q.
3. Angular momentum/conjugate momentum/omega conversions. Check coordinate
   frames, the factor of two, and zero-inertia axes.
4. Fixed-orientation torque kick. Check that zero torque changes nothing,
   space angular momentum changes by dt * torque, and q dot p stays zero.
5. RigidBodySystem member construction, static geometry, and dynamic setup.
6. Three-bead force/torque reduction and bead reconstruction.
7. Ensemble_Rigid_NVE's two integration stages and full NVE validation.

RigidBodySystem has declarations only, including its constructor/destructor;
it cannot yet be instantiated and linked. No placeholder function returns a
fake result. Eigenvector construction, kernels, and time integration remain
your implementation work.

## State and GPU storage

The manager owns GPU_Vector<BodyGeometry>, GPU_Vector<BodyState>, and
GPU_Vector<BodyLoad>. A Body is a logical aggregate for diagnostic snapshots,
not a fourth persistent array. Each geometry holds three bead indices and
three fixed body-frame displacements. The only authoritative rotational state
is normalized quat plus tangent conjugate momentum conjqm. Angular momentum,
angular velocity, and rotation axes are derived locally when needed.

Use one thread per body as the initial execution plan. Each thread gathers
three bead forces and writes its three distinct beads. No atomics or variable
member-list reduction are needed. Logical operations may be fused within an
integration-stage kernel. This is a starting design, not a measured performance
claim; use representative end-to-end timing before changing to strict SoA.

## Miller conventions

- Quaternion order is (w,x,y,z), with Hamilton multiplication.
- q rotates body coordinates to space coordinates.
- L_body = R(q)^T L_space and p = 2 q (0,L_body).
- q is unit length and q dot p = 0.
- During a torque kick, q is fixed and p changes by
  2 dt q (0,R(q)^T torque_space). Each half kick receives dt/2.
- Free drift uses axes 3,2,1,2,3 with durations dt/2,dt/2,dt,dt/2,dt/2.
- Time is GPUMD's internal time, not an unconverted value in femtoseconds.
- Zero-inertia axes are skipped by the existing free-flow function. That alone
  does not validate a linear/point-like body integrator: test degenerate cases
  separately, or explicitly reject them in the first initialization version.

Before the first step, initialize/project bead states, evaluate bead forces,
and form the initial load. Stage one applies half kicks and full translation/
rotation drift, then reconstructs bead positions and any required half-step
velocities. Evaluate new bead forces; stage two forms the new load, applies the
second half kicks, and reconstructs full-step velocities.

Reconstruction must agree with Force::compute's periodic wrapping/image
updates. Do not use wrapped bead differences as torque arms. Free beads must
remain outside rigid updates. External velocity corrections must act on whole
body state. Thermodynamic DOF and constraint virial require separate treatment.

## Current validation boundary

CTest runs the existing host quaternion checks and a GPU test exercising the
already implemented math directly from a different translation unit. Future
TODO functions and RigidBodySystem dynamics are not tested or implemented yet.

Run the implemented subset from the repository root:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target test_quaternion test_quaternion_gpu -j4
ctest --test-dir build -R '^unit.test_quaternion(_gpu)?$' --output-on-failure
```
