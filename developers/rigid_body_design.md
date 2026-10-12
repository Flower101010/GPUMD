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

Vec3 algebra, vector/quaternion norms and checked normalization, vector rotation
and inverse rotation, momentum/omega conversions, and the fixed-orientation
Miller torque kick are also implemented in the same header. Normalization uses
component scaling so finite very large or subnormal inputs can be normalized;
zero and non-finite inputs fail without changing the input. Rotation functions
require a unit quaternion and do not normalize it implicitly.

## What you must implement

The math interfaces are implemented. The rigid-body management and simulation
integration below remain your implementation work.

Suggested order:

1. RigidBodySystem member construction, static geometry, and dynamic setup.
2. Three-bead force/torque reduction and bead reconstruction.
3. Ensemble_Rigid_NVE's two integration stages and full NVE validation.

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

CTest runs host and GPU checks for the implemented math, including extreme
normalization inputs, known rotations, nontrivial frame/momentum conventions,
zero-inertia omega, and torque kicks. The host test also checks a complete
kick-drift-kick sequence against the constant axial torque analytic solution.
The GPU test calls the functions from a different translation unit without
device separate compilation. RigidBodySystem dynamics and full-system NVE are
not implemented or validated yet; the math tests do not establish linear-body
integration correctness.

Run the implemented subset from the repository root:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target test_quaternion test_quaternion_gpu -j4
ctest --test-dir build -R '^unit.test_quaternion(_gpu)?$' --output-on-failure
```
