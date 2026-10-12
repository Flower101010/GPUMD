# Variable-member rigid bodies: implementation guide

The initial target is a few hundred molecules mixing one-bead and three-bead
representations. Single-bead molecules are free particles (label -1), while
multi-bead groups form rigid bodies;
the data layout does not impose a three-member limit.
The rotational integrator is Miller's symmetric five-stage
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

RigidBodySystem initialization is implemented on the host, with GPU uploads.
It includes member construction, coherent coordinate preparation, principal
inertia/orientation, fixed member geometry, and Miller momentum initialization.
State accessors, device views and a self-contained host snapshot are implemented
so initialized values can be inspected. Failures clear the system and leave
initialized() false. Successful initialization leaves all body loads zero and
does not modify Atom's positions, velocities or images.

## What remains to implement

1. Variable-member force/torque reduction and bead reconstruction.
2. Runtime center remapping after translational drift.
3. Ensemble_Rigid_NVE's two integration stages and full NVE validation.

These methods remain declarations. Initialization alone does not make rigid-body
MD runnable: initial bead reconstruction/projection, a force evaluation and load
aggregation are needed before the first step. No run.in rigid-ensemble keyword
has been connected yet.

## Upper-layer body labels

Use a dedicated existing XYZ grouping method to identify molecules. All beads
in a molecule share a label; single-bead molecules each have a unique XYZ label,
which the adapter converts to -1. Do not
use species or thermostat-region labels: that would combine many molecules into
one rigid body. For example, a four-bead model with one point molecule and one
three-bead molecule can use:

```text
4
Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3:group:I:1 pbc="T T T"
C 1 2 3 0
C 4 4 4 1
C 5 4 4 1
C 4 5 4 1
```

After the normal XYZ reader has populated Run::group, the upper layer can use:

```cpp
#include "model/rigid_body_input.cuh"
#include "integrate/rigid_body.cuh"

// Inside the future rigid-ensemble setup, with Atom, Box and groups available:
const int molecule_grouping_method = 0; // index of the selected grouping method
auto body_labels =
  rigid_body_labels_from_group(group, molecule_grouping_method, atom.number_of_atoms);
// body_labels is {-1, 1, 1, 1}; the array index is the Atom index.
RigidBodySystem rigid;
rigid.initialize(atom, box, body_labels);
```

The adapter validates the method index, array length and nonnegative XYZ labels,
counts members, and converts singleton groups to -1 in a copied HOST array.
It does not initialize dynamics or add a run.in keyword. In the future ensemble,
keep rigid as a member with simulation lifetime rather than this local example.
The current XYZ reader permits only nonnegative group labels, so this adapter
uses its own -1 labels for singleton groups without modifying Group. For other
unmanaged beads, construct labels explicitly, for example
`std::vector<int> body_labels = {-1, 1, 1, 1, -1};` for a five-bead system.

## Reading the initialization implementation

Read the named methods in this order in src/integrate/rigid_body.cu:

1. initialize_impl validates sizes, labels and the box, then calls create_bodies.
2. create_bodies sorts (label, Atom index) pairs, builds contiguous member ranges
   and the reverse mapping, rejects single-member bodies before GPU allocation,
   then allocates the body/member GPU arrays.
3. initialize_impl prepares coherent coordinates as described below.
4. setup_bodies_static reads current DEVICE masses, computes M and the center
   relative to a reference bead, then builds the symmetric inertia tensor.
   principal_axes performs a scaled 3x3 Jacobi eigendecomposition, sorts moments,
   fixes eigenvector signs and enforces a right-handed basis. axes_to_quaternion
   stores the corresponding unit body-to-space quaternion. Member displacements
   are inverse-rotated relative positions. The center is wrapped into the primary box.
5. setup_bodies_dynamic reads current DEVICE velocities, computes V and
   L=sum(r_i cross m_i*(v_i-V)), then p=2*q*(0,R(q)^T*L).
6. Zero loads are uploaded and initialized becomes true only after all checks.

Static initialization rejects multi-bead geometries if the smallest principal
moment is <= 1e-10 times the largest; coincident bodies are also rejected.
Single-member nonnegative body labels are rejected; use -1 for free particles.
Two point-mass beads necessarily form a linear body and are rejected by the
current inertia check as well.

For a reviewable snapshot after initialize:

```cpp
RigidBodySystem::HostSnapshot snapshot;
rigid.copy_to_host(snapshot);
const auto& body = snapshot.bodies.at(0);
// inspect body.geometry, body.state and the member range in snapshot
```

## Two coordinate input modes

The default three-argument initialize assumes compact molecules. It chooses the
first member (lowest Atom index) as reference, converts each relative position
to fractional coordinates, and folds periodic components into [-1/2,1/2).
The resulting body must have fractional span strictly below 1/2 in every
periodic direction, with a small floating-point margin. It uses full H/H^-1,
including for triclinic boxes. This is a lattice-image convention, not a general
nearest-Euclidean-image search or a connectivity-based unwrap.

Atom.position_image and Atom.unwrapped_position are not read: rigid bodies do
not store or inherit crossing history. Current DEVICE masses, positions and
velocities are used instead of CPU caches. Managed bead data must be finite and
mass must be positive; unmanaged values are not used in physical calculations.

Wrapped coordinates cannot distinguish a truly large molecule from a compact
periodic alias. The default mode therefore requires the caller to satisfy the
compact-molecule assumption even when the span check passes. For general body
geometry, use the explicit overload:

```cpp
// HOST 3N SoA; all members of each body must be in the intended coherent image.
rigid.initialize(atom, box, body_labels, coherent_unwrapped_positions);
```

This mode uses exactly the supplied positions and ignores Atom position/image
buffers. Current DEVICE masses and velocities are still required. Member count
is unrestricted in both modes. Very large double coordinates in explicit input
can already have lost relative precision; initialization cannot restore
information missing from the input.

## State and GPU storage

The manager owns GPU_Vector<BodyGeometry>, GPU_Vector<BodyState>, and
GPU_Vector<BodyLoad>. A Body is a logical aggregate for diagnostic snapshots,
not a fourth persistent array. Each geometry holds first_member and natoms;
these specify a contiguous range in two manager-owned flat arrays:
GPU_Vector<int> member_atoms and GPU_Vector<Vec3> member_displace. Both arrays
have R = sum(natoms) elements, excluding unmanaged beads. Member ranges are
disjoint and together cover [0,R). No per-body allocation or variable-length
array is embedded in a GPU record. Each Atom index appears at most once.

For example, rigid bodies of sizes 3 and 4 have first_member values 0 and 3
and use seven entries in each member array. Free beads are excluded. Body labels
need not be contiguous: initialization compresses labels in ascending order and
sorts members by Atom index. A HostSnapshot copies both body records and member
arrays, so range indices remain interpretable on the host.

The only authoritative rotational state
is normalized quat plus tangent conjugate momentum conjqm. Angular momentum,
angular velocity, and rotation axes are derived locally when needed.

## Periodic center-of-mass coordinates

BodyState stores only the wrapped Cartesian center xcm, its velocity, and the
Miller quaternion/momentum. No body image counters or absolute displacement
history are retained. In periodic directions the fractional center lies in
[0,1); nonperiodic coordinates are not wrapped. Initialization uses coherent
member geometry before wrapping the center. For a length-10 box, members at
9.8 and 10.2 have center 10, stored as xcm=0 with fixed relative offsets -0.2
and +0.2. Directly averaging wrapped positions 9.8 and 0.2 would still be wrong.

After translational drift, remap_centers should compute f=H^-1*xcm, subtract
floor(f) in periodic components, and replace xcm with H*f. This also handles
multiple-box and negative crossings without storing counts. It changes no
velocity, quaternion, momentum or body-relative geometry and may be fused into
the drift kernel. Runtime remapping is still a declaration, not an implemented
kernel.

Rebuild each bead from y_i=xcm+R(q)*s_i, then wrap y_i using the same full box
matrix. Write only managed positions, preserving free beads. Torque arms remain
R(q)*s_i and require no coordinate history. Initialization does not touch Atom's
image buffers; the existing Atom image feature remains available to other code.
The rigid module does not provide coherent continuous trajectories or diffusion
history. When wiring the future rigid ensemble, any unwrapped-output feature
needs an explicit compatible tracking design before it can be used for rigid
members.

This version assumes a fixed box matrix and periodicity. Changing the box needs
an explicit coordinate scaling policy. Nonperiodic coordinates may still grow;
wrapping addresses only periodic travel.

Use one thread per body as the initial execution plan for the 1/3-bead mixture.
Each thread loops over its own member range, gathers forces, and writes its
distinct beads. No atomics are required. The same storage can support a warp or
block per body for substantially longer member lists later. Logical operations
may be fused within an integration-stage kernel. This is a starting design,
not a measured performance claim; use representative end-to-end timing before
changing execution strategies or adopting strict SoA.

## Free particles and supported bodies

Single-bead molecules use label -1 and follow the ordinary particle integrator.
They have no body state, geometry or member displacement in RigidBodySystem.
The upper-layer XYZ adapter maps single-member groups to -1 automatically; the
manager rejects singleton bodies passed directly with nonnegative labels.

The current initializer supports only nonlinear multi-bead bodies, with three
rotational DOF. Linear, near-linear and coincident bodies are rejected by the
principal-inertia check. Two point-mass beads are necessarily linear, so at
least three noncollinear beads are needed in the current version. Member count
alone does not establish that the body is nonlinear.

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
  does not validate multi-bead linear/coincident geometries: test these separately
  or explicitly reject them in the first initialization version.

Before the first step, initialize/project bead states, evaluate bead forces,
and form the initial load. Stage one applies half kicks and full translation/
rotation drift, then remaps centers and reconstructs bead positions and any required half-step
velocities. Evaluate new bead forces; stage two forms the new load, applies the
second half kicks, and reconstructs full-step velocities.

Reconstruction must agree with Force::compute's periodic wrapping. Do not use wrapped bead differences as torque arms. Free beads must
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

The initialization test checks singleton-to-free label conversion, explicit
singleton rejection and boundary-straddling three-bead bodies, known principal moments, unequal masses, triclinic/mixed PBC, rigid
velocity projection, repeated eigenvalues, 24 spatial orientations, explicit
large-body geometry, unmanaged/empty systems, invalid input, independence from
Atom's image buffer and cleanup after failure. It reads back real GPU state and deliberately supplies
incorrect CPU caches. Runtime remapping, bead reconstruction and force reduction
remain unimplemented; their crossing/force-consistency tests and full-system
NVE tests are still required.

Run the implemented subset from the repository root:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target test_quaternion test_quaternion_gpu test_rigid_body_initialization -j4
ctest --test-dir build -R '^unit.test_(quaternion(_gpu)?|rigid_body_initialization)$' --output-on-failure
```
