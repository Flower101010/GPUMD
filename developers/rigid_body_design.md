# Variable-member rigid bodies: implementation guide

The initial target is a few hundred molecules mixing one-bead and three-bead
representations. Each body can contain any positive number of point-mass beads;
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
in a molecule share a label; each single-bead molecule has its own label. Do not
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
// body_labels is {0, 1, 1, 1}; the array index is the Atom index.
RigidBodySystem rigid;
rigid.initialize(atom, box, body_labels);
```

The adapter copies HOST labels and validates the method index and array length.
It does not initialize dynamics or add a run.in keyword. In the future ensemble,
keep rigid as a member with simulation lifetime rather than this local example.
The current XYZ reader permits only nonnegative group labels, so this adapter
manages every bead. For unmanaged beads, construct labels explicitly, for example
`std::vector<int> body_labels = {0, 1, 1, 1, -1};` for a five-bead system.

## Reading the initialization implementation

Read the named methods in this order in src/integrate/rigid_body.cu:

1. initialize_impl validates sizes, labels and the box, then calls create_bodies.
2. create_bodies sorts (label, Atom index) pairs, builds contiguous member ranges
   and the reverse mapping, and allocates the body/member GPU arrays.
3. initialize_impl prepares coherent coordinates as described below.
4. setup_bodies_static reads current DEVICE masses, computes M and the center
   relative to a reference bead, then builds the symmetric inertia tensor.
   principal_axes performs a scaled 3x3 Jacobi eigendecomposition, sorts moments,
   fixes eigenvector signs and enforces a right-handed basis. axes_to_quaternion
   stores the corresponding unit body-to-space quaternion. Member displacements
   are inverse-rotated relative positions. The center is split into xcm/image.
5. setup_bodies_dynamic reads current DEVICE velocities, computes V and
   L=sum(r_i cross m_i*(v_i-V)), then p=2*q*(0,R(q)^T*L).
6. Zero loads are uploaded and initialized becomes true only after all checks.

Static initialization rejects multi-bead geometries if the smallest principal
moment is <= 1e-10 times the largest; coincident bodies are also rejected.
Single-bead bodies are handled separately with zero inertia and identity q.
Initialization verifies that reconstructed member images fit GPUMD's int image
storage, in addition to checking int64 body-image conversion/addition.

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

If Atom.position_image exists, only the reference bead's image is used as the
body's common absolute image. Other bead images may be inconsistent at startup;
they do not determine relative geometry in this mode. The common integer image
is added to the body image after local geometry is computed, never to large
double coordinates. Atom.unwrapped_position and CPU position/velocity caches
are not read. Managed bead data must be finite and mass must be positive;
unmanaged bead values are not used in the physical calculations.

Wrapped coordinates cannot distinguish a truly large molecule from a compact
periodic alias. The default mode therefore requires the caller to satisfy the
compact-molecule assumption even when the span check passes. For general body
geometry or authoritative image history, use the explicit overload:

```cpp
// HOST 3N SoA; all members of each body must be in the intended coherent image.
rigid.initialize(atom, box, body_labels, coherent_unwrapped_positions);
```

This mode uses exactly the supplied positions and ignores Atom position/image
buffers. Current DEVICE masses and velocities are still required. Member count
is unrestricted in both modes. Very large double coordinates in explicit input
can already have lost relative precision; integer image storage cannot restore
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

For example, bodies of sizes 1,3,1 have first_member values 0,1,4 and use five
entries in each member array. Body labels need not be contiguous: initialization
compresses labels in ascending order and sorts members by Atom index. A
HostSnapshot copies both body records and the member arrays, so the range
indices can still be interpreted on the host.

The only authoritative rotational state
is normalized quat plus tangent conjugate momentum conjqm. Angular momentum,
angular velocity, and rotation axes are derived locally when needed.

## Periodic center-of-mass coordinates

BodyState stores Cartesian xcm near the primary box and a BodyImage containing
three signed 64-bit integer lattice-image counters. In periodic directions,
the fractional components of xcm are in [0,1); nonperiodic directions are not
wrapped and their image counters remain zero. With box matrix H,
X_cm_unwrapped = xcm + H*image. This relation defines an absolute image, but
the large unwrapped coordinate is not reconstructed for integration.

Initialization computes or accepts coherently unwrapped members of each body.
Compute the center and fixed relative geometry from those coordinates, then
split the center into xcm and image. For a length-10 periodic box, members at
9.8 and 10.2 have center 10, stored as xcm=0 and image=1. Their relative offsets
are -0.2 and +0.2. Averaging wrapped coordinates 9.8 and 0.2 would give the
wrong center. A single-bead body follows the same coordinate splitting rule.

After each translational drift, remap_centers wraps the center and updates its
image. Let s=H^-1*xcm and k=floor(s) in periodic directions, with k=0 in other
directions. Replace xcm with xcm-H*k and image with image+k. Handle multiple
crossings in one call, including negative coordinates; check integer conversion
and counter addition for overflow. Remapping changes no velocity, quaternion,
momentum, or body-relative geometry. The operation can be fused into the drift
kernel; the separate interface describes its responsibility, not a required
additional launch.

Reconstruct each bead from the small local coordinate y_i=xcm+R(q)*s_i.
Wrapping y_i gives a local image k_i, position_i=y_i-H*k_i, and absolute bead
image_i=body.image+k_i. Overwrite the bead's old image rather than accumulating
it. In the example above, the beads have positions 9.8 and 0.2 and absolute
images 0 and 1. Torque arms remain R(q)*s_i, independent of all image counters.
Use the full H and its inverse, including for triclinic boxes.

Ensure Atom's image buffer exists before writing managed members. GPUMD's
position_image currently uses int, whereas BodyImage uses int64_t: check range
before conversion and report overflow, never silently truncate. The wider body
counter does not remove the existing bead-image range limit. If unwrapped output
is enabled, synchronize it from position_i+H*image_i; such output can still lose
precision at large absolute displacements and must not feed back into dynamics.
Already-wrapped bead coordinates and absolute images must remain consistent
with Force::compute's subsequent PBC processing.

The initial implementation assumes a fixed box matrix and fixed periodicity.
Changing the box needs an explicit center-scaling/image policy before NPT or
box deformation can be supported. Nonperiodic coordinates may still grow;
wrapping only addresses numerical growth from periodic travel.

Use one thread per body as the initial execution plan for the 1/3-bead mixture.
Each thread loops over its own member range, gathers forces, and writes its
distinct beads. No atomics are required. The same storage can support a warp or
block per body for substantially longer member lists later. Logical operations
may be fused within an integration-stage kernel. This is a starting design,
not a measured performance claim; use representative end-to-end timing before
changing execution strategies or adopting strict SoA.

## Single-bead molecules

Assign a nonnegative body label to a single-bead molecule to manage it as a point
body through the same translation path. Label -1 instead means an unmanaged
bead, updated by the ordinary bead integrator; never integrate it twice.

A single-bead body has mass m, xcm/image representing the bead's position,
vcm equal to its velocity, displace zero, inertia zero, and rotational_dof zero.
Use identity quat and zero conjqm. Its body force is the bead force, torque is
zero, and its velocity reconstruction is vcm. Skip all rotational kicks/drifts;
there are three translational DOF and no rotational DOF. These are implementation
contracts. The initialization part is implemented; runtime kernels remain unwritten.

Multiple beads may also form linear or coincident geometries. Do not infer
rotational DOF solely from natoms. Initialization explicitly rejects these
multi-bead degeneracies in this version.

## Miller conventions

- Quaternion order is (w,x,y,z), with Hamilton multiplication.
- q rotates body coordinates to space coordinates.
- L_body = R(q)^T L_space and p = 2 q (0,L_body).
- q is unit length and q dot p = 0.
- During a torque kick, q is fixed and p changes by
  2 dt q (0,R(q)^T torque_space). Each half kick receives dt/2.
- Free drift uses axes 3,2,1,2,3 with durations dt/2,dt/2,dt,dt/2,dt/2.
- Time is GPUMD's internal time, not an unconverted value in femtoseconds.
- Single-bead bodies bypass rotational integration entirely.
- Zero-inertia axes are skipped by the existing free-flow function. That alone
  does not validate multi-bead linear/coincident geometries: test these separately
  or explicitly reject them in the first initialization version.

Before the first step, initialize/project bead states, evaluate bead forces,
and form the initial load. Stage one applies half kicks and full translation/
rotation drift, then remaps centers and reconstructs bead positions and any required half-step
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

The initialization test checks single-bead and boundary-straddling three-bead
bodies, known principal moments, unequal masses, triclinic/mixed PBC, rigid
velocity projection, repeated eigenvalues, 24 spatial orientations, explicit
large-body geometry, unmanaged/empty systems, invalid input, image overflow and
cleanup after failure. It reads back real GPU state and deliberately supplies
incorrect CPU caches. Runtime remapping, bead reconstruction and force reduction
remain unimplemented; their crossing/force-consistency tests and full-system
NVE tests are still required.

Run the implemented subset from the repository root:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target test_quaternion test_quaternion_gpu test_rigid_body_initialization -j4
ctest --test-dir build -R '^unit.test_(quaternion(_gpu)?|rigid_body_initialization)$' --output-on-failure
```
