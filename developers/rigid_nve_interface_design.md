# Rigid NVE: ensemble interface and execution contract

Status: shared-operator/thermo refactor, 2026-10-14. The fixed-box rigid NVE
implementation and run.in entry have been validated by the user, and the prior
branch was merged into local master/tagged rigid-nve-v1. The current refactor
preserves NVE's two-stage physical algorithm and public interface while
extracting reusable integration operations and RigidThermo. Focused CTests,
CUDA memory checks and the gpumd build are used for this change; the final-scale
all_rigid_3000 and mixed_2700_300 simulations are not rerun. See the final
section for current APIs and validation scope. Earlier dated sections retain
the design history. This builds on rigid_body_design.md.

## Goal and first validation system

Implement a small fixed-box rigid NVE integrator connected to GPUMD's existing
bead-force interface, including NEP. The primary validation system is a mixture
of free single-bead particles and nonlinear three-bead triangles. Artificial
masses, geometry and interactions are acceptable, provided the reference model
is physically consistent. Use asymmetric triangles with unequal masses to
exercise the rotational equations. Agreement with physical invariants and
timestep convergence matters more than trajectory-by-trajectory agreement with
LAMMPS.

The implementation remains incremental: interfaces, free-particle selection,
integration stages, input wiring, and validation can be separate changes.

## Reuse the molecule grouping, not the meaning of `fix`

Use an existing XYZ grouping method dedicated to molecule membership. Its index
is selected explicitly; no implicit `group[0]` assumption is introduced.

```text
ensemble rigid_nve 0
```

This proposed command takes exactly one parameter: the grouping-method index.
Every nonempty multi-member group in that method becomes one rigid body. Every
singleton becomes one free particle. A singleton must have its own group label;
giving all free particles the same label would incorrectly create one body.
Nonlinear geometry is still checked by `RigidBodySystem::initialize`.

Reuse the existing XYZ reader, `Group`, integer/range validation conventions
used by `parse_fix`, and `rigid_body_labels_from_group`. The adapter already
copies `cpu_label` and maps singleton labels to -1 without changing `Group`.
Retain the manager's compressed membership storage and GPU reverse map rather
than making a second persistent member list in the ensemble.

GPUMD's current `fix [grouping_method] group_id` means immobilize atoms. It does
not select rigid molecules. Keep that command's grammar and semantics intact.
The first rigid NVE version rejects combinations with `fix` or `move`; support
for immobilized or driven whole bodies is a separate feature.

## Ownership and interfaces

`Ensemble_Rigid_NVE` derives directly from `Ensemble`, matching the organization
of `Ensemble_NVE`. It owns one `RigidBodySystem` for the lifetime of a run. The
manager owns geometry, membership, state and loads; the ensemble only chooses
integration stages. `Force::compute` continues to own interaction evaluation.

The proposed declaration is a design sketch, not a replacement for a live header:

```cpp
class Ensemble_Rigid_NVE : public Ensemble
{
public:
  explicit Ensemble_Rigid_NVE(int molecule_grouping_method);

  // Called once per run, before the first Force::compute.
  // Projects managed bead velocities and reconstructs managed positions.
  void initialize(Atom& atom, const Box& box, const std::vector<Group>& groups);

  void compute1(double dt, const std::vector<Group>& groups,
                Box& box, Atom& atom, GPU_Vector<double>& thermo) override;
  void compute2(double dt, const std::vector<Group>& groups,
                Box& box, Atom& atom, GPU_Vector<double>& thermo) override;

  int thermal_degrees_of_freedom() const override;
  const RigidBodySystem& rigid_bodies() const { return rigid_; }

private:
  enum class Stage { uninitialized, full_step, half_step };
  int molecule_grouping_method_;
  int thermal_dof_ = 0;
  Stage stage_ = Stage::uninitialized;
  double step_time_step_ = 0.0;
  RigidBodySystem rigid_;

  // Uses the existing rotation helpers, independent of translation.
  void integrate_rotation(bool is_step1, double dt);
};
```

`initialize` converts the selected group labels, initializes the manager,
rebuilds positions and velocities, computes the supported DOF count, and enters
`full_step`. It does not read atom forces: they have not been evaluated yet.
The projected state, not the pre-projection random bead state, defines the
initial energy for NVE diagnostics. Do not silently rescale after projection.

`rigid_bodies()` is a read-only diagnostic interface. It allows the existing
host-snapshot API to be used in tests without exposing mutable body state to
unrelated modules. Device views are transient non-owning arguments, not stored
as ensemble members. Existing default-stream ordering remains in force.

The private rotation helper applies the existing math functions; it does not replace
the manager's reconstruction or force aggregation methods. No general-purpose
integrator hierarchy or new callback framework is needed.

## One translational Verlet algorithm, overloaded for both representations

The revised implementation leaves the existing ordinary ensemble kernels
unchanged. `src/integrate/rigid_body_integrator.cuh` exposes three overloads in
`rigid_integrator`:

```cpp
// Host/device arithmetic shared by particles and body centers.
void velocity_verlet(bool step1, double dt, double mass,
                     const Vec3& force, Vec3& position, Vec3& velocity);
void velocity_verlet(bool step1, double dt,
                     const RigidBodySystem::BodyGeometry& geometry,
                     const RigidBodySystem::BodyLoad& load,
                     RigidBodySystem::BodyState& state);
// One GPU launch handles both representations.
void velocity_verlet(bool step1, double dt, Atom& atom, RigidBodySystem& rigid);
```

The body overload delegates to exactly the same mass/force/x/v arithmetic as a
free particle. A single launch has N_body + N_atom entries: the first entries
update body centers; the remaining entries update atoms only when the existing
`atom_to_body` map is -1. Membership still has to be consulted to prevent double
integration, but there is no new free-member list and no selector added to the
ordinary ensemble API. Managed bead buffers are untouched until reconstruction.

The bulk overload validates initialized state, array sizes and finite dt, and
returns without a launch for empty systems. It only advances translation; it
does not aggregate loads, rotate, wrap, or reconstruct beads. These operations
remain explicitly ordered by the ensemble. Negative dt is permitted for
reversibility tests. Runtime integration will use the caller's chosen dt.

Use this entry point from rigid NVE in both `USE_NEPCG` and ordinary builds.
`velocity_verlet_cg` removes group COM velocity and leaves
internal motion unconstrained; it cannot integrate freely translating rigid
bodies or singleton particles. Do not route through `Ensemble_NVE::compute1/2`,
which selects that CG method under `USE_NEPCG`.

## State authority and stage order

At a full-step boundary, body `xcm`, `vcm`, `quat`, and `conjqm` are authoritative.
Managed bead positions and velocities are reconstructed views. Free-particle
positions and velocities remain authoritative in `Atom`. Every particle has
exactly one integrator.

The existing main-loop force evaluation remains between the two stages:

```text
Integrate::initialize
  create ensemble; attach existing Ensemble context pointers
  rigid ensemble initialize: group adapter -> manager -> rebuild x and v
Run::compute_force                          # initial bead forces

repeat:
  compute1
    aggregate current bead forces/torques
    unified translation step 1, free particles and body centers
    body rotational half kick + full rotational drift
    remap body centers; rebuild managed bead positions and velocities
  existing force evaluation                 # NEP or another supported potential
  compute2
    aggregate newly evaluated bead forces/torques
    unified translation step 2, free particles and body centers
    body rotational second half kick
    rebuild managed bead velocities
    compute thermodynamic diagnostics
```

Aggregate loads at the start of both stages. The extra aggregation in later
`compute1` calls is intentional for the first version: it avoids a special
initial-force hook and dependence on cached-load readiness. For the intended
short bodies the implementation is small and easy to audit. Removing redundant
aggregation is a later measured optimization.

`compute1` requires `full_step`, then enters `half_step`; `compute2` requires
`half_step`, then returns to `full_step`. Both stages use the same dt for a step;
the caller already passes GPUMD internal time units. No per-stage HOST snapshots
or device synchronizations are needed for normal execution.

The translation overload performs the first two translational lines below;
the future rotation kernel performs the two rotational calls (pseudocode):

```cpp
state.vcm += (0.5 * dt / geometry.mass) * load.fcm;
rigid_math::kick_rotation(state.quat, state.conjqm, load.torque, 0.5 * dt);
state.xcm += dt * state.vcm;
rigid_math::integrate_free_rotation(
  state.quat, state.conjqm, geometry.inertia, dt);
```

Stage two updates `vcm` with the new half kick and calls `kick_rotation` with
the new torque and dt/2. It does not drift positions or orientation again.
`integrate_free_rotation` already implements Miller's symmetric five-axis flow.
Do not add hidden quaternion normalization: monitor unit length and tangency,
and diagnose a broken integration sequence rather than altering momenta silently.

Rebuild half-step bead velocities in `compute1` as well as full-step velocities
in `compute2`, because the current force/measurement interfaces receive Atom's
velocity buffer. Only conservative, position-dependent forces are in the first
physical validation contract; velocity-dependent forces need their own treatment.

## Minimal input and lifecycle wiring

Keep `Integrate::type == 0` for both ordinary and rigid NVE. This preserves the
existing NVE temperature bookkeeping and avoids the special `type >= 31` PIMD
branches in Run and measurements.

Add `rigid_grouping_method = -1` to `Integrate`. `parse_ensemble` resets it before
each new ensemble selection; `ensemble rigid_nve k` sets it after validation.
Like `type`, this selection persists through `Integrate::finalize`; a subsequent
ensemble command resets it. This avoids silently changing consecutive run blocks
to ordinary NVE. The case-0 construction chooses ordinary NVE for -1 and rigid
NVE otherwise. The existing unique_ptr owns the instance; a temporary typed
pointer performs rigid initialization after shared Ensemble context fields are
attached. A new virtual initialization hook on every ensemble is unnecessary.

For each `run`, the current Integrate lifecycle constructs a fresh ensemble.
The first version therefore reconstructs body state from current bead x/v at
each run boundary. It is a fresh initialization, not exact continuation of the
previous quaternion/principal-axis representation. Record the boundary and
check segmented versus uninterrupted runs before claiming restart fidelity.
The compact-coordinate assumptions apply again at each such initialization.

Run-level preflight also checks features outside Integrate's knowledge:
periodic velocity correction, enabled unwrapped storage, and active operations
that change positions/velocities/box independently of the body state. Reject
unsupported combinations before the first force evaluation; do not merely let
reconstruction overwrite a requested correction. `velocity.correct_velocity`
runs before every `compute1` in the current loop and would otherwise desynchronize
the two representations. Membership, masses and atom order are fixed within a
run. Whole-body corrections and continuous trajectory history remain future work.

## Thermodynamic output: preserve energy when fixing temperature

Reconstructed bead velocities give the correct total kinetic energy:

```text
K = sum_free 1/2 m v^2
    + sum_bodies [1/2 M V^2 + 1/2 L_body dot I^-1 L_body].
f = 3 N_free + 6 N_body                       # supported nonlinear bodies
T = 2 K / (f k_B).
```

The first version reports this full-space DOF convention; do not automatically
subtract global momentum DOFs. An ensemble with explicitly constrained total
momentum needs a documented alternative convention.

The implemented `virtual int thermal_degrees_of_freedom() const` in `Ensemble` returns 0
by default to mean legacy behavior. Rigid NVE returns its initialized f.
`find_thermo` uses a positive override in its temperature denominator, otherwise
retaining the current denominator. `Dump_Thermo` must use the same positive
override when reconstructing KE from T, otherwise retaining its existing path.
Record f and the rigid ensemble name in the output header.

These changes must be made together: changing only T would make the current
`KE = 1.5 N k_B T` output wrong and corrupt an apparent energy-drift check.
Require positive f when reporting temperature. Independently verify bead KE
against the body-state expression above.

Atom's force/virial arrays remain interaction-only. The rigid ensemble stores
constraint virial separately and adds it to global thermo pressure after the
existing interaction-plus-bead-kinetic reduction. The header now identifies this
as `interaction_plus_bead_kinetic_plus_rigid_constraint`. This does not supply
per-atom constraint stress or change NEP labels. Read the implementation and
validation boundary below before using other measurement interfaces.

## Implementation order and acceptance

1. Implement and validate the unified translation overloads. This part is now
   implemented; its targeted test includes analytic acceleration with distinct
   stage forces, interleaved membership, untouched managed bead buffers,
   all-rigid/all-free/empty cases, harmonic reversibility and a canonical
   determinant-one check. These tests do not establish full rigid NVE correctness.
2. Rigid ensemble initialization and the rotational stage kernel are implemented;
   test through compute1 -> controlled force evaluation -> compute2 without
   depending on a new run.in command.
3. Paired temperature/KE handling and output metadata are implemented. Ensemble
   guards reject incorrect stage order, changed dt, fixed/moving groups, box
   deformation, array mismatches and unsupported rigid unwrapped storage.
   Input selection and Run-owned compatibility guards are now implemented and
   the executable compiles. Input/runtime testing is left to the user.
4. Run the agreed free-particle/three-bead-mixture validation, then connect the
   same ensemble to NEP for integration validation.

The first dynamic tests should cover force-free translation, torque-free
asymmetric rotation, and a controlled kick before interacting mixtures. The
mixture should include unequal bead masses, interleaved atom indices, body
rotation and periodic crossings. Use a smooth conservative test potential with
known reference energy and forces; an inter-molecular pair-potential fixture may
exclude intra-body pairs explicitly, but that does not imply arbitrary exclusions
can be applied to a trained NEP.

Before assigning numerical timestep and interaction parameters, choose the
conservative test potential and its length/energy scales. Check at least three
successively halved timesteps over the same physical duration. Acceptance
requires rigid distances, unit quaternion and tangent momentum, total linear
momentum, bounded energy error with convergence, and equality of the two KE
calculations. Angular momentum is checked for an isolated body or an isolated
nonperiodic interacting system; do not impose global angular momentum conservation
on a general periodic cell. An all-free reference must use ordinary particle
Verlet, not the COM-removing CG branch.

Compilation and existing manager/math tests are necessary evidence but do not
establish end-to-end rigid NVE correctness. Keep those evidence boundaries in
the implementation report.

## Planned source footprint

| File | Planned responsibility |
| --- | --- |
| `src/model/rigid_body_input.cuh` | Reuse existing group-to-label adapter unchanged |
| `src/integrate/rigid_body.cuh/.cu` | Reuse existing manager; preserve current user edits |
| `src/utilities/math.cuh` | Reuse existing kick and free-rotation helpers |
| `src/integrate/ensemble_rigid_nve.cuh/.cu` | New ensemble and body-stage kernels |
| `src/integrate/rigid_body_integrator.cuh/.cu` | Implemented unified translation overloads and GPU launch |
| `src/integrate/ensemble.cuh/.cu` | Implemented temperature DOF hook; ordinary Verlet remains unchanged |
| `src/integrate/integrate.cuh/.cu` | Parse option, select ensemble, initialize and reset |
| `src/main_gpumd/run.cu` | Compatibility preflight using Run-owned feature settings |
| `src/measure/dump_thermo.cu` | Matched KE denominator and diagnostic metadata |
| Build definitions and new integration tests | Register new source/targets if required by the current build |

No NEP descriptor, training code or force interface change is part of this design.

## Ordering and symplectic structure

For a fixed-box conservative system, the Hamiltonian separates as
H = K_translation + K_rotation + U(X,q,x_free). The translational and rotational
kinetic terms depend on different canonical variables, so their Poisson bracket
is zero. Their free drifts commute. The rotational drift is approximated by the
existing symmetric Miller 3-2-1-2-3 composition of single-axis Hamiltonian flows.

The complete step is a potential half kick, translational and rotational free
drifts, a new force/torque evaluation, and a potential half kick. Its symmetric
composition is second order and symplectic on the rigid phase space in exact
arithmetic, under the fixed-step, conservative-force assumptions. See Miller
et al., [Symplectic quaternion scheme for biophysical molecular dynamics](https://authors.library.caltech.edu/records/4ap9g-14r38).

Translation and rotation can be called in either order inside each stage when
both use loads already evaluated at the SAME full-step pose. Translational
operations do not change q/p, and rotational operations do not change X/V;
with these stored loads the calls can be reordered into the common half-kick
then drift structure. No force recomputation may be inserted between those
calls. This statement does not authorize swapping an entire force-update cycle
or evaluating torque on a partly advanced pose.

In general, kicks and drifts do not commute, and the three rotational axis
flows do not commute for an asymmetric body. Preserve the two outer half kicks
and the internal Miller palindrome. Another symmetric axis ordering can define
another valid splitting, but it is not the same numerical map and is outside
this implementation's contract.

Do not claim symplecticity with state-dependent adaptive timesteps, velocity
rescaling/corrections, inconsistent force and torque poses, or unrelated
quaternion renormalization/projection each step. The current adaptive-step
option therefore needs to be rejected for the first validation mode. A chosen
fixed dt, a conservative NEP energy gradient, consistent force reduction, and
reconstruction from body state are the intended production contract. Finite
precision and actual implementation still require runtime validation; the
existing tests do not prove full-system symplecticity.

## Verification of the implemented translation subset (2026-10-13)

The gpumd target and the new translation test built successfully. CTest passed
`unit.test_rigid_body_translation`, `unit.test_quaternion`, and
`unit.test_quaternion_gpu` (3/3). CUDA Compute Sanitizer memcheck on the new
translation test reported zero errors. The translation test includes the
ensemble header without instantiating it.

The subsequent `test_ensemble_rigid_nve` directly instantiates the ensemble and
uses a nonperiodic mixture of a free particle and an unequal-mass triangle.
A conservative harmonic interaction between the free particle and one body
member supplies both COM force and nonzero torque. It checks rigid distances,
total linear momentum, quaternion norm/tangency, bead versus body kinetic energy,
rigid DOF temperature, potential energy, and velocity-reversal return to the
initial bead state. It also tests stage/input guards, ordinary legacy temperature,
all-free integration, and an all-rigid triangle straddling a periodic boundary.

At fixed physical duration 4 in GPUMD internal time units, maximum energy errors
at dt=0.04, 0.02 and 0.01 were 6.15134e-5, 1.53784e-5 and 3.84460e-6 in internal
energy units, respectively: successive ratios are approximately 4. Torque-free
rotation/translation at dt=0.02 had maximum energy error 1.91926e-8. These are
artificial-fixture results, not a NEP-driven trajectory or a complete proof of
symplecticity.

## Implemented constraint virial (2026-10-13)

The implementation follows LAMMPS `FixRigid::set_xv` / `set_v`, audited at local
commit c71ef819ab3785bba81c31c0ef5fce84603608d3:

```text
fc_i = m_i * (v_after_i - v_before_i) / (dt/2) - f_interaction_i
W_c  = 1/2 sum(r_old_i tensor fc_stage1_i)
     + 1/2 sum(r_new_i tensor fc_stage2_i)
```

The first-stage rotation kernel saves the old quaternion. Constraint tallying
runs after body integration but BEFORE managed bead velocities are overwritten.
It reads the actual old bead velocity, reconstructs the new velocity with the
same formula as the manager, and uses old orientation for stage-one position
and current orientation for stage two. These are discrete impulse estimates,
not an assertion of pointwise equality to analytic instantaneous constraint forces.

Positions are COM-relative arms R(q)*s_i. For the supported freely moving bodies,
sum(fc_i)=0 in each stage up to floating-point error, so shifting all body
positions by a common center does not change total constraint virial. This is
why coherent continuous trajectories/image histories are unnecessary for this
GLOBAL pressure calculation. It is not the same per-atom partition as LAMMPS
absolute-position stress/atom or its geometric-centroid stress.

The ensemble owns a 6*B DEVICE array in SoA order xx,yy,zz,xy,xz,yz and B old
quaternions. One thread per body loops over members. Stage one overwrites the
previous step; stage two adds the second half. A separate six-component reduction
adds sum_b W_c,b / volume to thermo[2:8] after find_thermo. Free particles require
no constraint tally. The read-only constraint_virial_per_body accessor exposes
the current stage result for diagnostics. Initialization clears it to zero;
only the completed compute1/force/compute2 cycle defines a full-step value.
No startup virial guesstimate is used.

The implementation does not modify force/virial labels, forces, body momenta,
or the integration sequence. It does not apply an empirical kinetic correction.
The bead kinetic tensor remains in pressure, with the explicit constraint term
accounting for the rigid kinematics.

The test evaluates each stage independently from CPU bead position/velocity
snapshots and checks all six components, both 1/2 factors, old versus new pose,
sum(fc)=0, resetting between steps, and inclusion in pressure. The harmonic
fixture now supplies its analytic interaction virial too. A second fixture
compares two rotating unequal-mass triangles plus a singleton with coherent
nonperiodic versus wrapped periodic inputs in a tilted box, checking the
per-body tensors and global reduction. Interaction force/virial arrays are
checked unchanged by the constraint tally.

For torque-free rigid/free motion, the target scalar numerator is
3PV = 2K_free + 2K_body_translation. At dt=0.08, 0.04, 0.02 the maximum residuals
are 8.25054e-6, 2.06314e-6, 5.15845e-7 in internal energy units. Each halving
reduces the finite-step residual by about four. The prior energy/reversibility
checks still pass with the same energy errors. The gpumd executable builds;
all six targeted CTests pass; Compute Sanitizer memcheck reports zero errors.

Current boundary: global thermo pressure after compute2 is corrected; Atom
per-atom virial is still the interaction contribution. Direct calls to the base
find_thermo only perform its raw reduction. In particular dump_observer observe
recomputes alternate model forces and raw thermo, so it is not compatible with
cached main-model constraint virial. The connected Run preflight rejects that
combination until a model-specific observer design exists. End-to-end input
validation, NEP-driven pressure comparison and per-atom constraint stress remain
separate work.

The updated gpumd executable builds and links. CTest passes the six targeted
tests (quaternion host/GPU, rigid initialization/runtime/translation, and the
new ensemble test). CUDA Compute Sanitizer memcheck on the new ensemble test
reports zero errors.

## Input connection and user testing handoff (2026-10-13)

`ensemble rigid_nve k` validates a zero-based molecule-grouping method and keeps
the existing type-0 force/integration path. It sets otherwise unused temperature
target fields to zero; temperature-dependent NEP models are explicitly rejected.
The rigid initialization runs before the first bead force evaluation. Runtime
logs report body/free-particle counts and thermal DOFs. No main-loop stage order
or NEP force interface is changed.

Run preflight rejects adaptive timesteps, periodic bead velocity corrections,
unwrapped-position tracking, and actions not yet adapted. Integrate rejects
fixed/moving groups and deformation. The initial supported action list is
dump_thermo, dump_xyz, dump_netcdf (if compiled), dump_restart, compute_rdf,
compute_adf, compute_angular_rdf and compute_orientorder. Wrapped dump_xyz now
enables image/unwrapped tracking only when that quantity is explicitly requested;
its earlier unconditional allocation would have blocked all rigid XYZ output.
A new plain time_step command clears an earlier adaptive distance limit.

Consecutive run commands retain the selection, but each run reconstructs a new
rigid manager from bead x/v, as described above. Ordinary ensemble commands
reset the rigid selection. Syntax and output conventions are documented in
doc/gpumd/input_parameters/ensemble_standard.rst.

For THIS input-connection change, only gpumd compilation/linking and diff
whitespace checks were performed. No CTest, runtime simulation or submitted
validation task was run, at the user's request. Earlier unit-test results apply
to the previously implemented components, not acceptance of the newly wired
run.in path. The executable is build/gpumd; the user will perform the runtime
and physical tests.


## Shared operations and thermo refactor (2026-10-14)

The user-validated NVE baseline was merged into local master and tagged
`rigid-nve-v1`. Refactoring continues on `codex/rigid-integrator-thermo`.
This change separates reusable code without implementing a thermostat/barostat
or changing the existing run.in selection and Ensemble compute1/compute2 API.

Responsibilities are now:

- RigidBodySystem owns body geometry/state, gathers model forces and reconstructs
  beads. Its fixed-box and nonlinear-body contract is unchanged.
- rigid_integrator owns explicit translation/rotation kick and drift operations.
  Every operation receives its actual duration h in internal time units. There
  is no hidden half-step factor: callers pass dt/2 to a half kick. Translation
  covers COMs and free particles; rotation covers bodies only. No operation
  reconstructs beads or computes thermo. Negative finite h is allowed for
  reversibility tests. Drift does not require an Atom force buffer.
- RigidThermo owns thermal DOFs, previous orientations and the per-body
  constraint virial. It does not integrate or change model force/virial arrays.
- Ensemble_Rigid_NVE retains selection, initialization, fixed-dt/stage checks
  and the ordering of these operations. It no longer owns physics kernels.

The legacy velocity_verlet overloads remain as small convenience wrappers for
existing callers/tests. Their GPU translation kick/drift remains fused into one
launch. NVE now calls the explicit operators, so their order is visible and can
later be composed with thermostat operators. Separate launches may cost more
than the former fused NVE kernels; no new performance claim is made here.

The two-stage sequence is:

```text
compute1:
  validate box; gather old forces
  thermo.begin_step (save old orientations)
  translation kick(dt/2), drift(dt)
  rotation kick(dt/2), free drift(dt)
  remap COMs; rebuild positions
  thermo.record_first_half(dt/2)
  rebuild half-step bead velocities

force module: evaluate new interaction forces/virial

compute2:
  validate box; gather new forces
  translation kick(dt/2), rotation kick(dt/2)
  thermo.record_second_half(dt/2)
  rebuild full-step bead velocities
  existing find_thermo reduction
  thermo.add_constraint_pressure
```

The two constraint samples retain the original NVE definition and weights:
fc=m*(v_after-v_before)/h-f_interaction; each contributes 0.5*r tensor fc.
The first sample uses the saved old orientation and overwrites the prior step;
the second uses the new orientation and accumulates. Atom velocities must still
be the velocities BEFORE the corresponding physical kick when a sample is
recorded. Consequently each record precedes rebuild_velocities.

RigidThermo enforces begin -> first -> second ordering and equal positive half
step durations, checks topology/Atom arrays, and disallows pressure correction
from incomplete samples. Pressure correction must be added once to a freshly
reduced base thermo array. It is not an idempotent full-thermo calculation.

This makes the physical sampling window explicit for future NVT, but does not
prove that the current impulse formula can be reused unchanged for NPT or
arbitrary thermostat placement. Bath/barostat impulses must not silently enter
a constraint sample; pressure and kinetic energy time levels still need a
separate design when those ensembles are implemented. Kinetic-energy partition
and momentum rescaling are also future work, not part of this refactor.

Validation: the six existing focused CTests pass, plus the new
`test_rigid_body_operators`, which checks full operation durations, independent
translation updates, known rotational impulse, analytic principal-axis rotation,
negative-time reversal, and the thermo window guards. Compute Sanitizer memcheck
passes both test_ensemble_rigid_nve and test_rigid_body_operators with zero errors.
The gpumd target is compiled/linked separately. At the user's request, neither
all_rigid_3000 nor mixed_2700_300 final-scale simulations is rerun; previous
user validation belongs to the tagged baseline, not a new long-run measurement.

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target test_ensemble_rigid_nve test_rigid_body_operators -j4
ctest --test-dir build -R '^unit.test_(quaternion(_gpu)?|rigid_body_(initialization|runtime|translation|operators)|ensemble_rigid_nve)$' --output-on-failure
```
