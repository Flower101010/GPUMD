.. _kw_nep_molecular_force:

``molecular_force``
===================

Enable fixed bonded terms and train the NEP residual using per-frame topology::

  molecular_force bonded_parameters.in per_frame

The filename is relative to the run directory. Specify the keyword once.
It is supported by the ordinary potential model in ``nep``, including resident
and streaming batches and generic and specialized training kernels. It is not
supported by ``gnep``, temperature, dipole, polarizability, charge or vdW models,
or atomic tensor labels.

Shared parameters
-----------------

The parameter file contains all three sections, in this order::

  gpumd_bonded_parameters 1
  harmonic_bond_parameters 2
  1.2 5.0
  1.4 3.0
  harmonic_angle_parameters 1
  1.9 2.0
  periodic_dihedral_parameters 2
  0.2 3 0.4
  0.07 1 -0.3

Table row indices are zero-based. Each section may have zero entries.
Blank lines and ``#`` comments are allowed. Units and column meanings are:

* Bond: equilibrium distance in angstrom, force constant in eV/angstrom squared;
  :math:`U = k(r-r_0)^2/2`.
* Angle: equilibrium angle in radians, angle constant in eV/radian squared;
  :math:`U = k(\theta-\theta_0)^2/2`.
* Proper dihedral: strength in eV, positive integer multiplicity, phase in radians;
  :math:`U = k[1+\cos(n\phi-\delta)]`.

Strengths must be finite and nonnegative. A zero strength disables the term.
Angles may have equilibrium values from zero to pi. This interface does not
optimize the bonded parameters.

Per-frame topology
------------------

Every frame of ``train.xyz`` and any supplied ``test.xyz`` must include the
following header fields, even when a list is empty::

  cg_topology_version=1 cg_bonds="0,1,0;1,2,1;2,3,0" cg_angles="0,1,2,0;1,2,3,0" cg_dihedrals="0,1,2,3,0;0,1,2,3,1"

A bond tuple is ``i,j,type``, an angle tuple is ``i,j,k,type`` with center ``j``,
and a proper dihedral tuple is the ordered ``i,j,k,l,type``. Separate tuples by
semicolons, use double quotes, and omit whitespace inside each list. All atom
indices are zero-based and local to the frame. The type indexes the shared table
of the same interaction family. Multiple terms on the same ordered atom tuple
are allowed. Explicitly empty lists use ``"none"``; missing, duplicated or unknown
``cg_`` fields are errors.

Frames can have different bead counts, molecule counts and connectivity, while
sharing the bead species in ``type`` and the coefficient tables. ``mol_id`` is
optional auxiliary metadata and is not used to infer bonds. The initial interface
assumes three-dimensional periodic cells; explicit ``pbc`` must be ``"T T T"``.
Use coordinates wrapped into the original cell. Consecutive minimum-image bond
vectors define the dihedral image convention, as in bonded MD.

Labels, predictions and precision
--------------------------------

Supply **total** reference energy, force and virial. Do not subtract the bonded
terms beforehand. The evaluator adds the fixed baseline once to each fresh NEP
prediction before computing fitness or writing outputs. XYZ energy and virial
are frame totals; existing output energy/virial columns remain per-bead values.
Forces remain in eV/angstrom. Missing frame virial labels remain optional.

Potential-model ``prediction 1`` regenerates train outputs and, when ``test.xyz``
is present, test outputs. The corresponding reference columns remain unchanged.
The topology does not change the ordinary NEP descriptors: identical coordinates
and bead species have identical residual predictions even if their connectivity
differs.

The baseline is evaluated and accumulated in double precision from the existing
float training coordinates and cell, then packed into float prediction arrays.
Very large bonded terms can obscure a small residual in float total labels and
predictions. Double evaluation alone does not remove that limitation.

The exported ``nep.txt`` contains the **residual NEP**, so it is not the complete
CG model. Retain the shared parameter file, bead-type mapping, units and topology
convention. Each MD system also requires its own topology and masses. The
training parameter file and XYZ topology fields are not yet a two-file MD
``molecular_force`` input; complete model packaging and that MD interface are a
separate implementation stage.
