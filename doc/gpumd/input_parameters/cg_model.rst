.. _kw_cg_model:

``cg_model``
============

Load a complete fixed bonded + residual NEP model::

  cg_model cg_model.json topology.in

This replaces the separate ``potential`` and ``molecular_force`` declarations.
Do not combine these declarations or repeat ``cg_model``. Use this keyword before
``run``; any ``replicate`` must precede it, and the supplied topology must describe
all atoms after replication. A restart XYZ needs the same model declaration and
its corresponding topology again. Masses and any initial velocities are supplied
by ``model.xyz``; they are properties of the MD system, not learned parameters.
Only ordinary NEP4/NEP4-ZBL and three-dimensional periodic cells are supported.

Model package
-------------

Training with :ref:`kw_nep_molecular_force` writes ``cg_model.json`` and
``cg_model.bonded.in`` alongside the residual ``nep.txt``. Saved checkpoints have
``<nep_filename>.cg_model.json`` and ``<nep_filename>.cg_model.bonded.in`` companions.
Copy the three files together. The bonded snapshot is serialized from the
parameters actually loaded for training, with enough digits to preserve doubles.
It does not depend on the original input parameter path.

The version-1 JSON manifest records:

* ``format`` = ``gpumd_cg_model`` and integer ``version`` = 1;
* ``units`` = ``eV_angstrom_radian``;
* ``bonded_convention`` = ``harmonic_half_k_periodic_proper_v1``;
* ``image_convention`` = ``consecutive_mic``;
* ``nep_file``, ``bonded_parameters_file`` and their lowercase SHA-256 checksums
  (``nep_sha256`` and ``bonded_parameters_sha256``);
* ``bead_types`` in the exact species order of the NEP header.

Member paths are filenames in the manifest directory. The manifest and topology
arguments are relative to the MD working directory, or may be absolute paths.
Checksums, conventions, type order, topology indices and atom count are checked
before loading the model. Missing and unknown fields or duplicate JSON keys are
errors. A system may contain only a subset of the shared bead types; the model's
full type order is retained.

A bare ``potential`` declaration pointing to a residual with a recognized
``cg_model.json`` or checkpoint companion is rejected. This check depends on the
companion being present and referring to that filename; deleting it or renaming
the residual alone removes the marker. Always deliver and load the whole package.

System topology
---------------

The independent topology file is versioned and uses zero-based atom and
interaction-type indices::

  gpumd_topology 1
  number_of_atoms 4
  bonds 3
  0 1 0
  1 2 1
  2 3 0
  angles 2
  0 1 2 0
  1 2 3 0
  dihedrals 2
  0 1 2 3 0
  0 1 2 3 1

All three sections must appear, even with zero entries. Blank lines and ``#``
comments are allowed. Parameters are shared between MD systems; topology is fixed
during each run. No molecule identifier is needed to compute the interactions.
Multiple dihedral terms on the same ordered quartet may be specified.
Consecutive minimum-image vectors define angles and proper dihedrals; each
connected vector must be recoverable using this image convention.
See :ref:`kw_nep_molecular_force` for coefficient units and potential formulas.
