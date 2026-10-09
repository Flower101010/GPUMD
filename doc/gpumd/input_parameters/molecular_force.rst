.. _kw_molecular_force:

``molecular_force``
===================

Add fixed bond, angle and proper-dihedral terms to an MD potential. Two forms
are supported::

  molecular_force combined.in

or::

  molecular_force bonded_parameters.in topology.in

The first form retains the existing ``gpumd_molecular_force`` version-1
(bonds only) and version-2 (bonds, angles and dihedrals) formats. The second reads
``gpumd_bonded_parameters 1`` shared coefficients and ``gpumd_topology 1``
connectivity separately. See :ref:`kw_nep_molecular_force` for the shared parameter
format and :ref:`kw_cg_model` for the topology format.

Specify the keyword once, before ``run``. Any ``replicate`` must precede it; atom
counts and indices refer to the complete replicated system. For a trained
residual CG model, use :ref:`kw_cg_model` to load and check its complete package.
