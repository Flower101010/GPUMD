.. _kw_cutoff:
.. index::
   single: cutoff (keyword in nep.in)

:attr:`cutoff`
==============

This keyword enables one to specify the radial (:math:`r_\mathrm{c}^\mathrm{R}`) and angular (:math:`r_\mathrm{c}^\mathrm{A}`) cutoffs of the :term:`NEP` model.
One syntax is::

  cutoff <radial_cutoff> <angular_cutoff>

where :attr:`<radial_cutoff>` and :attr:`<angular_cutoff>` correspond to :math:`r_\mathrm{c}^\mathrm{R}` and :math:`r_\mathrm{c}^\mathrm{A}`, respectively.
The cutoffs must satisfy the conditions 3 Å :math:`\leq r_\mathrm{c}^\mathrm{A} \leq r_\mathrm{c}^\mathrm{R} \leq` 10 Å.

The defaults are :math:`r_\mathrm{c}^\mathrm{R}` = 8 Å and :math:`r_\mathrm{c}^\mathrm{A}` = 4 Å.
It can be computationally beneficial to use (possibly much) smaller :math:`r_\mathrm{c}^\mathrm{R}` but the default values should be reasonable in most cases.

Another syntax is::

  cutoff <radial_cutoff_species_1> <angular_cutoff_species_1> <radial_cutoff_species_2> <angular_cutoff_species_2> ...
  
which can be used to specify a set of radial and angular cutoffs for each species.
The cutoff between two species (:math:`a` and :math:`b`) is the arithmetic average of the cutoffs for the two species:

.. math::
   
   r_\mathrm{c}^\mathrm{R/A}(a,b) = r_\mathrm{c}^\mathrm{R/A}(b,a) = \frac{r_\mathrm{c}^\mathrm{R/A}(a) + r_\mathrm{c}^\mathrm{R/A}(b)}{2}

An individual pair can override this arithmetic-average rule by placing one or more
``cross_cutoff`` lines after ``cutoff``::

  cross_cutoff <type_i> <type_j> <radial_cutoff> <angular_cutoff>

Here ``type_i`` and ``type_j`` are different, zero-based indices in the order given by
:ref:`type <kw_type>`.  The override is symmetric, so only one of ``i j`` and ``j i`` may be
specified.  Pairs without an override continue to use the arithmetic average.  For example::

  type 3 C H O
  cutoff 6 4 5 3.5 7 4.5
  cross_cutoff 0 2 8 5

uses explicit radial and angular cutoffs of 8 Å and 5 Å for C--O, while C--H and H--O retain
the arithmetic-average rule.  Pair-specific cutoffs are currently supported for ordinary NEP
potential, dipole, polarizability, and temperature-dependent models.  They are not supported
for qNEP, gNEP, vdW, or charge-vdW models.
