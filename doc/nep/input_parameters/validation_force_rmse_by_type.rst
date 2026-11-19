.. _kw_validation_force_rmse_by_type:
.. index::
   single: validation_force_rmse_by_type (keyword in nep.in)

``validation_force_rmse_by_type``
=================================

Set ``validation_force_rmse_by_type 1`` in ``nep.in`` to report unweighted
validation force RMSE for water beads (``O``) and methanol beads (``C``).
The default is ``0``. The report uses the existing ``test.xyz`` predictions
at each :ref:`kw_output_interval`. With a fixed bonded baseline, it compares
the total predicted forces, including bonded forces, with the total labels.

The output is ``validation_force_rmse_by_type.csv``. At each reporting step,
rows with scope ``all`` summarize the full validation set by type. Rows
with scope ``concentration`` give the same statistics for each distinct
methanol fraction, recorded as the reduced integer ratio of C beads to
all beads in the configuration. For each row, ``N`` is the number of beads.
The RMSE in eV/Å uses the sum of squared Cartesian residuals divided by
``3*N`` before taking the square root. A missing type has ``N=0`` and
an empty RMSE field.

The CSV starts fresh at the first validation report of each invocation,
including when training starts from a restart file. Later reports in
that invocation are appended. With no ``test.xyz``, no CSV is written.
