.. _kw_composition_force_weights:
.. index::
   single: composition_force_weights (keyword in nep.in)

``composition_force_weights``
=============================

For NEP force training with water beads ``O`` and methanol beads ``C``,
set the following in ``nep.in``:

.. code-block:: text

   composition_force_weights force_weights.txt
   lambda_f 1
   lambda_v 0.1

The example coefficients are independent of the table. In
``force_weights.txt`` each row contains the methanol fraction as two integers
separated by ``/``, followed by the raw weights for ``O`` and ``C``:

.. code-block:: text

   # N_C/N_total   O_weight   C_weight
   0/1            1          0
   1/20           1          2
   1/1            0          1

Use the exact ratio of methanol beads to total beads. Include one row for each
ratio in ``train.xyz``. Equivalent fractions match the same ratio, and table
row order is arbitrary. Blank lines and comments starting with ``#`` are
allowed. Weights must be finite and nonnegative, with a positive weight for
at least one species present at that ratio. This keyword is mutually
exclusive with :ref:`kw_type_weight`.

The training force term multiplies each bead's squared residual by its
normalized squared weight. For raw weights :math:`a_O,a_C` and methanol
fraction :math:`x`, the multiplier for type :math:`t` is
:math:`a_t^2/[(1-x)a_O^2+xa_C^2]`. The present species in a pure frame
has multiplier one. Both resident and streaming training use the table.
With a fixed bonded baseline, the prediction entering the residual includes
both the NEP and bonded forces. The reported training and validation force
RMSE and the per-bead predictions and labels retain their original units
and unweighted reporting.
