.. _kw_composition_descriptor:

``composition_descriptor``
==========================

For a binary liquid represented by one bead per molecule, declare the water
and methanol bead types in ``nep.in``. For example:

.. code-block:: text

   type 2 O C
   composition_descriptor O C

The arguments are in water, methanol order. The ``type`` line can list the
same symbols in either order. For each configuration, NEP appends
:math:`N_\mathrm{methanol}/(N_\mathrm{water}+N_\mathrm{methanol})` to every bead's
ANN input, where both counts come from the actual species in that configuration.
A pure water or pure methanol configuration uses 0 or 1, respectively.
The scaler of this input is fixed at 1. The composition influences force and
virial through the ANN's derivatives of the local descriptors.

Omit the keyword to train an ordinary NEP4 model. The keyword is independent
of :ref:`kw_composition_force_weights`. It supports one-hidden-layer potential
NEP4 with resident or streaming training and the ordinary or specialized
training kernels.

The model stores the mapping in its header. Standalone prediction uses that
mapping and checks any explicit ``composition_descriptor`` in ``nep.in``.
For a fixed-composition single-GPU MD run, use the saved model with
``potential``. Both the mixed and pure configurations use the full binary
model. Multi-GPU MD, MCMD, and ``fine_tune`` are unavailable for this model
format.

Resume training with the matching ``nep.txt`` and ``nep.restart`` pair. A
checkpoint records the ordered types, composition mapping and ANN layout.
On resume, the matching model supplies the saved descriptor scalers.
