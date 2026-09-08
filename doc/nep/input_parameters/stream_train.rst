.. _kw_stream_train:

   single: stream_train (keyword in nep.in or gnep.in)

``stream_train``
================

This keyword controls the lifetime of training datasets stored on GPUs::

  stream_train <flag>

``flag`` must be ``0`` or ``1`` and defaults to ``0``. With ``0``, all
training batches use the original V5.8.1 resident-dataset path. With ``1``,
only the current training batch (or its per-device GNEP shards) resides on
the GPUs. The parsed structures and batch metadata remain on the CPU.

Streaming does not change NEP batch scheduling, fitness weights, energy
shifts, or SNES updates. It does not change GNEP shuffling, gradient
reduction, Adam updates, epoch boundaries, or learning-rate scheduling.
Descriptor and gradient scratch space remains resident and is sized from a
capacity scan of all batches. Validation datasets also remain resident, so a
large validation set can still be an independent source of GPU memory use.

Streaming is synchronous. Each current batch is destroyed before the next
one is loaded. Lines beginning with ``STREAM_MEMORY`` report synchronized
CUDA memory and the number of live training datasets at load, compute, and
destroy boundaries.
