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
Descriptor and gradient scratch space remains resident. For NEP, its atom and
configuration capacity is obtained from the CPU-side batch metadata; the
q-scaler traversal still visits every batch and records the global descriptor
range and maximum neighbor counts. Validation datasets also remain resident,
so a large validation set can still be an independent source of GPU memory
use.

NEP streaming remains synchronous, but reuses the current Dataset's GPU
allocation capacity when the next batch is loaded. The old batch is no longer
logically live after a generation, while the allocation stays cached for the
next batch. This avoids repeated ``cudaFree``/``cudaMalloc`` calls and bounds
the cache by the largest batch seen. Set ``GPUMD_STREAM_TELEMETRY=1`` to print
synchronized ``STREAM_MEMORY`` diagnostics; telemetry is disabled by default
because each sample synchronizes the device. Set ``GPUMD_STREAM_VERBOSE=1``
to restore per-batch Dataset and neighbor statistics. GNEP retains the
original synchronous streaming lifecycle.
