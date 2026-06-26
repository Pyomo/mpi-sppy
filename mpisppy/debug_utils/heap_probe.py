###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2025, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Phase-boundary heap-integrity probe for the LOR_bug investigation (PR #717).

OFF by default. Set env ``MPISPPY_LOR_HEAP_PROBE=1`` to activate.

When active, ``heap_probe(label)`` forces the glibc allocator to walk its own
heap metadata: a two-pass malloc/free sweep across the tcache size classes plus
some fastbin/smallbin/largebin sizes, followed by ``malloc_trim(0)``. The first
pass frees chunks into the tcache bins; the second pass pulls them back out --
exactly the ``tcache_get`` path that raises "malloc(): unaligned tcache chunk
detected". If the heap is already corrupted, glibc aborts INSIDE the probe
(SIGABRT, no Python traceback), pinning the corrupting write to the phase
between the last printed ``[LOR_bug HEAP PROBE OK]`` marker and this call site.

Even when the probe does not actively abort, the OK markers bracket the
corruption: the last marker before the crash names the surviving phase.

Numpy and MPI buffers live in the same glibc arena this probe walks, so a stomp
into an adjacent chunk's header is visible here. This is debug-only
instrumentation; remove before merging to main.
"""

import os
import sys
import ctypes

_ENABLED = bool(os.environ.get("MPISPPY_LOR_HEAP_PROBE"))

# Per size class, pull a burst large enough to fully drain the tcache bin
# (glibc default depth is 7) and reach into the fastbins/smallbins behind it,
# so a poisoned entry sitting anywhere in the bin is pulled (and tripped) here.
_BURST = 16

_libc = None
_sizes = None


def enabled():
    return _ENABLED


def _init():
    global _libc, _sizes
    if _libc is not None:
        return
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _libc.malloc.restype = ctypes.c_void_p
    _libc.malloc.argtypes = [ctypes.c_size_t]
    _libc.free.argtypes = [ctypes.c_void_p]
    _libc.malloc_trim.argtypes = [ctypes.c_size_t]
    _libc.malloc_trim.restype = ctypes.c_int
    # tcache bins cover ~16..1032 byte requests in 16-byte steps; append a few
    # fastbin/smallbin/largebin sizes so malloc_trim's consolidation walks them.
    _sizes = list(range(16, 1040, 16)) + [1536, 2048, 4096, 8192, 65536, 262144]


def heap_probe(label, rank=None):
    """Walk the glibc heap; abort here if its metadata is already corrupted.

    No-op (just an env check) unless MPISPPY_LOR_HEAP_PROBE is set.
    """
    if not _ENABLED:
        return
    _init()
    # Drain every tcache/fastbin size class by pulling a burst from each (the
    # tcache_get path that raises "unaligned tcache chunk detected" against any
    # poisoned entry), holding them all so the bin actually empties, then free
    # them back and trim. A single alloc/free per size would never reach a
    # poisoned entry behind the bin head, so the burst depth is essential.
    held = []
    for sz in _sizes:
        for _ in range(_BURST):
            held.append(_libc.malloc(sz))
    for p in held:
        _libc.free(p)
    _libc.malloc_trim(0)
    tag = f" rank={rank}" if rank is not None else ""
    print(f"[LOR_bug HEAP PROBE OK] {label}{tag}", flush=True)
    sys.stderr.flush()
