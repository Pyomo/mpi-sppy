#!/usr/bin/env python3
"""Standalone stress test for the collectives used by XhatShuffle.

This script imports only mpi4py, NumPy, and the Python standard library.  Its
default communicator layout matches an mpi-sppy run with four equal cylinders:
global ranks 3, 7, 11, ... form the Xhat communicator.  With 400 world ranks,
that communicator therefore contains 100 ranks.

Standards-compliant object-collective stress test::

    srun -u -n 400 python -m mpi4py \
        examples/mpi4py_xhat_collective_reproducer.py \
        --mode object --iterations 10000

Add ``--with-rma`` to create the same four-rank strata communicators and an
MPI window on each one.  Before every Xhat collective, the hub rank publishes
a padded buffer to its window and the corresponding Xhat rank retrieves it
with ``Win.Get``.  Strata barriers make this mode standards-compliant and keep
the reproducer deterministic.

Typed-collective control::

    srun -u -n 400 python -m mpi4py \
        examples/mpi4py_xhat_collective_reproducer.py \
        --mode typed --iterations 10000

The ``invalid-mismatch`` mode deliberately violates MPI collective-ordering
rules.  It is useful only for determining whether this MPI stack produces the
same ``Negative size passed to PyBytes_FromStringAndSize``/``MemoryError``
signature when ranks are out of phase.  A failure in that mode is expected and
is not evidence of an MPI bug.
"""

import argparse
import sys
import traceback

import numpy as np
import mpi4py
from mpi4py import MPI


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("object", "typed", "invalid-mismatch"),
        default="object",
    )
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--payload-length", type=int, default=256)
    parser.add_argument("--cylinders", type=int, default=4)
    parser.add_argument("--xhat-index", type=int, default=3)
    parser.add_argument(
        "--mismatch-iteration", type=int, default=10,
        help="Iteration at which invalid-mismatch deliberately desynchronizes.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--with-rma", action="store_true",
        help="Exercise Win.Allocate/Put/Get on the four-rank strata comms.",
    )
    return parser.parse_args()


def _object_iteration(comm, iteration, payload_length, root_value=None):
    """Reproduce the object allgather + object bcast used by 29f9f71b."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = (57 + 37 * iteration) % size
    scenario = f"scenario_{root:06d}"
    if rank == root:
        value = (np.arange(payload_length, dtype=np.float64) + iteration
                 if root_value is None else root_value)
    else:
        value = None

    reports = comm.allgather((scenario, root, value is not None))
    selections = {(report[0], report[1]) for report in reports}
    if selections != {(scenario, root)}:
        raise RuntimeError(
            f"rank {rank}: selection disagreement at iteration {iteration}: "
            f"{selections!r}"
        )
    if not reports[root][2]:
        raise RuntimeError(
            f"rank {rank}: root {root} reported no payload at iteration "
            f"{iteration}"
        )

    result = comm.bcast(value, root=root)
    expected = np.arange(payload_length, dtype=np.float64) + iteration
    if not isinstance(result, np.ndarray) or not np.array_equal(result, expected):
        raise RuntimeError(
            f"rank {rank}: corrupt object broadcast at iteration {iteration}"
        )


def _typed_iteration(comm, iteration, payload_length, root_value=None):
    """Equivalent operation using fixed-size buffers and typed collectives."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = (57 + 37 * iteration) % size

    local_selection = np.array([root, root], dtype='i')
    min_selection = np.empty(2, dtype='i')
    max_selection = np.empty(2, dtype='i')
    comm.Allreduce([local_selection, MPI.INT],
                   [min_selection, MPI.INT], op=MPI.MIN)
    comm.Allreduce([local_selection, MPI.INT],
                   [max_selection, MPI.INT], op=MPI.MAX)
    if not np.array_equal(min_selection, max_selection):
        raise RuntimeError(
            f"rank {rank}: typed selection disagreement at iteration {iteration}"
        )

    if rank == root:
        result = (np.arange(payload_length, dtype=np.float64) + iteration
                  if root_value is None else root_value.copy())
    else:
        result = np.empty(payload_length, dtype=np.float64)
    comm.Bcast([result, MPI.DOUBLE], root=root)

    expected = np.arange(payload_length, dtype=np.float64) + iteration
    if not np.array_equal(result, expected):
        raise RuntimeError(
            f"rank {rank}: corrupt typed broadcast at iteration {iteration}"
        )


def _invalid_mismatch(comm, iteration):
    """Deliberately match an object allgather against an integer Allreduce."""
    rank = comm.Get_rank()
    if rank % 2 == 0:
        comm.allgather((f"scenario_{iteration}", 57, False))
    else:
        send = np.array([iteration], dtype='i')
        recv = np.empty(1, dtype='i')
        comm.Allreduce([send, MPI.INT], [recv, MPI.INT], op=MPI.MIN)


def _make_rma_window(strata_comm, payload_length):
    """Allocate a padded double window like SPWindow's equal-rank path."""
    logical_length = payload_length + 1  # payload plus trailing write ID
    padded_length = ((logical_length + 7) // 8) * 8
    window = MPI.Win.Allocate(
        MPI.DOUBLE.Get_size() * padded_length,
        MPI.DOUBLE.Get_size(),
        comm=strata_comm,
    )
    exposed = np.ndarray(
        dtype=np.float64,
        shape=(padded_length,),
        buffer=window.tomemory(),
    )
    exposed.fill(np.nan)
    exposed[payload_length] = 0.0
    strata_comm.Barrier()
    return window, padded_length


def _rma_iteration(window, strata_comm, cylinder_index, xhat_index,
                   iteration, payload_length, padded_length):
    """Publish on the hub rank and retrieve on its paired Xhat rank."""
    if cylinder_index == 0:
        publish = np.full(padded_length, np.nan, dtype=np.float64)
        publish[:payload_length] = (
            np.arange(payload_length, dtype=np.float64) + iteration)
        publish[payload_length] = iteration + 1
        window.Lock(0, MPI.LOCK_EXCLUSIVE)
        window.Put([publish, MPI.DOUBLE], 0, 0)
        window.Unlock(0)

    # This deliberately makes the RMA test deterministic.  A later stress
    # variant can remove this barrier and use write-ID agreement to reject
    # reads that straddle publications.
    strata_comm.Barrier()

    received = None
    if cylinder_index == xhat_index:
        received = np.empty(padded_length, dtype=np.float64)
        window.Lock(0, MPI.LOCK_SHARED)
        window.Get([received, MPI.DOUBLE], 0, 0)
        window.Unlock(0)

        expected = np.arange(payload_length, dtype=np.float64) + iteration
        if not np.array_equal(received[:payload_length], expected):
            raise RuntimeError(f"corrupt RMA payload at iteration {iteration}")
        if received[payload_length] != iteration + 1:
            raise RuntimeError(f"corrupt RMA write ID at iteration {iteration}")

    strata_comm.Barrier()
    return None if received is None else received[:payload_length].copy()


def main():
    args = _parse_args()
    world = MPI.COMM_WORLD
    world_rank = world.Get_rank()
    world_size = world.Get_size()

    if args.iterations <= 0 or args.payload_length <= 0:
        raise ValueError("iterations and payload-length must be positive")
    if args.cylinders <= 0 or world_size % args.cylinders:
        raise ValueError(
            f"world size {world_size} must be divisible by --cylinders "
            f"{args.cylinders}"
        )
    if not 0 <= args.xhat_index < args.cylinders:
        raise ValueError("xhat-index must be in [0, cylinders)")

    cylinder_index = world_rank % args.cylinders
    color = 0 if cylinder_index == args.xhat_index else MPI.UNDEFINED
    xhat_comm = world.Split(color=color, key=world_rank)

    strata_comm = None
    rma_window = None
    padded_length = None
    if args.with_rma:
        strata_comm = world.Split(
            color=world_rank // args.cylinders, key=world_rank)
        if strata_comm.Get_rank() != cylinder_index:
            raise RuntimeError("unexpected strata communicator rank ordering")
        rma_window, padded_length = _make_rma_window(
            strata_comm, args.payload_length)

    if world_rank == 0:
        print(
            f"MPI vendor={MPI.get_vendor()!r}; mpi4py={mpi4py.__version__}; "
            f"MPI standard={MPI.Get_version()!r}; "
            f"world={world_size}; mode={args.mode}; with_rma={args.with_rma}",
            flush=True,
        )

    try:
        if xhat_comm != MPI.COMM_NULL:
            xhat_rank = xhat_comm.Get_rank()
            if xhat_rank == 0:
                print(
                    f"Xhat communicator size={xhat_comm.Get_size()}; "
                    f"iterations={args.iterations}; "
                    f"payload_length={args.payload_length}",
                    flush=True,
                )

        for iteration in range(args.iterations):
            root_value = None
            if args.with_rma:
                root_value = _rma_iteration(
                    rma_window,
                    strata_comm,
                    cylinder_index,
                    args.xhat_index,
                    iteration,
                    args.payload_length,
                    padded_length,
                )

            if xhat_comm != MPI.COMM_NULL:
                if (args.mode == "invalid-mismatch"
                        and iteration == args.mismatch_iteration):
                    _invalid_mismatch(xhat_comm, iteration)
                elif args.mode == "typed":
                    _typed_iteration(
                        xhat_comm, iteration, args.payload_length, root_value)
                else:
                    _object_iteration(
                        xhat_comm, iteration, args.payload_length, root_value)

                if (xhat_rank == 0 and args.progress_every > 0
                        and (iteration + 1) % args.progress_every == 0):
                    print(f"completed {iteration + 1} iterations", flush=True)

        if rma_window is not None:
            rma_window.Free()
            strata_comm.Free()
        if xhat_comm != MPI.COMM_NULL:
            xhat_comm.Free()

        world.Barrier()
        if world_rank == 0:
            print("PASS", flush=True)
    except BaseException:
        print(
            f"FAIL world_rank={world_rank}, cylinder_index={cylinder_index}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        world.Abort(1)


if __name__ == "__main__":
    main()
