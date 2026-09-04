#!/usr/bin/env python3
"""Minimal mpi4py stress test for many communicators and MPI windows.

Unlike ``mpi4py_xhat_collective_reproducer.py``, this test needs only two or
four ranks.  Every rank duplicates ``MPI_COMM_WORLD`` many times and allocates
one window on each duplicate.  This tests whether MVAPICH's communicator/window
resource handling alone can reproduce the failure seen at larger scale.

Example using two ranks on separate nodes::

    srun -u -N 2 -n 2 --ntasks-per-node=1 \
        python -m mpi4py examples/mpi4py_many_windows_reproducer.py \
        --windows 100 --iterations 100 --mode rma-only

Increase ``--windows`` through 1, 16, 32, 64, and 100 to identify a threshold.
"""

import argparse
import sys
import traceback

import mpi4py
import numpy as np
from mpi4py import MPI


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--payload-length", type=int, default=256)
    parser.add_argument(
        "--mode", choices=("rma-only", "object", "typed"),
        default="rma-only",
    )
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def _allocate_windows(world, count, payload_length):
    padded_length = ((payload_length + 1 + 7) // 8) * 8
    communicators = []
    windows = []
    exposed_arrays = []
    for index in range(count):
        comm = world.Dup()
        window = MPI.Win.Allocate(
            MPI.DOUBLE.Get_size() * padded_length,
            MPI.DOUBLE.Get_size(),
            comm=comm,
        )
        exposed = np.ndarray(
            dtype=np.float64,
            shape=(padded_length,),
            buffer=window.tomemory(),
        )
        exposed.fill(np.nan)
        communicators.append(comm)
        windows.append(window)
        exposed_arrays.append(exposed)
        if world.rank == 0 and (index + 1) % 10 == 0:
            print(f"allocated {index + 1} windows", flush=True)
    return communicators, windows, exposed_arrays, padded_length


def _rma_round(comm, window, iteration, payload_length, padded_length):
    rank = comm.rank
    if rank == 0:
        publish = np.full(padded_length, np.nan, dtype=np.float64)
        publish[:payload_length] = (
            np.arange(payload_length, dtype=np.float64) + iteration)
        publish[payload_length] = iteration + 1
        window.Lock(0, MPI.LOCK_EXCLUSIVE)
        window.Put([publish, MPI.DOUBLE], 0, 0)
        window.Unlock(0)

    comm.Barrier()
    if rank == comm.size - 1:
        received = np.empty(padded_length, dtype=np.float64)
        window.Lock(0, MPI.LOCK_SHARED)
        window.Get([received, MPI.DOUBLE], 0, 0)
        window.Unlock(0)
        expected = np.arange(payload_length, dtype=np.float64) + iteration
        if not np.array_equal(received[:payload_length], expected):
            raise RuntimeError(f"corrupt RMA payload at iteration {iteration}")
        if received[payload_length] != iteration + 1:
            raise RuntimeError(f"corrupt RMA write ID at iteration {iteration}")
    comm.Barrier()


def _collective_round(world, mode, iteration, payload_length):
    if mode == "rma-only":
        return
    root = iteration % world.size
    if mode == "object":
        reports = world.allgather((iteration, root, world.rank == root))
        if any(report[:2] != (iteration, root) for report in reports):
            raise RuntimeError(f"object allgather mismatch at {iteration}")
        value = (np.arange(payload_length, dtype=np.float64) + iteration
                 if world.rank == root else None)
        result = world.bcast(value, root=root)
    else:
        state = np.array([iteration, root], dtype='i')
        reduced = np.empty(2, dtype='i')
        world.Allreduce([state, MPI.INT], [reduced, MPI.INT], op=MPI.MIN)
        result = (np.arange(payload_length, dtype=np.float64) + iteration
                  if world.rank == root
                  else np.empty(payload_length, dtype=np.float64))
        world.Bcast([result, MPI.DOUBLE], root=root)

    expected = np.arange(payload_length, dtype=np.float64) + iteration
    if not np.array_equal(result, expected):
        raise RuntimeError(f"corrupt {mode} collective at {iteration}")


def main():
    args = _arguments()
    world = MPI.COMM_WORLD
    if args.windows <= 0 or args.iterations <= 0 or args.payload_length <= 0:
        raise ValueError("windows, iterations, and payload-length must be positive")
    if world.size < 2:
        raise ValueError("run this reproducer with at least two MPI ranks")

    if world.rank == 0:
        print(
            f"MPI vendor={MPI.get_vendor()!r}; mpi4py={mpi4py.__version__}; "
            f"world={world.size}; windows={args.windows}; mode={args.mode}",
            flush=True,
        )

    try:
        comms, windows, exposed, padded_length = _allocate_windows(
            world, args.windows, args.payload_length)
        for iteration in range(args.iterations):
            index = iteration % args.windows
            _rma_round(
                comms[index], windows[index], iteration,
                args.payload_length, padded_length)
            _collective_round(
                world, args.mode, iteration, args.payload_length)
            if (world.rank == 0 and args.progress_every > 0
                    and (iteration + 1) % args.progress_every == 0):
                print(f"completed {iteration + 1} iterations", flush=True)

        # Keep the NumPy views alive until their corresponding windows are
        # collectively freed.
        assert len(exposed) == len(windows)
        for window in reversed(windows):
            window.Free()
        for comm in reversed(comms):
            comm.Free()
        if world.rank == 0:
            print("PASS", flush=True)
    except BaseException:
        print(f"FAIL rank={world.rank}", file=sys.stderr, flush=True)
        traceback.print_exc()
        world.Abort(1)


if __name__ == "__main__":
    main()
