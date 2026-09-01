#!/usr/bin/env python3
"""Two-rank reproducer for an intermittent MVAPICH passive-target RMA hang.

Run one rank on each of two nodes::

    srun -u -N 2 -n 2 --ntasks-per-node=1 \
        python -m mpi4py examples/mpi_rma_get_reproducer.py 100
"""

import argparse
from array import array

from mpi4py import MPI


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iterations", nargs="?", type=int, default=100)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    return args


def main():
    args = _parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            print(f"world size must be two; got {size}", flush=True)
        comm.Abort(2)

    window = MPI.Win.Allocate(
        MPI.DOUBLE.Get_size(), MPI.DOUBLE.Get_size(), comm=comm)
    if window.Get_attr(MPI.WIN_MODEL) != MPI.WIN_UNIFIED:
        if rank == 0:
            print("this reproducer requires the unified MPI window model",
                  flush=True)
        comm.Abort(2)

    exposed = window.tomemory().cast("d")
    exposed[0] = float(rank)
    received = array("d", [0.0])

    comm.Barrier()

    if rank == 0:
        print(f"world={size} iterations={args.iterations} count=1", flush=True)

    for iteration in range(args.iterations):
        if rank == 1:
            window.Lock(0, MPI.LOCK_SHARED)
            window.Get([received, MPI.DOUBLE], 0, 0)
            window.Unlock(0)

        comm.Barrier()

        if rank == 1:
            print(f"completed {iteration + 1} iterations", flush=True)

    del exposed
    window.Free()
    if rank == 0:
        print("PASS", flush=True)


if __name__ == "__main__":
    main()
