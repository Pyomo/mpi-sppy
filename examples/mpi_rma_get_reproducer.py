#!/usr/bin/env python3
"""Four-rank reproducer for an intermittent MVAPICH passive-target RMA hang.

Run one rank on each of four nodes::

    srun -u -N 4 -n 4 --ntasks-per-node=1 \
        python -m mpi4py examples/mpi_rma_get_reproducer.py 100

World ranks 0-1 publish and ranks 2-3 receive. Each publisher/receiver pair
shares a strata communicator; only the receivers synchronize on their cylinder
communicator before each Get.
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

    if size != 4:
        if rank == 0:
            print(f"world size must be four; got {size}", flush=True)
        comm.Abort(2)

    role = rank // 2
    pair = rank % 2
    cylinder_comm = comm.Split(role, pair)
    strata_comm = comm.Split(pair, role)
    strata_rank = strata_comm.Get_rank()

    window = MPI.Win.Allocate(
        MPI.DOUBLE.Get_size(), MPI.DOUBLE.Get_size(), comm=strata_comm)
    if window.Get_attr(MPI.WIN_MODEL) != MPI.WIN_UNIFIED:
        if rank == 0:
            print("this reproducer requires the unified MPI window model",
                  flush=True)
        comm.Abort(2)

    initial = array("d", [float(rank)])
    window.Lock(strata_rank, MPI.LOCK_EXCLUSIVE)
    window.Put([initial, MPI.DOUBLE], strata_rank, 0)
    window.Unlock(strata_rank)

    received = array("d", [0.0])

    comm.Barrier()

    if rank == 0:
        print(f"world={size} iterations={args.iterations} count=1", flush=True)

    for iteration in range(args.iterations):
        if role == 1:
            cylinder_comm.Barrier()
            window.Lock(0, MPI.LOCK_SHARED)
            window.Get([received, MPI.DOUBLE], 0, 0)
            window.Unlock(0)
            print(
                f"world_rank={rank} completed {iteration + 1} iterations",
                flush=True,
            )

    # Keep the target and its window alive until every Get epoch has ended.
    comm.Barrier()

    window.Free()
    strata_comm.Free()
    cylinder_comm.Free()
    if rank == 0:
        print("PASS", flush=True)


if __name__ == "__main__":
    main()
