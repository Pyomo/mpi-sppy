# This software is distributed under the 3-clause BSD License.
''' Basic test script to make sure that the one-sided MPI calls work as
    intended. Runs on two processes. Will raise an assertion error if the
    Lock/Get/Unlock combination blocks. Otherwise, exits normally and produces
    no output.

    Just because this test passes doesn't mean that MPI one-sided calls will
    work as expected.

    Takes about 4 seconds to execute, with all the sleep() calls.
'''
import mpi4py.MPI as mpi
import numpy as np
import time

def main():
    assert(mpi.COMM_WORLD.Get_size() == 2)
    rank = mpi.COMM_WORLD.Get_rank()

    array_size = 10
    win = mpi.Win.Allocate(mpi.DOUBLE.size*array_size, mpi.DOUBLE.size, 
                           comm=mpi.COMM_WORLD)
    buff = np.ndarray(buffer=win.tomemory(), dtype='d', shape=(array_size,)) 
     
    if (rank == 0):
        buff[:] = 3. * np.ones(array_size, dtype='d')
        time.sleep(3)
        buff[:] = np.arange(array_size)

        win.Lock(1)
        win.Put((buff, array_size, mpi.DOUBLE), target_rank=1)
        win.Unlock(1)

    elif (rank == 1):
        buff = np.ones(array_size, dtype='d')
        time.sleep(1)

        win.Lock(0)
        win.Get((buff, array_size, mpi.DOUBLE), target_rank=0)
        win.Unlock(0)

        assert(buff[-1] == 3.)

        time.sleep(3)

        win.Lock(1)
        win.Get((buff, array_size, mpi.DOUBLE), target_rank=1)
        win.Unlock(1)

        assert(buff[-1] == array_size-1)

    del buff # Important!
    win.Free()

if __name__=='__main__':
    main()
