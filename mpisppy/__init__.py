from pyutilib.misc.timing import TicTocTimer
import mpi4py.MPI as mpi

tt_timer = TicTocTimer()
if mpi.COMM_WORLD.Get_rank() == 0:
    tt_timer.toc("Initializing mpi-sppy", delta=False)
