from pyutilib.misc.timing import TicTocTimer
try:
    import mpi4py.MPI as mpi
    haveMPI=True
except:
    haveMPI=False

tt_timer = TicTocTimer()
if not haveMPI or mpi.COMM_WORLD.Get_rank() == 0:
    tt_timer.toc("Initializing mpi-sppy", delta=False)
