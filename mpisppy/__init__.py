# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import os
from pyutilib.misc.timing import TicTocTimer
try:
    import mpi4py.MPI as mpi
    haveMPI=True
except:
    haveMPI=False

tt_timer = TicTocTimer()

if haveMPI:
    global_rank = mpi.COMM_WORLD.Get_rank()
else:
    global_rank = 0

global_toc = lambda msg, cond=(global_rank==0) : tt_timer.toc(msg, delta=False) if cond else None
