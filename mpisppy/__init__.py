# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

from pyomo.common.timing import TicTocTimer as _TTT

try:
    import mpi4py.MPI as _mpi
    haveMPI=True
except:
    haveMPI=False

tt_timer = _TTT()

if haveMPI:
    _global_rank = _mpi.COMM_WORLD.Get_rank()
else:
    _global_rank = 0

global_toc = lambda msg, cond=(_global_rank==0) : tt_timer.toc(msg, delta=False) if cond else None
