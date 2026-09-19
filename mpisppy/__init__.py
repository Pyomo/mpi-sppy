###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
from __future__ import annotations  # avoid potential troubles with mpi4py

from pyomo.common.timing import TicTocTimer as _TTT
from pyomo.common.dependencies import numpy_available as _np_avail

from mpisppy.MPI import COMM_WORLD, haveMPI as haveMPI
from mpisppy.utils.mpi_abort import abort_on_uncaught_exception

# Register numpy types in Pyomo, see https://github.com/Pyomo/pyomo/issues/3091
bool(_np_avail)
tt_timer = _TTT()

_global_rank = COMM_WORLD.rank

def global_toc(msg, cond=_global_rank == 0):
    return tt_timer.toc(msg, delta=False) if cond else None
global_toc("Initializing mpi-sppy")

# Installed on import, rather than where a wheel is spun, so that it covers
# the whole of a driver's run: the failures that strike one rank and not the
# others are mostly the early ones -- a scenario file, a per-rank solver
# license -- and the ranks they miss go on to the next collective and block
# there. See mpisppy.utils.mpi_abort; a serial run, and an install without
# mpi4py, are left alone.
abort_on_uncaught_exception()
