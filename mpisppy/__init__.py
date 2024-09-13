###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
from pyomo.common.timing import TicTocTimer as _TTT
# Register numpy types in Pyomo, see https://github.com/Pyomo/pyomo/issues/3091
from pyomo.common.dependencies import numpy_available as _np_avail
bool(_np_avail)

from mpisppy.MPI import COMM_WORLD, _haveMPI as haveMPI

tt_timer = _TTT()

_global_rank = COMM_WORLD.rank

global_toc = lambda msg, cond=(_global_rank==0) : tt_timer.toc(msg, delta=False) if cond else None
global_toc("Initializing mpi-sppy")
