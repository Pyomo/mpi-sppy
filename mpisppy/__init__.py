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

# Register numpy types in Pyomo, see https://github.com/Pyomo/pyomo/issues/3091
bool(_np_avail)
tt_timer = _TTT()

_global_rank = COMM_WORLD.rank

def global_toc(msg, cond=_global_rank == 0):
    return tt_timer.toc(msg, delta=False) if cond else None
global_toc("Initializing mpi-sppy")


def git_commit_hash():
    """DEBUG (LOR_bug): short hash of the running source checkout.

    Claude and the cluster experiments run on different machines; printing
    the commit the experiment is actually running removes confusion about
    which version produced a given output. Returns "unknown" outside a git
    checkout (e.g. an installed package). Remove with the LOR_bug
    instrumentation.
    """
    import os
    import subprocess
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "-C", repo_dir, "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"
