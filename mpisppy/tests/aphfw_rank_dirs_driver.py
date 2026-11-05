###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
Each hub rank has its own scenario directory holding only its own scenarios,
the C++ still knows the global scenario count (it sets when a cycle ends), and
each rank's directory is removed at the end.

Must be run under MPI with exactly 3 ranks:
    mpirun -np 3 python -m mpi4py aphfw_rank_dirs_driver.py
"""

import os
import tempfile

from mpi4py import MPI

from mpisppy.opt.aphfw import APHFWOpt
from mpisppy.tests.examples.farmer import scenario_creator, scenario_denouement
from mpisppy.tests.utils import aphfw_test_solver

NUM_SCENS = 6  # two per rank
SOLVE_FRACTION = 0.5


def main():
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 3, "this driver needs exactly 3 ranks"

    base = comm.bcast(tempfile.mkdtemp(prefix="aphfw_test_") if comm.Get_rank() == 0
                      else None, root=0)
    options = {
        "solver_name": aphfw_test_solver(),
        "defaultPHrho": 1.0,
        "convthresh": 0,
        "PHIterLimit": 2,
        "time_limit": 600,
        "verbose": False,
        "display_progress": False,
        "display_convergence_detail": False,
        "iter0_solver_options": {},
        "iterk_solver_options": {},
        "tee-rank0-solves": False,
        "trace_prefix": "",
        "presolve": False,
        "rounding_bias": 0.0,
        "warmstart_subproblems": False,
        "user_warmstart": False,
        "aph_block_sync": False,
        "aph_frac_needed": SOLVE_FRACTION,
        "aphfw_verbosity": 0,
        "aphfw_scenario_dir": base,
    }
    opt = APHFWOpt(options, [f"Scenario{i}" for i in range(NUM_SCENS)],
                   scenario_creator, scenario_denouement=scenario_denouement,
                   scenario_creator_kwargs={"num_scens": NUM_SCENS})
    opt.aphfw_main(finalize=False)

    # Counted over all ranks, not from this rank's two scenarios.
    trigger = opt._cpp.get_cycle_trigger()
    assert trigger == round(NUM_SCENS * SOLVE_FRACTION), f"cycle trigger {trigger}"

    # Each rank's directory holds its own scenarios and no one else's.
    scenario_dir = opt._scenario_dir
    assert os.path.dirname(scenario_dir) == base, scenario_dir
    assert len(set(comm.allgather(scenario_dir))) == comm.Get_size()
    stems = {os.path.splitext(f)[0] for f in os.listdir(scenario_dir)
             if f.endswith(".mps")}
    assert stems == set(opt.local_scenarios), f"{stems} in {scenario_dir}"
    assert not [f for f in os.listdir(scenario_dir) if f.endswith(".lp")]

    # finalize removes this rank's directory, whatever the others have done.
    opt.aphfw_finalize()
    assert not os.path.exists(scenario_dir), scenario_dir

    comm.Barrier()
    if comm.Get_rank() == 0:
        assert os.listdir(base) == [], os.listdir(base)
        os.rmdir(base)
        print("PASSED")


if __name__ == "__main__":
    main()
