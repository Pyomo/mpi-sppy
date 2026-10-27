###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
Test that per-variable rho values flow correctly from Python to C++.

Verifies the three rho-passing fixes:
  1. _push_rho_to_cpp() after pre_iter0() propagates extension rho changes
  2. C++ reads per-scenario CSV files (not --rho-all) so per-variable values
     from rho_setter are preserved
  3. Each scenario reads its own CSV (not a single shared file)

Must be run under MPI with exactly 3 ranks:
    mpirun -np 3 python -m mpi4py aphfw_rho_from_python_driver.py
"""

import sys

import numpy as np
from mpi4py import MPI

from mpisppy.extensions.extension import Extension, MultiExtension
from mpisppy.opt.aphfw import APHFWOpt
from mpisppy.tests.examples.farmer import (
    scenario_creator,
    scenario_denouement,
)
from mpisppy.tests.utils import aphfw_test_solver


# ---------------------------------------------------------------------------
# Test extension that sets distinct per-variable rho in pre_iter0
# ---------------------------------------------------------------------------
class PreIter0RhoSetter(Extension):
    """Sets rho = 10*(i+1) for each nonant i during pre_iter0."""

    def pre_iter0(self):
        for sname, scenario in self.opt.local_scenarios.items():
            for idx, ndn_i in enumerate(scenario._mpisppy_data.nonant_indices):
                scenario._mpisppy_model.rho[ndn_i]._value = 10.0 * (idx + 1)

    def post_iter0(self):
        pass

    def miditer(self):
        pass

    def enditer(self):
        pass

    def post_everything(self):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_options(extensions=None, extension_kwargs=None):
    """Build options dict for APHFWOpt on the farmer problem."""
    num_scens = 3
    all_scenario_names = [f"Scenario{i}" for i in range(num_scens)]
    scenario_creator_kwargs = {"num_scens": num_scens}

    options = {
        "solver_name": aphfw_test_solver(),
        "defaultPHrho": 1.0,
        "convthresh": 0,
        "PHIterLimit": 0,
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
        # aphfw-specific
        "aphfw_fw_tmax": 2,
        "aphfw_fw_tau": 0.01,
        "aphfw_max_cycles": 3,
        "aphfw_primal_term_thresh": 0.0001,
        "aphfw_dual_term_thresh": 0.0001,
        "aphfw_display_progress": False,
        "aphfw_block_sync": True,
        "aphfw_verbosity": 0,
    }

    kwargs = dict(
        all_scenario_names=all_scenario_names,
        scenario_creator=scenario_creator,
        scenario_denouement=scenario_denouement,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )
    if extensions is not None:
        kwargs["extensions"] = extensions
    if extension_kwargs is not None:
        kwargs["extension_kwargs"] = extension_kwargs

    return options, kwargs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_rho_from_python():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 3:
        if rank == 0:
            print(f"ERROR: This test requires exactly 3 MPI ranks, got {size}")
        sys.exit(1)

    passed = True

    # =================================================================
    # Test 1: per-scenario CSV path (no rho_setter, no extensions)
    #
    # With the fix, C++ reads {scenarioName}_rho.csv instead of using
    # --rho-all.  The CSVs are written by Scenario_lp_mps_files with
    # defaultPHrho=1.0.  Verify C++ gets 1.0 for all nonants.
    # =================================================================
    if rank == 0:
        print("\n=== Test 1: per-scenario CSV rho (defaultPHrho=1.0) ===")

    options, kwargs = _make_options()
    opt = APHFWOpt(options, **kwargs)
    opt.aphfw_main(finalize=False)

    n = opt._local_nonant_length
    rho_buf = np.empty(n, dtype=np.float64)
    opt._cpp.get_all_local_rho(rho_buf)

    if rank == 0:
        print(f"  local nonant length: {n}")
        print(f"  rho from C++: {rho_buf}")

    if not np.allclose(rho_buf, 1.0):
        print(f"  [rank {rank}] FAIL: expected rho=1.0 from CSV, got {rho_buf}")
        passed = False
    elif rank == 0:
        print("  PASS: C++ read rho=1.0 from per-scenario CSVs")

    opt.aphfw_finalize()
    comm.Barrier()

    # =================================================================
    # Test 2: pre_iter0 extension sets per-variable rho
    #
    # PreIter0RhoSetter sets rho = [10, 20, 30] during pre_iter0.
    # With the fix, _push_rho_to_cpp() is called after pre_iter0,
    # so C++ should have these values after the first cycle.
    # =================================================================
    if rank == 0:
        print("\n=== Test 2: pre_iter0 rho push (per-variable) ===")

    options2, kwargs2 = _make_options(
        extensions=MultiExtension,
        extension_kwargs={"ext_classes": [PreIter0RhoSetter]},
    )

    opt2 = APHFWOpt(options2, **kwargs2)
    opt2.aphfw_main(finalize=False)

    n2 = opt2._local_nonant_length
    rho_buf2 = np.empty(n2, dtype=np.float64)
    opt2._cpp.get_all_local_rho(rho_buf2)

    # Each scenario has the same nonants, so rho should be [10, 20, 30]
    # repeated for each local scenario.
    nonants_per_scen = n2 // len(list(opt2.local_scenarios.keys()))
    expected_per_scen = np.array([10.0 * (i + 1) for i in range(nonants_per_scen)])
    expected = np.tile(expected_per_scen, len(list(opt2.local_scenarios.keys())))

    if rank == 0:
        print(f"  local nonant length: {n2}")
        print(f"  rho from C++: {rho_buf2}")
        print(f"  expected:     {expected}")

    if not np.allclose(rho_buf2, expected):
        print(f"  [rank {rank}] FAIL: rho mismatch")
        print(f"    got:      {rho_buf2}")
        print(f"    expected: {expected}")
        passed = False
    elif rank == 0:
        print("  PASS: pre_iter0 per-variable rho arrived in C++")

    opt2.aphfw_finalize()
    comm.Barrier()

    # =================================================================
    # Test 3: different defaultPHrho value propagates via CSV
    #
    # Use defaultPHrho=5.5 (not the typical 1.0) and verify C++ sees it.
    # This confirms the CSV path works for arbitrary values.
    # =================================================================
    if rank == 0:
        print("\n=== Test 3: non-default rho via CSV (defaultPHrho=5.5) ===")

    options3, kwargs3 = _make_options()
    options3["defaultPHrho"] = 5.5

    opt3 = APHFWOpt(options3, **kwargs3)
    opt3.aphfw_main(finalize=False)

    n3 = opt3._local_nonant_length
    rho_buf3 = np.empty(n3, dtype=np.float64)
    opt3._cpp.get_all_local_rho(rho_buf3)

    if rank == 0:
        print(f"  rho from C++: {rho_buf3}")

    if not np.allclose(rho_buf3, 5.5):
        print(f"  [rank {rank}] FAIL: expected rho=5.5, got {rho_buf3}")
        passed = False
    elif rank == 0:
        print("  PASS: C++ read rho=5.5 from per-scenario CSVs")

    opt3.aphfw_finalize()
    comm.Barrier()

    # --- Summary ---
    all_passed = comm.allreduce(int(passed), op=MPI.MIN)
    if rank == 0:
        if all_passed:
            print("\nAll rho-from-python tests PASSED")
        else:
            print("\nSome rho-from-python tests FAILED")

    return bool(all_passed)


if __name__ == "__main__":
    result = test_rho_from_python()
    if not result:
        sys.exit(1)
