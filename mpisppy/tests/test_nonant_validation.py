###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for nonant name validation in spbase.py."""
import unittest
import pyomo.environ as pyo

import mpisppy.opt.ef
from mpisppy.tests.utils import get_solver
from examples.farmer import farmer, bad_farmer

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()


class TestNonantNameValidation(unittest.TestCase):

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_good_farmer_ef(self):
        """Normal farmer should pass validation and solve fine."""
        options = {"solver": solver_name}
        all_scenario_names = farmer.scenario_names_creator(3)
        ef = mpisppy.opt.ef.ExtensiveForm(
            options,
            all_scenario_names,
            farmer.scenario_creator,
            scenario_creator_kwargs={"num_scens": 3},
        )
        results = ef.solve_extensive_form(tee=False)
        pyo.assert_optimal_termination(results)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_bad_farmer_ef_catches_mismatch(self):
        """bad_farmer reverses nonants for scen2; validation should catch it."""
        options = {"solver": solver_name}
        all_scenario_names = bad_farmer.scenario_names_creator(3)
        with self.assertRaises(RuntimeError) as ctx:
            mpisppy.opt.ef.ExtensiveForm(
                options,
                all_scenario_names,
                bad_farmer.scenario_creator,
                scenario_creator_kwargs={"num_scens": 3},
            )
        self.assertIn("nonant variable name mismatch", str(ctx.exception))

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_bad_farmer_with_check_disabled(self):
        """bad_farmer with turn_off_names_check should not raise."""
        options = {"solver": solver_name, "turn_off_names_check": True}
        all_scenario_names = bad_farmer.scenario_names_creator(3)
        mpisppy.opt.ef.ExtensiveForm(
            options,
            all_scenario_names,
            bad_farmer.scenario_creator,
            scenario_creator_kwargs={"num_scens": 3},
        )
        # It will construct, though the solution would be wrong


if __name__ == "__main__":
    unittest.main()
