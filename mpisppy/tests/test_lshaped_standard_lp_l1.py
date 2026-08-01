###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import unittest
import os

import pyomo.environ as pyo

from mpisppy.opt.lshaped import LShapedMethod
from mpisppy.utils.lshaped_cuts import StandardLPL1CutGenerator
from mpisppy.tests.examples import farmer


XPRESS_SOLVER = os.environ.get("MPISPPY_XPRESS_TEST_SOLVER", "xpress_persistent")
RUN_XPRESS_TESTS = os.environ.get("MPISPPY_RUN_XPRESS_TESTS") == "1"


class TestStandardLPL1CutGenerator(unittest.TestCase):
    def test_l1_transform_does_not_add_objective_row(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=1000 * m.y)
        m.eq = pyo.Constraint(expr=m.y == m.x + 1)

        gen = StandardLPL1CutGenerator()
        gen._build_l1_model(m)

        self.assertFalse(m.obj.active)
        self.assertTrue(m._mpisppy_l1_obj.active)
        # One equality becomes two relaxed rows. The objective is not converted
        # into an additional eta/violation row in the standard_lp_l1 path.
        self.assertEqual(len(m._mpisppy_l1_cons), 2)

    def test_validate_rejects_quadratic_objective(self):
        m = pyo.ConcreteModel()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y * m.y)
        m.con = pyo.Constraint(expr=m.y >= 0)

        gen = StandardLPL1CutGenerator()
        with self.assertRaisesRegex(ValueError, "linear subproblem objective"):
            gen._validate_linear_subproblem(m)

    def test_validate_rejects_quadratic_constraint(self):
        m = pyo.ConcreteModel()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.con = pyo.Constraint(expr=m.y * m.y <= 1)

        gen = StandardLPL1CutGenerator()
        with self.assertRaisesRegex(ValueError, "linear subproblem constraints"):
            gen._validate_linear_subproblem(m)

    def test_unknown_lshaped_cut_generator_option_errors(self):
        names = farmer.scenario_names_creator(3)
        options = {
            "root_solver": "xpress_persistent",
            "sp_solver": "xpress_persistent",
            "sp_solver_options": {},
            "valid_eta_lb": {name: -1e6 for name in names},
            "lshaped_cut_generator": "not_a_generator",
        }
        ls = LShapedMethod(
            options,
            names,
            farmer.scenario_creator,
            scenario_creator_kwargs={"num_scens": 3},
        )

        with self.assertRaisesRegex(ValueError, "Unknown lshaped_cut_generator"):
            ls.lshaped_algorithm()

    def test_standard_lp_l1_rejects_unsupported_solver(self):
        gen = StandardLPL1CutGenerator()
        with self.assertRaisesRegex(NotImplementedError, "currently supports xpress"):
            gen._solver_sign("not_a_solver")


class TestStandardLPL1LShapedSolve(unittest.TestCase):
    @unittest.skipUnless(
        RUN_XPRESS_TESTS,
        "set MPISPPY_RUN_XPRESS_TESTS=1 to run xpress-dependent tests",
    )
    def test_farmer_lshaped_standard_lp_l1(self):
        names = farmer.scenario_names_creator(3)
        options = {
            "root_solver": XPRESS_SOLVER,
            "sp_solver": XPRESS_SOLVER,
            "sp_solver_options": {},
            "valid_eta_lb": {name: -1e6 for name in names},
            "lshaped_cut_generator": "standard_lp_l1",
            "max_iter": 20,
            "verbose": False,
        }
        ls = LShapedMethod(
            options,
            names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs={"num_scens": 3},
        )

        ls.lshaped_algorithm()

        self.assertAlmostEqual(ls._LShaped_bound, -108390.0, delta=5.0)


if __name__ == "__main__":
    unittest.main()
