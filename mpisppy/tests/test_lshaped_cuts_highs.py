###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# LShapedCutGenerator cuts from the HiGHS interfaces (issue #954). These tests
# require highspy and fail rather than skip without it; CI installs it.

import unittest

import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn

from mpisppy.utils.lshaped_cuts import LShapedCutGenerator


class _LShaped:
    """Just enough of LShapedMethod for LShapedCutGeneratorData."""
    def __init__(self):
        self.all_scenario_names = ["scen0"]
        self.local_scenario_names = ["scen0"]
        self.root = pyo.ConcreteModel()
        self.root.x = pyo.Var(bounds=(0, 10), initialize=1.0)
        self.root.eta = pyo.Var(self.all_scenario_names, initialize=0.0)


class TestHighsCuts(unittest.TestCase):

    def _check_cut(self, solver_name):
        if not pyo.SolverFactory(solver_name).available(exception_flag=False):
            self.fail(f"{solver_name} requires highspy (pip install highspy)")

        ls = _LShaped()
        root = ls.root

        def subproblem_fn(scenario_name):
            # At x <= 5, the recourse cost Q(x) = 2y with y >= 5 - x is
            # 10 - 2x, so the cut must be eta >= 10 - 2x.
            m = pyo.ConcreteModel()
            m.x = pyo.Var(bounds=(0, 10))
            m.y = pyo.Var(bounds=(0, None))
            m.c = pyo.Constraint(expr=m.y >= 5 - m.x)
            m.obj = pyo.Objective(expr=2 * m.y)
            return m, pyo.ComponentMap([(root.x, m.x)])

        root.bender = LShapedCutGenerator()
        root.bender.set_input(root_vars=[root.x], tol=1e-6)
        root.bender.set_ls(ls)
        root.bender.add_subproblem(
            subproblem_fn, {"scenario_name": "scen0"}, root.eta["scen0"],
            subproblem_solver=solver_name)

        cuts = root.bender.generate_cut()
        self.assertEqual(len(cuts), 1)

        # generate_cut scales the cut, so compare after dividing by eta's
        # coefficient: constant + c_eta * eta + c_x * x <= 0, with c_eta < 0.
        repn = generate_standard_repn(cuts[0].body)
        coefs = {v.name: c for v, c in zip(repn.linear_vars, repn.linear_coefs)}
        c_eta = coefs["eta[scen0]"]
        self.assertLess(c_eta, 0)
        self.assertAlmostEqual(repn.constant / -c_eta, 10.0, places=6)
        self.assertAlmostEqual(coefs["x"] / -c_eta, -2.0, places=6)

    def test_appsi_highs(self):
        self._check_cut("appsi_highs")

    def test_highs(self):
        self._check_cut("highs")


if __name__ == "__main__":
    unittest.main()
