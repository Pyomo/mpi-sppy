###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import unittest

import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables

from mpisppy.opt.lshaped import LShapedMethod
from mpisppy.utils.lshaped_cuts import StandardLPL1CutGenerator, solver_dual_sign_convention
from mpisppy.tests.examples import farmer
from mpisppy.tests.utils import get_solver
import mpisppy.utils.sputils as sputils


solver_available, solver_name, _, _ = get_solver()
standard_lp_l1_solver_available = (
    solver_available
    and solver_name in solver_dual_sign_convention
)


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

    def test_l1_clone_drops_stale_dual_suffix_values(self):
        m = pyo.ConcreteModel()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.con = pyo.Constraint(expr=m.y >= 0)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.dual[m.con] = 1.0

        gen = StandardLPL1CutGenerator()
        clone = gen._clone_subproblem_for_l1(m)

        self.assertTrue(hasattr(m, "dual"))
        self.assertTrue(hasattr(clone, "dual"))
        self.assertEqual(len(m.dual), 0)
        self.assertEqual(len(clone.dual), 0)

    def test_unknown_lshaped_cut_generator_option_errors(self):
        names = farmer.scenario_names_creator(3)
        options = {
            "root_solver": solver_name,
            "sp_solver": solver_name,
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

    def test_ambiguous_recourse_status_with_zero_l1_violation_errors(self):
        root = pyo.ConcreteModel()
        root.x = pyo.Var(initialize=0.0)
        root.eta = pyo.Var(initialize=0.0)
        subproblem = pyo.ConcreteModel()
        subproblem.x = pyo.Var()
        subproblem.obj = pyo.Objective(expr=0.0)

        class _Result:
            class solver:
                termination_condition = pyo.TerminationCondition.infeasibleOrUnbounded

        gen = StandardLPL1CutGenerator()
        gen.root_vars = [root.x]
        gen.tol = 1e-6
        gen.subproblems = [subproblem]
        gen.complicating_vars_maps = [pyo.ComponentMap([(root.x, subproblem.x)])]
        gen.subproblem_solvers = [object()]
        gen.subproblem_solver_names = ["cbc"]
        gen._solve_model = lambda *args: _Result()
        gen._solve_l1_feasibility = lambda *args: {
            "constant": 0.0,
            "coefficients": [0.0],
            "needs_cut": False,
            "infeasible": True,
        }

        with self.assertRaisesRegex(RuntimeError, "may be unbounded"):
            gen._solve_recourse_or_l1(0, root.eta)


class TestStandardLPL1LShapedSolve(unittest.TestCase):
    @unittest.skipUnless(
        standard_lp_l1_solver_available,
        "%s solver is not available or is not supported by standard_lp_l1"
        % (solver_name,),
    )
    def test_farmer_lshaped_standard_lp_l1(self):
        names = farmer.scenario_names_creator(3)
        options = {
            "root_solver": solver_name,
            "sp_solver": solver_name,
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

    @unittest.skipUnless(
        standard_lp_l1_solver_available,
        "%s solver is not available or is not supported by standard_lp_l1"
        % (solver_name,),
    )
    def test_infeasible_recourse_generates_l1_feasibility_cuts(self):
        names = ["low", "high"]
        demand = {"low": 5.0, "high": 7.0}

        def scenario_creator(scenario_name):
            model = pyo.ConcreteModel(name=scenario_name)
            model.capacity = pyo.Var(bounds=(0.0, 10.0))
            model.emergency_supply = pyo.Var(bounds=(0.0, 1.0))
            model.meet_demand = pyo.Constraint(
                expr=(
                    model.capacity + model.emergency_supply
                    >= demand[scenario_name]
                )
            )
            model.total_cost = pyo.Objective(
                expr=model.capacity + 0.1 * model.emergency_supply
            )
            model._mpisppy_probability = 1.0 / len(names)
            sputils.attach_root_node(model, model.capacity, [model.capacity])
            return model

        options = {
            "root_solver": solver_name,
            "sp_solver": solver_name,
            "sp_solver_options": {},
            "valid_eta_lb": {name: 0.0 for name in names},
            "lshaped_cut_generator": "standard_lp_l1",
            "max_iter": 10,
            "verbose": False,
        }
        ls = LShapedMethod(
            options,
            names,
            scenario_creator,
        )

        ls.lshaped_algorithm()

        capacity = ls.root.capacity
        cuts = list(ls.root._standard_lshaped_l1_cuts.values())
        eta_ids = {id(eta) for eta in ls.root.eta.values()}
        feasibility_cuts = [
            cut for cut in cuts
            if eta_ids.isdisjoint(id(var) for var in identify_variables(cut.body))
        ]

        # The initial master picks capacity=0.  Each scenario rejects it with
        # an eta-free feasibility cut (capacity >= 4 and capacity >= 6).
        self.assertEqual(len(feasibility_cuts), 2)
        final_capacity = pyo.value(capacity)
        capacity.set_value(0.0)
        for cut in feasibility_cuts:
            self.assertGreater(pyo.value(cut.body - cut.upper), 1e-7)

        # Both cuts accept the converged solution, where the high-demand
        # scenario is binding.  One subsequent optimality cut prices its
        # emergency-supply recourse.
        capacity.set_value(final_capacity)
        for cut in feasibility_cuts:
            self.assertLessEqual(pyo.value(cut.body - cut.upper), 1e-7)
        self.assertEqual(len(cuts), 3)
        self.assertAlmostEqual(final_capacity, 6.0, places=6)
        self.assertAlmostEqual(ls._LShaped_bound, 6.05, places=6)


if __name__ == "__main__":
    unittest.main()
