###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for SMPS reader using the sizes example.

The sizes SMPS instance (examples/sizes/SMPS) has 10 scenarios each
with probability 0.1.  The parsing tests use all 10 scenarios, but
the EF solve test uses only 3 to stay within CPLEX Community Edition
problem size limits.  The 3-scenario subset has probabilities that
don't sum to 1.0, but the EF weights by given probabilities so the
solve is still valid for testing the read-solve pipeline.
"""
import unittest
import os
import pyomo.environ as pyo
from pyomo.common.config import ConfigDict, ConfigValue

import mpisppy.problem_io.smps_reader as smps_reader
import mpisppy.problem_io.smps_module as smps_module
import mpisppy.opt.ef
from mpisppy.tests.utils import get_solver, round_pos_sig

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()

# Path to the sizes SMPS example
_this_dir = os.path.dirname(os.path.abspath(__file__))
_sizes_smps_dir = os.path.join(
    _this_dir, "..", "..", "examples", "sizes", "SMPS")


class TestSmpsReader(unittest.TestCase):
    """Test the low-level SMPS parsing functions."""

    def test_find_smps_files(self):
        cor, tim, sto = smps_reader._find_smps_files(_sizes_smps_dir)
        self.assertTrue(cor.endswith(".cor"))
        self.assertTrue(tim.endswith(".tim"))
        self.assertTrue(sto.endswith(".sto"))

    def test_parse_tim(self):
        _, tim_path, _ = smps_reader._find_smps_files(_sizes_smps_dir)
        stages = smps_reader.parse_tim(tim_path)
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0][0], "ROOT")
        self.assertEqual(stages[1][0], "STAGE-2")
        # Check first var of each stage
        self.assertEqual(stages[0][1], "Z01JJ01")
        self.assertEqual(stages[1][1], "Z01JJ02")

    def test_parse_sto(self):
        _, _, sto_path = smps_reader._find_smps_files(_sizes_smps_dir)
        scenarios = smps_reader.parse_sto_discrete(sto_path)
        self.assertEqual(len(scenarios), 10)
        # Check first scenario
        s1 = scenarios[0]
        self.assertEqual(s1["name"], "SCEN01")
        self.assertAlmostEqual(s1["probability"], 0.1)
        self.assertEqual(s1["parent"], "ROOT")
        self.assertEqual(s1["stage"], "STAGE-2")
        self.assertEqual(len(s1["modifications"]), 10)

    def test_var_order(self):
        cor_path, _, _ = smps_reader._find_smps_files(_sizes_smps_dir)
        var_order = smps_reader.get_var_order_from_mps(cor_path)
        self.assertGreater(len(var_order), 0)
        # First var should be Z01JJ01 (first column in the MPS file)
        self.assertEqual(var_order[0], "Z01JJ01")

    def test_partition_vars(self):
        cor_path, tim_path, _ = smps_reader._find_smps_files(_sizes_smps_dir)
        stages = smps_reader.parse_tim(tim_path)
        var_order = smps_reader.get_var_order_from_mps(cor_path)
        vars_by_stage = smps_reader.partition_vars_by_stage(var_order, stages)
        self.assertIn("ROOT", vars_by_stage)
        self.assertIn("STAGE-2", vars_by_stage)
        # ROOT vars should include Z01JJ01 but not Z01JJ02
        self.assertIn("Z01JJ01", vars_by_stage["ROOT"])
        self.assertNotIn("Z01JJ02", vars_by_stage["ROOT"])
        # STAGE-2 vars should include Z01JJ02
        self.assertIn("Z01JJ02", vars_by_stage["STAGE-2"])


class TestSmpsModule(unittest.TestCase):
    """Test the SMPS module interface."""

    def _make_cfg(self):
        """Create a minimal Config-like object."""
        cfg = ConfigDict()
        cfg.declare("smps_dir", ConfigValue(default=None, domain=str))
        cfg.smps_dir = _sizes_smps_dir
        if solver_available:
            cfg.declare("solver_name", ConfigValue(default=solver_name, domain=str))
        return cfg

    def test_scenario_names(self):
        cfg = self._make_cfg()
        smps_module.kw_creator(cfg)
        names = smps_module.scenario_names_creator(num_scens=None)
        self.assertEqual(len(names), 10)
        self.assertEqual(names[0], "SCEN01")
        self.assertEqual(names[9], "SCEN10")

    def test_scenario_names_subset(self):
        cfg = self._make_cfg()
        smps_module.kw_creator(cfg)
        names = smps_module.scenario_names_creator(num_scens=3, start=2)
        self.assertEqual(len(names), 3)
        self.assertEqual(names[0], "SCEN03")

    def test_scenario_creator(self):
        cfg = self._make_cfg()
        smps_module.kw_creator(cfg)
        model = smps_module.scenario_creator("SCEN01", cfg=cfg)
        # Check mpi-sppy annotations
        self.assertTrue(hasattr(model, "_mpisppy_probability"))
        self.assertAlmostEqual(model._mpisppy_probability, 0.1)
        self.assertTrue(hasattr(model, "_mpisppy_node_list"))
        # 2-stage: only ROOT node in the node list
        self.assertEqual(len(model._mpisppy_node_list), 1)
        root_node = model._mpisppy_node_list[0]
        self.assertEqual(root_node.name, "ROOT")
        self.assertGreater(len(root_node.nonant_list), 0)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_ef_solve(self):
        """Solve EF on the sizes SMPS instance and check the objective."""
        cfg = self._make_cfg()
        smps_module.kw_creator(cfg)
        # Use 3 scenarios to stay within CPLEX Community Edition limits
        scenario_names = smps_module.scenario_names_creator(num_scens=3)

        options = {"solver": solver_name}
        ef = mpisppy.opt.ef.ExtensiveForm(
            options,
            scenario_names,
            smps_module.scenario_creator,
            scenario_creator_kwargs={"cfg": cfg},
        )
        results = ef.solve_extensive_form(tee=False)
        pyo.assert_optimal_termination(results)
        objval = pyo.value(ef.ef.EF_Obj)
        # Known optimal for sizes with 3 scenarios is ~179765.
        sig2obj = round_pos_sig(objval, 2)
        self.assertEqual(sig2obj, 180000.0)


class TestSmpsModifications(unittest.TestCase):
    """Test coefficient and bounds modifications using a tiny synthetic model.

    The synthetic SMPS instance (tests/examples/smps_synthetic) has two
    variables (X, Y), two constraints (C1, C2), and two scenarios:
      SCEN1: RHS modification only (C2 RHS -> 30)
      SCEN2: coefficient change (Y in C2 -> 4), upper bound on X -> 50,
             lower bound on Y -> 8
    """
    _synth_dir = os.path.join(_this_dir, "examples", "smps_synthetic")

    def setUp(self):
        """Reset the cached parse state so each test starts clean."""
        smps_module._parsed = None

    def _make_cfg(self):
        cfg = ConfigDict()
        cfg.declare("smps_dir", ConfigValue(default=None, domain=str))
        cfg.smps_dir = self._synth_dir
        if solver_available:
            cfg.declare("solver_name", ConfigValue(default=solver_name, domain=str))
        return cfg

    def test_bounds_modification_up(self):
        """Upper-bound modification (X has UP bound in .cor)."""
        cfg = self._make_cfg()
        smps_module.kw_creator(cfg)
        model = smps_module.scenario_creator("SCEN2", cfg=cfg)
        x = model.find_component("X")
        self.assertEqual(x.ub, 50.0)

    def test_bounds_modification_lo(self):
        """Lower-bound modification (Y has LO bound in .cor)."""
        cfg = self._make_cfg()
        smps_module.kw_creator(cfg)
        model = smps_module.scenario_creator("SCEN2", cfg=cfg)
        y = model.find_component("Y")
        self.assertEqual(y.lb, 8.0)

    def test_coefficient_modification(self):
        """Coefficient of Y in C2 changes from 3 to 4 in SCEN2."""
        cfg = self._make_cfg()
        smps_module.kw_creator(cfg)
        model = smps_module.scenario_creator("SCEN2", cfg=cfg)
        # Check that the constraint C2 body includes Y with coefficient 4
        c2 = model.find_component("C2")
        body = c2.body
        y = model.find_component("Y")
        # Evaluate body with X=0, Y=1 to extract Y's coefficient
        x = model.find_component("X")
        x.value = 0.0
        y.value = 1.0
        coeff_y = pyo.value(body)
        self.assertAlmostEqual(coeff_y, 4.0)

    def test_rhs_modification(self):
        """RHS of C2 changes to 30 in SCEN1."""
        cfg = self._make_cfg()
        smps_module.kw_creator(cfg)
        model = smps_module.scenario_creator("SCEN1", cfg=cfg)
        c2 = model.find_component("C2")
        # For >= constraints, check the lower bound
        self.assertAlmostEqual(pyo.value(c2.lower), 30.0)

    def test_unmodified_bounds_unchanged(self):
        """SCEN1 has no bounds mods; X and Y keep original bounds."""
        cfg = self._make_cfg()
        smps_module.kw_creator(cfg)
        model = smps_module.scenario_creator("SCEN1", cfg=cfg)
        x = model.find_component("X")
        y = model.find_component("Y")
        self.assertEqual(x.ub, 100.0)
        self.assertEqual(y.lb, 5.0)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_solve_with_modifications(self):
        """Solve both synthetic scenarios as EF and verify feasibility."""
        cfg = self._make_cfg()
        smps_module.kw_creator(cfg)
        scenario_names = smps_module.scenario_names_creator(num_scens=None)

        options = {"solver": solver_name}
        ef = mpisppy.opt.ef.ExtensiveForm(
            options,
            scenario_names,
            smps_module.scenario_creator,
            scenario_creator_kwargs={"cfg": cfg},
        )
        results = ef.solve_extensive_form(tee=False)
        pyo.assert_optimal_termination(results)


if __name__ == "__main__":
    unittest.main()
