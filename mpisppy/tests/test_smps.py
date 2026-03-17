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

import mpisppy.utils.smps_reader as smps_reader
import mpisppy.utils.smps_module as smps_module
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


if __name__ == "__main__":
    unittest.main()
