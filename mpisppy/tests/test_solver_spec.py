###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for mpisppy/utils/solver_spec.py solver_specification() function.

These tests cover the logic that looks up a solver name and options from
a Config object, supporting plain, prefixed, and list-of-prefix lookups.
No actual solver is required.
"""

import unittest

from mpisppy.utils.config import Config
from mpisppy.utils.solver_spec import solver_specification


class TestSolverSpecificationNoPrefix(unittest.TestCase):
    """Tests with an empty-string prefix (the common case)."""

    def _make_cfg(self, solver_name=None, solver_options=None):
        cfg = Config()
        cfg.add_solver_specs()
        if solver_name is not None:
            cfg.solver_name = solver_name
        if solver_options is not None:
            cfg.solver_options = solver_options
        return cfg

    def test_returns_solver_name(self):
        cfg = self._make_cfg(solver_name="glpk")
        sroot, name, opts = solver_specification(cfg, prefix="")
        self.assertEqual(name, "glpk")

    def test_returns_empty_string_sroot(self):
        cfg = self._make_cfg(solver_name="glpk")
        sroot, name, opts = solver_specification(cfg, prefix="")
        self.assertEqual(sroot, "")

    def test_returns_empty_options_when_none(self):
        cfg = self._make_cfg(solver_name="glpk")
        sroot, name, opts = solver_specification(cfg, prefix="")
        self.assertEqual(opts, {})

    def test_returns_parsed_options(self):
        cfg = self._make_cfg(solver_name="glpk", solver_options="threads=4 mipgap=0.01")
        sroot, name, opts = solver_specification(cfg, prefix="")
        self.assertEqual(opts["threads"], 4)
        self.assertAlmostEqual(opts["mipgap"], 0.01)

    def test_missing_solver_name_raises_runtime_error(self):
        cfg = self._make_cfg()  # solver_name stays None
        with self.assertRaises(RuntimeError):
            solver_specification(cfg, prefix="")

    def test_missing_solver_name_no_required_does_not_raise(self):
        cfg = self._make_cfg()  # solver_name stays None
        # name_required=False: the for/else skips without raising;
        # returns None for solver_name and {} for solver_options.
        sroot, name, opts = solver_specification(cfg, prefix="", name_required=False)
        self.assertIsNone(name)
        self.assertEqual(opts, {})


class TestSolverSpecificationWithPrefix(unittest.TestCase):
    """Tests with a non-empty prefix (e.g. 'PH')."""

    def _make_cfg_with_prefix(self, prefix, solver_name=None, solver_options=None):
        cfg = Config()
        cfg.add_solver_specs(prefix=prefix)
        name_key = f"{prefix}_solver_name"
        opts_key = f"{prefix}_solver_options"
        if solver_name is not None:
            cfg[name_key] = solver_name
        if solver_options is not None:
            cfg[opts_key] = solver_options
        return cfg

    def test_prefixed_solver_name(self):
        cfg = self._make_cfg_with_prefix("PH", solver_name="cplex")
        sroot, name, opts = solver_specification(cfg, prefix="PH")
        self.assertEqual(name, "cplex")
        self.assertEqual(sroot, "PH")

    def test_prefixed_solver_options_parsed(self):
        cfg = self._make_cfg_with_prefix("EF", solver_name="cplex",
                                          solver_options="threads=2")
        sroot, name, opts = solver_specification(cfg, prefix="EF")
        self.assertEqual(opts["threads"], 2)

    def test_prefix_not_in_cfg_raises(self):
        cfg = Config()
        cfg.add_solver_specs(prefix="PH")
        # PH_solver_name is None → should raise
        with self.assertRaises(RuntimeError):
            solver_specification(cfg, prefix="PH")


class TestSolverSpecificationListOfPrefixes(unittest.TestCase):
    """Tests with a list of prefixes."""

    def test_first_found_in_list(self):
        cfg = Config()
        cfg.add_solver_specs(prefix="PH")
        cfg.add_solver_specs()  # empty-prefix solver_name
        cfg["PH_solver_name"] = "gurobi"
        cfg["solver_name"] = "cplex"
        sroot, name, opts = solver_specification(cfg, prefix=["PH", ""])
        # PH comes first, so we should get gurobi
        self.assertEqual(name, "gurobi")
        self.assertEqual(sroot, "PH")

    def test_fallback_to_second_in_list(self):
        cfg = Config()
        cfg.add_solver_specs(prefix="PH")
        cfg.add_solver_specs()
        # PH_solver_name is None, but solver_name is set
        cfg["solver_name"] = "xpress"
        sroot, name, opts = solver_specification(cfg, prefix=["PH", ""])
        self.assertEqual(name, "xpress")
        self.assertEqual(sroot, "")

    def test_none_found_in_list_raises(self):
        cfg = Config()
        cfg.add_solver_specs(prefix="PH")
        cfg.add_solver_specs()
        # Neither PH_solver_name nor solver_name is set
        with self.assertRaises(RuntimeError):
            solver_specification(cfg, prefix=["PH", ""])

    def test_single_element_tuple_same_as_string(self):
        cfg = Config()
        cfg.add_solver_specs()
        cfg["solver_name"] = "glpk"
        sroot, name, opts = solver_specification(cfg, prefix=("",))
        self.assertEqual(name, "glpk")


if __name__ == "__main__":
    unittest.main()
