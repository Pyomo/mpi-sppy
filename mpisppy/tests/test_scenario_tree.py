###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for mpisppy/scenario_tree.py ScenarioNode class."""

import unittest

import pyomo.environ as pyo
from pyomo.core.base.var import VarData
from pyomo.common.collections import ComponentSet

from mpisppy.scenario_tree import ScenarioNode


def _make_simple_model():
    """Create a minimal two-stage Pyomo model for testing."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], initialize=0.0)
    m.y = pyo.Var(initialize=1.0)
    m.obj = pyo.Objective(expr=m.y, sense=pyo.minimize)
    return m


class TestScenarioNodeBasic(unittest.TestCase):
    """Tests for basic ScenarioNode construction and attribute access."""

    def setUp(self):
        self.m = _make_simple_model()

    def test_name_attribute(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.x], self.m)
        self.assertEqual(node.name, "ROOT")

    def test_cond_prob_attribute(self):
        node = ScenarioNode("ROOT", 0.5, 1, self.m.obj, [self.m.x], self.m)
        self.assertAlmostEqual(node.cond_prob, 0.5)

    def test_stage_attribute(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.x], self.m)
        self.assertEqual(node.stage, 1)

    def test_cost_expression_attribute(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.x], self.m)
        self.assertIs(node.cost_expression, self.m.obj)

    def test_nonant_list_attribute(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.x], self.m)
        self.assertEqual(node.nonant_list, [self.m.x])

    def test_parent_name_none_for_root(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.x], self.m)
        self.assertIsNone(node.parent_name)

    def test_parent_name_set(self):
        node = ScenarioNode("ROOT_0", 0.5, 2, self.m.obj, [self.m.x], self.m,
                            parent_name="ROOT")
        self.assertEqual(node.parent_name, "ROOT")


class TestScenarioNodeVardataList(unittest.TestCase):
    """Tests for ScenarioNode nonant_vardata_list expansion."""

    def setUp(self):
        self.m = _make_simple_model()

    def test_indexed_var_expanded(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.x], self.m)
        # m.x is indexed over [1,2], so vardata_list should have 2 entries
        self.assertEqual(len(node.nonant_vardata_list), 2)

    def test_scalar_var_expanded(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.y], self.m)
        self.assertEqual(len(node.nonant_vardata_list), 1)

    def test_mixed_vars_expanded(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj,
                            [self.m.x, self.m.y], self.m)
        # x has 2 indices, y has 1 => total 3
        self.assertEqual(len(node.nonant_vardata_list), 3)

    def test_vardata_are_vardata_objects(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.x], self.m)
        for v in node.nonant_vardata_list:
            self.assertIsInstance(v, VarData)


class TestScenarioNodeNonantEfSuppl(unittest.TestCase):
    """Tests for ScenarioNode nonant_ef_suppl_list."""

    def setUp(self):
        self.m = _make_simple_model()

    def test_no_suppl_list_gives_empty(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.x], self.m)
        self.assertEqual(node.nonant_ef_suppl_vardata_list, [])

    def test_suppl_list_expanded(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.x], self.m,
                            nonant_ef_suppl_list=[self.m.y])
        self.assertEqual(len(node.nonant_ef_suppl_vardata_list), 1)

    def test_suppl_list_attribute_stored(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.x], self.m,
                            nonant_ef_suppl_list=[self.m.y])
        self.assertEqual(node.nonant_ef_suppl_list, [self.m.y])


class TestScenarioNodeSurrogateNonants(unittest.TestCase):
    """Tests for ScenarioNode surrogate_nonant_list."""

    def setUp(self):
        self.m = _make_simple_model()

    def test_no_surrogate_gives_empty_set(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.x], self.m)
        self.assertIsInstance(node.surrogate_vardatas, ComponentSet)
        self.assertEqual(len(node.surrogate_vardatas), 0)

    def test_surrogate_added_to_nonant_vardata_list(self):
        # Surrogates are appended to nonant_vardata_list
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.x], self.m,
                            surrogate_nonant_list=[self.m.y])
        # x has 2 entries + y has 1 surrogate entry => 3 total
        self.assertEqual(len(node.nonant_vardata_list), 3)

    def test_surrogate_in_surrogate_vardatas(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.x], self.m,
                            surrogate_nonant_list=[self.m.y])
        self.assertIn(self.m.y, node.surrogate_vardatas)

    def test_surrogate_attribute_stored(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, [self.m.x], self.m,
                            surrogate_nonant_list=[self.m.y])
        self.assertEqual(node.surrogate_nonant_list, [self.m.y])


class TestScenarioNodeNoneNonantList(unittest.TestCase):
    """Tests for ScenarioNode when nonant_list is None (warns, empty list)."""

    def setUp(self):
        self.m = _make_simple_model()

    def test_none_nonant_list_gives_empty_vardata(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, None, self.m)
        self.assertEqual(node.nonant_vardata_list, [])

    def test_none_nonant_list_stored(self):
        node = ScenarioNode("ROOT", 1.0, 1, self.m.obj, None, self.m)
        self.assertIsNone(node.nonant_list)


if __name__ == "__main__":
    unittest.main()
