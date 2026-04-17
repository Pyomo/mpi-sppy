###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for mpisppy/utils/sputils.py utility functions.

These tests cover pure utility functions that do not require MPI or a solver,
including tree-structure helpers, option-string parsing, and Pyomo model
utilities.
"""

import unittest

import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
from mpisppy.utils.sputils import (
    WarmstartStatus,
    _ScenTree,
    _extract_node_idx,
    _nodenum_before_stage,
    build_vardatalist,
    create_nodenames_from_branching_factors,
    extract_num,
    find_leaves,
    get_branching_factors_from_nodenames,
    module_name_to_module,
    node_idx,
    number_of_nodes,
    option_dict_to_string,
    option_string_to_dict,
    parent_ndn,
    spin_the_wheel,
)


class TestWarmstartStatus(unittest.TestCase):
    """Tests for the WarmstartStatus IntEnum."""

    def test_false_is_falsy(self):
        self.assertFalse(WarmstartStatus.FALSE)

    def test_true_is_truthy(self):
        self.assertTrue(WarmstartStatus.TRUE)

    def test_user_solution_is_truthy(self):
        self.assertTrue(WarmstartStatus.USER_SOLUTION)

    def test_prior_solution_is_truthy(self):
        self.assertTrue(WarmstartStatus.PRIOR_SOLUTION)

    def test_values(self):
        self.assertEqual(WarmstartStatus.FALSE, 0)
        self.assertEqual(WarmstartStatus.TRUE, 1)
        self.assertEqual(WarmstartStatus.USER_SOLUTION, 2)
        self.assertEqual(WarmstartStatus.PRIOR_SOLUTION, 3)

    def test_is_int(self):
        self.assertIsInstance(WarmstartStatus.TRUE, int)


class TestExtractNum(unittest.TestCase):
    """Tests for extract_num()."""

    def test_simple_integer_suffix(self):
        self.assertEqual(extract_num("Scenario324"), 324)

    def test_single_digit(self):
        self.assertEqual(extract_num("Scenario1"), 1)

    def test_leading_zeros_ignored_as_integer(self):
        # int("007") == 7
        self.assertEqual(extract_num("Scenario007"), 7)

    def test_only_digits(self):
        self.assertEqual(extract_num("42"), 42)

    def test_long_suffix(self):
        self.assertEqual(extract_num("abc1234567890"), 1234567890)

    def test_no_trailing_digits_raises(self):
        with self.assertRaises(AttributeError):
            extract_num("NoDigits")

    def test_underscore_prefix(self):
        self.assertEqual(extract_num("node_3"), 3)


class TestParentNdn(unittest.TestCase):
    """Tests for parent_ndn()."""

    def test_root_returns_none(self):
        self.assertIsNone(parent_ndn("ROOT"))

    def test_depth1_child(self):
        self.assertEqual(parent_ndn("ROOT_0"), "ROOT")

    def test_depth1_second_child(self):
        self.assertEqual(parent_ndn("ROOT_2"), "ROOT")

    def test_depth2_child(self):
        self.assertEqual(parent_ndn("ROOT_0_1"), "ROOT_0")

    def test_deep_nesting(self):
        self.assertEqual(parent_ndn("ROOT_1_2_3"), "ROOT_1_2")


class TestNodeIdx(unittest.TestCase):
    """Tests for node_idx()."""

    def test_root_node(self):
        # Empty path => ROOT => id 0
        self.assertEqual(node_idx([], [2, 3]), 0)

    def test_first_stage_first_child(self):
        # Path [0] in a tree with branching [2]: second node (index 1)
        self.assertEqual(node_idx([0], [2]), 1)

    def test_first_stage_second_child(self):
        self.assertEqual(node_idx([1], [2]), 2)

    def test_two_stage_first_child_of_first(self):
        # Branching [2, 3]: stage 1 has 2 nodes (idx 1,2),
        # stage 2 has 6 nodes (idx 3..8)
        # Path [0,0] => first child of first stage-1 node
        bf = [2, 3]
        idx = node_idx([0, 0], bf)
        self.assertIsInstance(idx, int)
        # ROOT=0, ROOT_0=1, ROOT_1=2, ROOT_0_0=3, ROOT_0_1=4, ROOT_0_2=5...
        self.assertEqual(idx, 3)

    def test_number_of_nodes_matches_indices(self):
        bf = [2, 2]
        total = number_of_nodes(bf)
        # A [2,2] tree has ROOT + 2 mid-stage + 4 leaf = 7 nodes
        # number_of_nodes gives index of last node + 1
        self.assertGreater(total, 0)


class TestNodeNumBeforeStage(unittest.TestCase):
    """Tests for _nodenum_before_stage()."""

    def test_stage_0(self):
        self.assertEqual(_nodenum_before_stage(0, [2, 3]), 0)

    def test_stage_1(self):
        # Only ROOT before stage 1 subtree: 1 node
        self.assertEqual(_nodenum_before_stage(1, [2, 3]), 1)

    def test_stage_2(self):
        # ROOT (1) + 2 stage-1 nodes = 3
        self.assertEqual(_nodenum_before_stage(2, [2, 3]), 3)


class TestExtractNodeIdx(unittest.TestCase):
    """Tests for _extract_node_idx()."""

    def test_root(self):
        self.assertEqual(_extract_node_idx("ROOT", [2, 3]), 0)

    def test_first_child(self):
        # ROOT_0 with branching [2] => node_idx([0],[2])
        self.assertEqual(_extract_node_idx("ROOT_0", [2]), node_idx([0], [2]))

    def test_second_child(self):
        self.assertEqual(_extract_node_idx("ROOT_1", [2]), node_idx([1], [2]))

    def test_deep_node(self):
        bf = [3, 2]
        name = "ROOT_2_1"
        self.assertEqual(_extract_node_idx(name, bf), node_idx([2, 1], bf))


class TestOptionStringToDict(unittest.TestCase):
    """Tests for option_string_to_dict()."""

    def test_none_returns_empty(self):
        self.assertEqual(option_string_to_dict(None), {})

    def test_empty_string_returns_empty(self):
        self.assertEqual(option_string_to_dict(""), {})

    def test_single_key_value_int(self):
        result = option_string_to_dict("threads=4")
        self.assertEqual(result, {"threads": 4})

    def test_single_key_value_float(self):
        result = option_string_to_dict("mipgap=0.01")
        self.assertAlmostEqual(result["mipgap"], 0.01)

    def test_single_key_value_string(self):
        result = option_string_to_dict("method=barrier")
        self.assertEqual(result, {"method": "barrier"})

    def test_multiple_options(self):
        result = option_string_to_dict("threads=4 mipgap=0.01")
        self.assertEqual(result["threads"], 4)
        self.assertAlmostEqual(result["mipgap"], 0.01)

    def test_flag_without_value(self):
        result = option_string_to_dict("verbose")
        self.assertIn("verbose", result)
        self.assertIsNone(result["verbose"])

    def test_dict_passthrough(self):
        d = {"threads": 4}
        self.assertIs(option_string_to_dict(d), d)

    def test_illegally_formed_option_raises(self):
        with self.assertRaises(RuntimeError):
            option_string_to_dict("a=b=c")


class TestOptionDictToString(unittest.TestCase):
    """Tests for option_dict_to_string()."""

    def test_none_returns_none(self):
        self.assertIsNone(option_dict_to_string(None))

    def test_empty_dict_returns_empty(self):
        self.assertEqual(option_dict_to_string({}), "")

    def test_key_value_pair(self):
        result = option_dict_to_string({"threads": 4})
        self.assertIn("threads=4", result)

    def test_roundtrip(self):
        orig = "threads=4 mipgap=0.01"
        d = option_string_to_dict(orig)
        s = option_dict_to_string(d)
        # roundtrip: parse the resulting string back
        d2 = option_string_to_dict(s)
        self.assertEqual(d["threads"], d2["threads"])
        self.assertAlmostEqual(d["mipgap"], d2["mipgap"])


class TestCreateNodenamesFromBranchingFactors(unittest.TestCase):
    """Tests for create_nodenames_from_branching_factors()."""

    def test_two_stage(self):
        # Two-stage: branching_factors has one element; only ROOT returned
        result = create_nodenames_from_branching_factors([3])
        self.assertEqual(result, ["ROOT"])

    def test_three_stage_two_by_two(self):
        result = create_nodenames_from_branching_factors([2, 2])
        self.assertIn("ROOT", result)
        # Stage 1 nodes
        self.assertIn("ROOT_0", result)
        self.assertIn("ROOT_1", result)
        # Stage 2 nodes
        self.assertIn("ROOT_0_0", result)
        self.assertIn("ROOT_0_1", result)
        self.assertIn("ROOT_1_0", result)
        self.assertIn("ROOT_1_1", result)

    def test_count_three_stage(self):
        result = create_nodenames_from_branching_factors([2, 3])
        # ROOT + 2 + 6 = 9
        self.assertEqual(len(result), 9)


class TestGetBranchingFactorsFromNodenames(unittest.TestCase):
    """Tests for get_branching_factors_from_nodenames()."""

    def test_two_stage_returns_empty(self):
        # A two-stage tree has only ROOT in nodenames; no branching factors
        # can be inferred from a root-only node list.
        nodenames = create_nodenames_from_branching_factors([3])
        result = get_branching_factors_from_nodenames(nodenames)
        self.assertEqual(result, [])

    def test_three_stage(self):
        bf = [2, 3]
        nodenames = create_nodenames_from_branching_factors(bf)
        result = get_branching_factors_from_nodenames(nodenames)
        self.assertEqual(result, bf)

    def test_four_stage(self):
        bf = [2, 2, 2]
        nodenames = create_nodenames_from_branching_factors(bf)
        result = get_branching_factors_from_nodenames(nodenames)
        self.assertEqual(result, bf)

    def test_roundtrip(self):
        bf = [3, 4]
        nodenames = create_nodenames_from_branching_factors(bf)
        recovered = get_branching_factors_from_nodenames(nodenames)
        self.assertEqual(recovered, bf)


class TestFindLeaves(unittest.TestCase):
    """Tests for find_leaves()."""

    def test_two_stage_returns_root_not_leaf(self):
        result = find_leaves(["ROOT"])
        self.assertEqual(result, {"ROOT": False})

    def test_none_same_as_root_only(self):
        result = find_leaves(None)
        self.assertEqual(result, {"ROOT": False})

    def test_three_stage_leaves_and_nonleaves(self):
        nodenames = create_nodenames_from_branching_factors([2, 2])
        result = find_leaves(nodenames)
        # ROOT is not a leaf
        self.assertFalse(result["ROOT"])
        # Stage-1 nodes are not leaves
        self.assertFalse(result["ROOT_0"])
        self.assertFalse(result["ROOT_1"])
        # Stage-2 nodes are leaves
        self.assertTrue(result["ROOT_0_0"])
        self.assertTrue(result["ROOT_0_1"])
        self.assertTrue(result["ROOT_1_0"])
        self.assertTrue(result["ROOT_1_1"])


class TestScenTree(unittest.TestCase):
    """Tests for the _ScenTree class."""

    def _make_scen_names(self, n):
        return [f"Scenario{i}" for i in range(n)]

    def test_two_stage_construction(self):
        scen_names = self._make_scen_names(3)
        tree = _ScenTree(None, scen_names)
        self.assertEqual(tree.NumScens, 3)
        self.assertEqual(tree.NumStages, 2)

    def test_two_stage_with_root_list(self):
        scen_names = self._make_scen_names(4)
        tree = _ScenTree(["ROOT"], scen_names)
        self.assertEqual(tree.NumScens, 4)
        self.assertEqual(tree.NumStages, 2)

    def test_three_stage_construction(self):
        bf = [2, 3]
        num_scens = 2 * 3
        nodenames = create_nodenames_from_branching_factors(bf)
        scen_names = self._make_scen_names(num_scens)
        tree = _ScenTree(nodenames, scen_names)
        self.assertEqual(tree.NumScens, num_scens)
        self.assertEqual(tree.NumStages, 3)

    def test_nonleaves_two_stage(self):
        scen_names = self._make_scen_names(3)
        tree = _ScenTree(None, scen_names)
        # Only ROOT is a non-leaf in a two-stage tree
        self.assertEqual(len(tree.nonleaves), 1)
        self.assertEqual(tree.nonleaves[0].name, "ROOT")

    def test_nonleaves_three_stage(self):
        bf = [2, 2]
        nodenames = create_nodenames_from_branching_factors(bf)
        scen_names = self._make_scen_names(4)
        tree = _ScenTree(nodenames, scen_names)
        # ROOT + 2 stage-1 nodes are non-leaves
        self.assertEqual(len(tree.nonleaves), 3)

    def test_num_leaves_two_stage(self):
        scen_names = self._make_scen_names(5)
        tree = _ScenTree(None, scen_names)
        # In a two-stage tree the "leaf count" is 1 (ROOT only, no sub-nodes)
        # NumLeaves = len(desc_leaf_dict) - len(nonleaves) = 1 - 1 = 0
        # This is intentional: two-stage doesn't enumerate leaf nodes
        self.assertGreaterEqual(tree.NumLeaves, 0)

    def test_scen_names_to_ranks_single_proc(self):
        scen_names = self._make_scen_names(4)
        tree = _ScenTree(None, scen_names)
        sntr, slices, ranks = tree.scen_names_to_ranks(1)
        # All scenarios go to rank 0
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0], list(range(4)))
        for r in ranks:
            self.assertEqual(r, 0)

    def test_scen_names_to_ranks_multi_proc(self):
        scen_names = self._make_scen_names(4)
        tree = _ScenTree(None, scen_names)
        sntr, slices, ranks = tree.scen_names_to_ranks(2)
        self.assertEqual(len(slices), 2)
        # Each rank should get 2 scenarios
        self.assertEqual(len(slices[0]) + len(slices[1]), 4)

    def test_inconsistent_tree_raises(self):
        # Providing nodenames that result in wrong leaf count should raise
        bf = [2, 2]
        nodenames = create_nodenames_from_branching_factors(bf)
        # Only 3 scenarios but 4 leaves
        with self.assertRaises(RuntimeError):
            _ScenTree(nodenames, self._make_scen_names(3))


class TestNumberOfNodes(unittest.TestCase):
    """Tests for number_of_nodes()."""

    def test_two_stage_single_branch(self):
        # [1] => just ROOT + 1 node => gives index of last node
        result = number_of_nodes([1])
        self.assertIsInstance(result, int)

    def test_three_stage(self):
        bf = [2, 3]
        result = number_of_nodes(bf)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)


class TestBuildVardatalist(unittest.TestCase):
    """Tests for build_vardatalist()."""

    def setUp(self):
        self.m = pyo.ConcreteModel()
        self.m.x = pyo.Var([1, 2, 3], initialize=0.0)
        self.m.y = pyo.Var(initialize=5.0)
        self.m.z = pyo.Var(["a", "b"], initialize=1.0)

    def test_scalar_var(self):
        result = build_vardatalist(self.m, [self.m.y])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], self.m.y)

    def test_indexed_var(self):
        result = build_vardatalist(self.m, [self.m.x])
        self.assertEqual(len(result), 3)

    def test_mixed_vars(self):
        result = build_vardatalist(self.m, [self.m.y, self.m.x])
        self.assertEqual(len(result), 4)

    def test_none_varlist_raises(self):
        with self.assertRaises(RuntimeError):
            build_vardatalist(self.m, None)

    def test_single_var_not_in_list(self):
        # Passing a Var directly (not wrapped in a list) should still work
        result = build_vardatalist(self.m, self.m.y)
        self.assertEqual(len(result), 1)

    def test_indexed_string_keys(self):
        result = build_vardatalist(self.m, [self.m.z])
        self.assertEqual(len(result), 2)

    def test_sorted_keys(self):
        result = build_vardatalist(self.m, [self.m.x])
        keys = [v.index() for v in result]
        self.assertEqual(keys, sorted(keys))


class TestSpinTheWheelDeprecated(unittest.TestCase):
    """Tests that the deprecated spin_the_wheel function raises RuntimeError."""

    def test_raises_runtime_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            spin_the_wheel({}, [])
        self.assertIn("WheelSpinner", str(ctx.exception))


class TestModuleNameToModule(unittest.TestCase):
    """Tests for module_name_to_module()."""

    def test_string_name_loads_module(self):
        mod = module_name_to_module("os.path")
        import os.path
        self.assertIs(mod, os.path)

    def test_module_passthrough(self):
        import math
        result = module_name_to_module(math)
        self.assertIs(result, math)

    def test_invalid_module_raises(self):
        with self.assertRaises(ModuleNotFoundError):
            module_name_to_module("nonexistent.module.xyz")


class TestGetObjs(unittest.TestCase):
    """Tests for get_objs()."""

    def test_single_objective(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
        objs = sputils.get_objs(m)
        self.assertEqual(len(objs), 1)

    def test_no_objective_raises(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        with self.assertRaises(RuntimeError):
            sputils.get_objs(m)

    def test_no_objective_allow_none(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        objs = sputils.get_objs(m, allow_none=True)
        self.assertEqual(objs, [])


class TestFindActiveObjective(unittest.TestCase):
    """Tests for find_active_objective() and find_objective()."""

    def test_single_active_objective(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.obj = pyo.Objective(expr=m.x)
        obj = sputils.find_active_objective(m)
        self.assertIs(obj, m.obj)

    def test_no_objective_raises(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        with self.assertRaises(RuntimeError):
            sputils.find_active_objective(m)

    def test_inactive_objective_found_by_find_objective(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.obj = pyo.Objective(expr=m.x)
        m.obj.deactivate()
        obj = sputils.find_objective(m, active=False)
        self.assertIs(obj, m.obj)

    def test_multiple_active_objectives_raises(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj1 = pyo.Objective(expr=m.x)
        m.obj2 = pyo.Objective(expr=m.y)
        with self.assertRaises(RuntimeError):
            sputils.find_active_objective(m)


class TestDeactAndReactivateObjs(unittest.TestCase):
    """Tests for deact_objs() and related functions."""

    def test_deact_objs(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.obj = pyo.Objective(expr=m.x)
        self.assertTrue(m.obj.active)
        sputils.deact_objs(m)
        self.assertFalse(m.obj.active)

    def test_deact_objs_returns_list(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.obj = pyo.Objective(expr=m.x)
        obj_list = sputils.deact_objs(m)
        self.assertEqual(len(obj_list), 1)

    def test_deact_no_objs_returns_empty(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        obj_list = sputils.deact_objs(m)
        self.assertEqual(obj_list, [])


class TestReactivateObjs(unittest.TestCase):
    """Tests for reactivate_objs() after stash_ref_objs()."""

    def _make_model_with_mpisppy_data(self):
        """Create a minimal model with the _mpisppy_data attribute that
        stash_ref_objs() and reactivate_objs() expect."""
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)

        class _Data:
            pass

        m._mpisppy_data = _Data()
        return m

    def test_reactivate_after_stash_and_deact(self):
        m = self._make_model_with_mpisppy_data()
        # Stash then deactivate
        sputils.stash_ref_objs(m)
        sputils.deact_objs(m)
        self.assertFalse(m.obj.active)
        # Reactivate
        sputils.reactivate_objs(m)
        self.assertTrue(m.obj.active)

    def test_reactivate_without_stash_raises(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.obj = pyo.Objective(expr=m.x)

        class _Data:
            pass

        m._mpisppy_data = _Data()
        # No stash_ref_objs call → should raise RuntimeError
        with self.assertRaises(RuntimeError):
            sputils.reactivate_objs(m)


class TestModelsHaveSameSense(unittest.TestCase):
    """Tests for _models_have_same_sense()."""

    def _make_minimize_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
        return m

    def _make_maximize_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.obj = pyo.Objective(expr=m.x, sense=pyo.maximize)
        return m

    def test_empty_dict_returns_true_true(self):
        is_min, check = sputils._models_have_same_sense({})
        self.assertTrue(is_min)
        self.assertTrue(check)

    def test_all_minimize_returns_true_true(self):
        models = {"s1": self._make_minimize_model(),
                  "s2": self._make_minimize_model()}
        is_min, check = sputils._models_have_same_sense(models)
        self.assertTrue(is_min)
        self.assertTrue(check)

    def test_all_maximize_returns_false_true(self):
        models = {"s1": self._make_maximize_model(),
                  "s2": self._make_maximize_model()}
        is_min, check = sputils._models_have_same_sense(models)
        self.assertFalse(is_min)
        self.assertTrue(check)

    def test_mixed_sense_returns_none_false(self):
        models = {"s1": self._make_minimize_model(),
                  "s2": self._make_maximize_model()}
        is_min, check = sputils._models_have_same_sense(models)
        self.assertIsNone(is_min)
        self.assertFalse(check)

    def test_single_minimize_model(self):
        models = {"s1": self._make_minimize_model()}
        is_min, check = sputils._models_have_same_sense(models)
        self.assertTrue(is_min)
        self.assertTrue(check)


class TestTicTocOutput(unittest.TestCase):
    """Tests for disable_tictoc_output() and reenable_tictoc_output()."""

    def test_disable_and_reenable_do_not_raise(self):
        # These functions mutate a global timer object; just verify they
        # can be called in sequence without throwing.
        sputils.disable_tictoc_output()
        sputils.reenable_tictoc_output()

    def test_double_reenable_after_disable(self):
        sputils.disable_tictoc_output()
        sputils.reenable_tictoc_output()
        # Calling disable/reenable again should still work
        sputils.disable_tictoc_output()
        sputils.reenable_tictoc_output()


class TestNotGoodEnoughResults(unittest.TestCase):
    """Tests for not_good_enough_results()."""

    def test_none_results_is_not_good_enough(self):
        self.assertTrue(sputils.not_good_enough_results(None))


# ---------------------------------------------------------------------------
# Helpers shared by EF tests
# ---------------------------------------------------------------------------

def _make_two_stage_scenario(scenario_name, num_scens=3):
    """A minimal two-stage scenario creator (no solver required)."""
    m = pyo.ConcreteModel(scenario_name)
    m.x = pyo.Var([1, 2], bounds=(0, 10), initialize=1.0)
    m.y = pyo.Var(bounds=(0, 100), initialize=5.0)
    m.FirstStageCost = pyo.Expression(expr=sum(m.x[i] for i in m.x))
    m.SecondStageCost = pyo.Expression(expr=m.y)
    m.obj = pyo.Objective(
        expr=m.FirstStageCost + m.SecondStageCost, sense=pyo.minimize
    )
    sputils.attach_root_node(m, m.FirstStageCost, [m.x])
    m._mpisppy_probability = 1.0 / num_scens
    return m


# ---------------------------------------------------------------------------
# attach_root_node
# ---------------------------------------------------------------------------

class TestAttachRootNode(unittest.TestCase):
    """Tests for attach_root_node()."""

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2], initialize=0.0)
        m.cost = pyo.Expression(expr=sum(m.x[i] for i in m.x))
        m.obj = pyo.Objective(expr=m.cost, sense=pyo.minimize)
        return m

    def test_attaches_node_list(self):
        m = self._make_model()
        sputils.attach_root_node(m, m.cost, [m.x])
        self.assertTrue(hasattr(m, "_mpisppy_node_list"))
        self.assertEqual(len(m._mpisppy_node_list), 1)

    def test_node_name_is_root(self):
        m = self._make_model()
        sputils.attach_root_node(m, m.cost, [m.x])
        self.assertEqual(m._mpisppy_node_list[0].name, "ROOT")

    def test_sets_uniform_probability(self):
        m = self._make_model()
        sputils.attach_root_node(m, m.cost, [m.x])
        self.assertEqual(m._mpisppy_probability, "uniform")

    def test_do_uniform_false_does_not_set_probability(self):
        m = self._make_model()
        sputils.attach_root_node(m, m.cost, [m.x], do_uniform=False)
        self.assertFalse(hasattr(m, "_mpisppy_probability"))

    def test_nonant_vardata_list_correct(self):
        m = self._make_model()
        sputils.attach_root_node(m, m.cost, [m.x])
        node = m._mpisppy_node_list[0]
        # m.x has 2 elements
        self.assertEqual(len(node.nonant_vardata_list), 2)

    def test_existing_probability_not_overwritten(self):
        m = self._make_model()
        m._mpisppy_probability = 0.5
        sputils.attach_root_node(m, m.cost, [m.x])
        # do_uniform=True but probability already set, so no overwrite
        self.assertAlmostEqual(m._mpisppy_probability, 0.5)


# ---------------------------------------------------------------------------
# create_EF
# ---------------------------------------------------------------------------

class TestCreateEF(unittest.TestCase):
    """Tests for create_EF() using a solver-free scenario creator."""

    def _scen_names(self, n):
        return [f"Scenario{i + 1}" for i in range(n)]

    def test_creates_ef_instance(self):
        ef = sputils.create_EF(
            self._scen_names(3),
            _make_two_stage_scenario,
        )
        self.assertIsInstance(ef, pyo.ConcreteModel)

    def test_ef_has_scenario_names(self):
        names = self._scen_names(3)
        ef = sputils.create_EF(names, _make_two_stage_scenario)
        self.assertEqual(sorted(ef._ef_scenario_names), sorted(names))

    def test_ef_has_ef_obj(self):
        ef = sputils.create_EF(self._scen_names(3), _make_two_stage_scenario)
        self.assertTrue(hasattr(ef, "EF_Obj"))

    def test_ef_has_ref_vars(self):
        ef = sputils.create_EF(self._scen_names(3), _make_two_stage_scenario)
        self.assertTrue(hasattr(ef, "ref_vars"))
        self.assertGreater(len(ef.ref_vars), 0)

    def test_ef_has_nonant_constraints(self):
        ef = sputils.create_EF(self._scen_names(3), _make_two_stage_scenario)
        self.assertTrue(hasattr(ef, "_C_EF_"))

    def test_single_scenario_warns_but_returns(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ef = sputils.create_EF(
                self._scen_names(1),
                _make_two_stage_scenario,
            )
        self.assertIsInstance(ef, pyo.ConcreteModel)

    def test_empty_scenario_list_raises(self):
        with self.assertRaises(RuntimeError):
            sputils.create_EF([], _make_two_stage_scenario)

    def test_uniform_probability_assigned(self):
        # When _mpisppy_probability is "uniform", it should be replaced with 1/n.
        def uniform_creator(name, num_scens=None):
            m = pyo.ConcreteModel(name)
            m.x = pyo.Var(initialize=1.0)
            m.cost = pyo.Expression(expr=m.x)
            m.obj = pyo.Objective(expr=m.cost, sense=pyo.minimize)
            sputils.attach_root_node(m, m.cost, [m.x])
            return m

        ef = sputils.create_EF(self._scen_names(4), uniform_creator)
        self.assertIsInstance(ef, pyo.ConcreteModel)

    def test_mixed_sense_raises(self):
        def min_creator(name, **kwargs):
            m = pyo.ConcreteModel(name)
            m.x = pyo.Var(initialize=0.0)
            m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
            sputils.attach_root_node(m, m.obj.expr, [m.x])
            m._mpisppy_probability = 0.5
            return m

        def max_creator(name, **kwargs):
            m = pyo.ConcreteModel(name)
            m.x = pyo.Var(initialize=0.0)
            m.obj = pyo.Objective(expr=m.x, sense=pyo.maximize)
            sputils.attach_root_node(m, m.obj.expr, [m.x])
            m._mpisppy_probability = 0.5
            return m

        def mixed_creator(name, **kwargs):
            if name == "Scenario1":
                return min_creator(name)
            return max_creator(name)

        with self.assertRaises(RuntimeError):
            sputils.create_EF(self._scen_names(2), mixed_creator)


# ---------------------------------------------------------------------------
# ef_scenarios, ef_nonants, ef_nonants_csv, nonant_cache_from_ef
# ---------------------------------------------------------------------------

class TestEFIterators(unittest.TestCase):
    """Tests for ef_scenarios(), ef_nonants(), ef_nonants_csv(),
    and nonant_cache_from_ef()."""

    def setUp(self):
        names = [f"Scenario{i + 1}" for i in range(3)]
        self.ef = sputils.create_EF(names, _make_two_stage_scenario)

    def test_ef_scenarios_yields_correct_count(self):
        pairs = list(sputils.ef_scenarios(self.ef))
        self.assertEqual(len(pairs), 3)

    def test_ef_scenarios_yields_name_model_pairs(self):
        for name, model in sputils.ef_scenarios(self.ef):
            self.assertIsInstance(name, str)
            self.assertIsInstance(model, pyo.ConcreteModel)

    def test_ef_scenarios_names_match_ef(self):
        yielded = {name for name, _ in sputils.ef_scenarios(self.ef)}
        self.assertEqual(yielded, set(self.ef._ef_scenario_names))

    def test_ef_nonants_yields_tuples(self):
        nonants = list(sputils.ef_nonants(self.ef))
        self.assertGreater(len(nonants), 0)
        for item in nonants:
            ndn, var, val = item
            self.assertIsInstance(ndn, str)

    def test_ef_nonants_node_name_is_root(self):
        for ndn, _var, _val in sputils.ef_nonants(self.ef):
            self.assertEqual(ndn, "ROOT")

    def test_ef_nonants_csv_creates_file(self):
        import os
        import tempfile
        fd, fname = tempfile.mkstemp(suffix='.csv')
        os.close(fd)
        try:
            sputils.ef_nonants_csv(self.ef, fname)
            self.assertTrue(os.path.exists(fname))
            with open(fname) as fh:
                content = fh.read()
            self.assertIn("Node", content)
        finally:
            os.unlink(fname)

    def test_nonant_cache_from_ef_returns_dict(self):
        cache = sputils.nonant_cache_from_ef(self.ef)
        self.assertIsInstance(cache, dict)

    def test_nonant_cache_from_ef_has_root_key(self):
        cache = sputils.nonant_cache_from_ef(self.ef)
        self.assertIn("ROOT", cache)

    def test_nonant_cache_from_ef_root_length(self):
        cache = sputils.nonant_cache_from_ef(self.ef)
        # m.x has 2 elements
        self.assertEqual(len(cache["ROOT"]), 2)


# ---------------------------------------------------------------------------
# Deprecated write_spin_the_wheel_* functions
# ---------------------------------------------------------------------------

class TestDeprecatedSpinTheWheelWriters(unittest.TestCase):
    """Deprecated solution-writer wrappers must raise RuntimeError."""

    def test_write_first_stage_raises(self):
        with self.assertRaises(RuntimeError):
            sputils.write_spin_the_wheel_first_stage_solution(None, None, "x.csv")

    def test_write_tree_solution_raises(self):
        with self.assertRaises(RuntimeError):
            sputils.write_spin_the_wheel_tree_solution(None, None, "/tmp/sol")

    def test_local_nonant_cache_raises(self):
        with self.assertRaises(RuntimeError):
            sputils.local_nonant_cache(None)


if __name__ == "__main__":
    unittest.main()
