###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for preference-driven in-hub slamming
(mpisppy/extensions/slammer.py).

Three groups, mirroring the three decoupled layers of the design
(doc/designs/slamming_design.md):

  * parser/matcher    -- the directives file and its wildcard semantics,
  * action            -- each direction fixes to the expected value, plus
                         selection / eligibility / stickiness,
  * trigger + config  -- the iteration-count predicate and the backward
                         compatibility contract in Config.checker().

The action tests bypass MPI and the solver: a fake ``opt`` carries Pyomo
scenario models directly, and a one-process ``_FakeComm`` makes the min/max
Allreduce an identity, so the fix decisions can be checked deterministically.
"""

import os
import tempfile
import types
import unittest

import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet

import mpisppy.utils.config as config
from mpisppy.extensions.slammer import (
    Slammer,
    SlamDirective,
    parse_directives_file,
    resolve_directive,
    VALID_DIRECTIONS,
)

NDN = "ROOT"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_csv(text):
    """Write text to a temp CSV (under the repo tree's default tmp) and return
    its path; caller removes it."""
    fd, path = tempfile.mkstemp(suffix=".csv", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def _make_scenario(specs):
    """One Pyomo scenario model from a list of var specs.

    Each spec is a dict with keys: value, and optional lb, ub, xbar, integer,
    binary, modeler_fixed, surrogate.  A var is named ``v[i]``.
    """
    n = len(specs)

    def _bounds(m, i):
        return (specs[i].get("lb", 0.0), specs[i].get("ub", 10.0))

    def _domain(m, i):
        if specs[i].get("binary"):
            return pyo.Binary
        return pyo.Integers if specs[i].get("integer") else pyo.Reals

    sm = pyo.ConcreteModel()
    sm.v = pyo.Var(range(n), bounds=_bounds, domain=_domain)

    nonant_indices = {}
    xbars = {}
    surrogates = ComponentSet()
    for i, sp in enumerate(specs):
        var = sm.v[i]
        var._value = sp["value"]
        if sp.get("modeler_fixed"):
            var.fix(sp["value"])
        nonant_indices[(NDN, i)] = var
        xbars[(NDN, i)] = sp.get("xbar", sp["value"])  # pyo.value() of a float is the float
        if sp.get("surrogate"):
            surrogates.add(var)

    scenario = types.SimpleNamespace(
        _solver_plugin=None,  # is_persistent(None) is False -> no update_var calls
        _mpisppy_data=types.SimpleNamespace(
            nonant_indices=nonant_indices,
            all_surrogate_nonants=surrogates,
        ),
        _mpisppy_model=types.SimpleNamespace(xbars=xbars),
    )
    return sm, scenario


class _FakeComm:
    """One-process communicator: Allreduce is the identity (local == global)."""
    def Allreduce(self, sendbuf, recvbuf, op=None):
        recvbuf[0][:] = sendbuf[0]


def _make_opt(scenarios, slammer_options):
    """A fake opt carrying the given scenarios and slammer options."""
    opt = types.SimpleNamespace(
        cylinder_rank=0,
        _PHIter=1,
        comms={NDN: _FakeComm()},
        local_scenarios={f"Scen{i+1}": s for i, s in enumerate(scenarios)},
        options={"slammer_options": slammer_options},
    )
    opt.local_scenario_names = list(opt.local_scenarios.keys())
    return opt


def _build(directives, scenario_specs, slam_start_iter=1, iters_between_slams=1,
           rounding_bias=0.0):
    """Build (ext, opt, models) from directives and a list of scenarios, each a
    list of var specs.  pre_iter0() is called so the eligibility map is ready."""
    models, scenarios = [], []
    for specs in scenario_specs:
        m, sc = _make_scenario(specs)
        models.append(m)
        scenarios.append(sc)

    opt = _make_opt(scenarios, {
        "directives": directives,
        "slam_start_iter": slam_start_iter,
        "iters_between_slams": iters_between_slams,
        "rounding_bias": rounding_bias,
        "verbose": False,
    })

    ext = Slammer(opt)
    ext.pre_iter0()
    return ext, opt, models


def _var(models, scen_idx, var_idx):
    return models[scen_idx].v[var_idx]


# --------------------------------------------------------------------------- #
# parser / matcher
# --------------------------------------------------------------------------- #
class Test_parser(unittest.TestCase):
    def test_basic_parse(self):
        path = _write_csv(
            "name,can_slam,directions,priority\n"
            "DoBuild[*],1,ub|lb,100\n"
            "NumUnits[*],1,nearest,50\n"
        )
        try:
            ds = parse_directives_file(path)
        finally:
            os.remove(path)
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0].pattern, "DoBuild[*]")
        self.assertTrue(ds[0].can_slam)
        self.assertEqual(ds[0].directions, ("ub", "lb"))
        self.assertEqual(ds[0].priority, 100.0)

    def test_can_slam_defaults_true(self):
        path = _write_csv("name,directions\nDoBuild[*],lb\n")
        try:
            ds = parse_directives_file(path)
        finally:
            os.remove(path)
        self.assertTrue(ds[0].can_slam)
        self.assertEqual(ds[0].priority, 0.0)  # default

    def test_can_slam_zero_allows_empty_directions(self):
        path = _write_csv(
            "name,can_slam,directions,priority\n"
            "NumUnits[*],1,nearest,50\n"
            "NumUnits[7],0,,0\n"
        )
        try:
            ds = parse_directives_file(path)
        finally:
            os.remove(path)
        self.assertFalse(ds[1].can_slam)
        self.assertEqual(ds[1].directions, ())

    def test_blank_name_rows_skipped(self):
        path = _write_csv("name,directions\nDoBuild[*],lb\n,\n")
        try:
            ds = parse_directives_file(path)
        finally:
            os.remove(path)
        self.assertEqual(len(ds), 1)

    def test_comments_and_blanks_ignored(self):
        # comments are allowed before the header and between rows
        path = _write_csv(
            "# leading comment\n"
            "\n"
            "name,directions,priority\n"
            "# a note about DoBuild\n"
            "DoBuild[*],lb,1\n"
            "\n"
            "NumUnits[*],ub,2\n"
        )
        try:
            ds = parse_directives_file(path)
        finally:
            os.remove(path)
        self.assertEqual([d.pattern for d in ds], ["DoBuild[*]", "NumUnits[*]"])

    def test_quoted_multi_index_name(self):
        # a name with a comma must be quoted to survive the CSV split
        path = _write_csv(
            'name,directions,priority\n'
            '"Cut[*,*]",max,10\n'
        )
        try:
            ds = parse_directives_file(path)
        finally:
            os.remove(path)
        self.assertEqual(ds[0].pattern, "Cut[*,*]")
        self.assertTrue(ds[0].matches("Cut[1,2]"))

    def test_sizes_example_file_parses(self):
        # the shipped example must stay valid
        here = os.path.dirname(__file__)
        example = os.path.join(
            here, "..", "..", "examples", "sizes", "config",
            "slamming_directives.csv")
        ds = parse_directives_file(example)
        self.assertTrue(any(d.pattern == "NumUnitsCutFirstStage[*,*]" for d in ds))
        # last-match-wins gives the per-index priority, slammed to max
        d = resolve_directive(ds, "NumProducedFirstStage[10]")
        self.assertEqual(d.directions, ("max",))
        self.assertEqual(d.priority, 10.0)

    def test_unknown_direction_raises(self):
        path = _write_csv("name,directions\nDoBuild[*],sideways\n")
        try:
            with self.assertRaises(ValueError):
                parse_directives_file(path)
        finally:
            os.remove(path)

    def test_can_slam_one_no_directions_raises(self):
        path = _write_csv("name,can_slam,directions\nDoBuild[*],1,\n")
        try:
            with self.assertRaises(ValueError):
                parse_directives_file(path)
        finally:
            os.remove(path)

    def test_missing_name_column_raises(self):
        path = _write_csv("pattern,directions\nDoBuild[*],lb\n")
        try:
            with self.assertRaises(ValueError):
                parse_directives_file(path)
        finally:
            os.remove(path)

    def test_unparseable_priority_raises(self):
        path = _write_csv("name,directions,priority\nDoBuild[*],lb,high\n")
        try:
            with self.assertRaises(ValueError):
                parse_directives_file(path)
        finally:
            os.remove(path)


class Test_matcher(unittest.TestCase):
    def test_brackets_are_literal(self):
        # the design's own examples; raw fnmatch would fail all of these
        d = SlamDirective("DoBuild[*]", True, ("lb",), 0.0)
        self.assertTrue(d.matches("DoBuild[Seattle]"))
        self.assertFalse(d.matches("DoBuildX"))  # bracket is literal, not a class
        d2 = SlamDirective("Reservoir*.spill[*]", True, ("lb",), 0.0)
        self.assertTrue(d2.matches("Reservoir1.spill[3]"))
        d3 = SlamDirective("NumUnits[7]", True, ("lb",), 0.0)
        self.assertTrue(d3.matches("NumUnits[7]"))
        self.assertFalse(d3.matches("NumUnits7"))

    def test_question_mark_one_char(self):
        d = SlamDirective("NumUnits[?]", True, ("lb",), 0.0)
        self.assertTrue(d.matches("NumUnits[7]"))
        self.assertFalse(d.matches("NumUnits[70]"))

    def test_full_match_required(self):
        d = SlamDirective("DoBuild", True, ("lb",), 0.0)
        self.assertTrue(d.matches("DoBuild"))
        self.assertFalse(d.matches("DoBuildMore"))

    def test_last_match_wins(self):
        ds = [
            SlamDirective("NumUnits[*]", True, ("nearest",), 50.0),
            SlamDirective("NumUnits[7]", False, (), 0.0),
        ]
        self.assertTrue(resolve_directive(ds, "NumUnits[3]").can_slam)
        self.assertFalse(resolve_directive(ds, "NumUnits[7]").can_slam)

    def test_unmatched_is_none(self):
        ds = [SlamDirective("NumUnits[*]", True, ("lb",), 0.0)]
        self.assertIsNone(resolve_directive(ds, "Other[1]"))


# --------------------------------------------------------------------------- #
# action -- directions
# --------------------------------------------------------------------------- #
class Test_directions(unittest.TestCase):
    def test_slam_lb(self):
        ds = [SlamDirective("v[*]", True, ("lb",), 1.0)]
        ext, opt, models = _build(ds, [[{"lb": 2.0, "ub": 9.0, "value": 5.0}]])
        ext._slam_one()
        self.assertTrue(_var(models, 0, 0).fixed)
        self.assertEqual(_var(models, 0, 0).value, 2.0)

    def test_slam_ub(self):
        ds = [SlamDirective("v[*]", True, ("ub",), 1.0)]
        ext, opt, models = _build(ds, [[{"lb": 2.0, "ub": 9.0, "value": 5.0}]])
        ext._slam_one()
        self.assertEqual(_var(models, 0, 0).value, 9.0)

    def test_nearest_picks_closer_bound(self):
        ds = [SlamDirective("v[*]", True, ("nearest",), 1.0)]
        # xbar 2 is closer to lb 0 than ub 10
        ext, _, models = _build(ds, [[{"lb": 0.0, "ub": 10.0, "value": 2.0, "xbar": 2.0}]])
        ext._slam_one()
        self.assertEqual(_var(models, 0, 0).value, 0.0)
        # xbar 8 is closer to ub 10
        ds = [SlamDirective("v[*]", True, ("nearest",), 1.0)]
        ext, _, models = _build(ds, [[{"lb": 0.0, "ub": 10.0, "value": 8.0, "xbar": 8.0}]])
        ext._slam_one()
        self.assertEqual(_var(models, 0, 0).value, 10.0)

    def test_anywhere_rounds_integer(self):
        ds = [SlamDirective("v[*]", True, ("anywhere",), 1.0)]
        ext, _, models = _build(
            ds, [[{"lb": 0, "ub": 10, "value": 3.4, "xbar": 3.4, "integer": True}]])
        ext._slam_one()
        self.assertEqual(_var(models, 0, 0).value, 3)  # round(3.4)

    def test_anywhere_rounding_bias(self):
        ds = [SlamDirective("v[*]", True, ("anywhere",), 1.0)]
        ext, _, models = _build(
            ds, [[{"lb": 0, "ub": 10, "value": 3.4, "xbar": 3.4, "integer": True}]],
            rounding_bias=0.2)
        ext._slam_one()
        self.assertEqual(_var(models, 0, 0).value, 4)  # round(3.4 + 0.2)

    def test_anywhere_continuous_uses_xbar(self):
        ds = [SlamDirective("v[*]", True, ("anywhere",), 1.0)]
        ext, _, models = _build(ds, [[{"lb": 0.0, "ub": 10.0, "value": 3.7, "xbar": 3.7}]])
        ext._slam_one()
        self.assertAlmostEqual(_var(models, 0, 0).value, 3.7)

    def test_min_across_scenarios(self):
        ds = [SlamDirective("v[*]", True, ("min",), 1.0)]
        ext, _, models = _build(ds, [
            [{"lb": 0.0, "ub": 10.0, "value": 5.5}],
            [{"lb": 0.0, "ub": 10.0, "value": 2.5}],
        ])
        ext._slam_one()
        # fixed to the min (2.5) in *every* scenario
        self.assertAlmostEqual(_var(models, 0, 0).value, 2.5)
        self.assertAlmostEqual(_var(models, 1, 0).value, 2.5)

    def test_max_across_scenarios(self):
        ds = [SlamDirective("v[*]", True, ("max",), 1.0)]
        ext, _, models = _build(ds, [
            [{"lb": 0.0, "ub": 10.0, "value": 5.5}],
            [{"lb": 0.0, "ub": 10.0, "value": 2.5}],
        ])
        ext._slam_one()
        self.assertAlmostEqual(_var(models, 0, 0).value, 5.5)
        self.assertAlmostEqual(_var(models, 1, 0).value, 5.5)

    def test_infinite_bound_falls_through(self):
        # lb is not finite (unbounded below) -> skip lb, use anywhere
        ds = [SlamDirective("v[*]", True, ("lb", "anywhere"), 1.0)]
        ext, _, models = _build(
            ds, [[{"lb": None, "ub": None, "value": 4.0, "xbar": 4.0}]])
        ext._slam_one()
        self.assertAlmostEqual(_var(models, 0, 0).value, 4.0)  # anywhere -> xbar

    def test_no_applicable_direction_is_skipped(self):
        # only lb requested, but var is unbounded below -> nothing to slam
        ds = [SlamDirective("v[*]", True, ("lb",), 1.0)]
        ext, _, models = _build(
            ds, [[{"lb": None, "ub": None, "value": 4.0, "xbar": 4.0}]])
        ext._slam_one()
        self.assertFalse(_var(models, 0, 0).fixed)


# --------------------------------------------------------------------------- #
# action -- selection / eligibility / stickiness
# --------------------------------------------------------------------------- #
class Test_selection(unittest.TestCase):
    def test_highest_priority_first(self):
        ds = [
            SlamDirective("v[0]", True, ("lb",), 10.0),
            SlamDirective("v[1]", True, ("lb",), 50.0),
        ]
        ext, _, models = _build(ds, [[
            {"lb": 1.0, "ub": 9.0, "value": 5.0},
            {"lb": 2.0, "ub": 9.0, "value": 5.0},
        ]])
        ext._slam_one()
        self.assertFalse(_var(models, 0, 0).fixed)  # priority 10 -> not yet
        self.assertTrue(_var(models, 0, 1).fixed)   # priority 50 -> slammed
        self.assertEqual(_var(models, 0, 1).value, 2.0)

    def test_priority_tie_broken_by_name(self):
        ds = [SlamDirective("v[*]", True, ("lb",), 5.0)]  # both same priority
        ext, _, models = _build(ds, [[
            {"lb": 1.0, "ub": 9.0, "value": 5.0},
            {"lb": 2.0, "ub": 9.0, "value": 5.0},
        ]])
        ext._slam_one()
        self.assertTrue(_var(models, 0, 0).fixed)   # 'v[0]' < 'v[1]'
        self.assertFalse(_var(models, 0, 1).fixed)

    def test_sticky_one_per_event(self):
        ds = [
            SlamDirective("v[0]", True, ("lb",), 10.0),
            SlamDirective("v[1]", True, ("lb",), 50.0),
        ]
        ext, _, models = _build(ds, [[
            {"lb": 1.0, "ub": 9.0, "value": 5.0},
            {"lb": 2.0, "ub": 9.0, "value": 5.0},
        ]])
        ext._slam_one()  # slams v[1] (priority 50)
        ext._slam_one()  # v[1] sticky -> now slams v[0]
        self.assertTrue(_var(models, 0, 0).fixed)
        self.assertTrue(_var(models, 0, 1).fixed)
        self.assertEqual(len(ext._slammed), 2)

    def test_surrogate_skipped(self):
        ds = [SlamDirective("v[*]", True, ("lb",), 1.0)]
        ext, _, models = _build(
            ds, [[{"lb": 1.0, "ub": 9.0, "value": 5.0, "surrogate": True}]])
        ext._slam_one()
        self.assertFalse(_var(models, 0, 0).fixed)

    def test_modeler_fixed_skipped(self):
        ds = [SlamDirective("v[*]", True, ("lb",), 1.0)]
        ext, _, models = _build(
            ds, [[{"lb": 1.0, "ub": 9.0, "value": 5.0, "modeler_fixed": True}]])
        ext._slam_one()
        self.assertTrue(_var(models, 0, 0).fixed)
        self.assertEqual(_var(models, 0, 0).value, 5.0)  # not moved to lb

    def test_already_fixed_skipped(self):
        ds = [SlamDirective("v[*]", True, ("lb",), 1.0)]
        ext, _, models = _build(ds, [[{"lb": 1.0, "ub": 9.0, "value": 5.0}]])
        _var(models, 0, 0).fix(7.0)  # fixed after pre_iter0 (e.g. another fixer)
        ext._slam_one()
        self.assertEqual(_var(models, 0, 0).value, 7.0)  # left alone

    def test_unmatched_nonant_never_slammed(self):
        # v[1] is matched by no rule -> never slammed (the coverage default)
        ds = [SlamDirective("v[0]", True, ("lb",), 1.0)]
        ext, _, models = _build(ds, [[
            {"lb": 1.0, "ub": 9.0, "value": 5.0},
            {"lb": 2.0, "ub": 9.0, "value": 5.0},
        ]])
        ext._slam_one()
        self.assertTrue(_var(models, 0, 0).fixed)    # v[0] matched -> slammed
        self.assertFalse(_var(models, 0, 1).fixed)   # v[1] unmatched -> never


class Test_zero_match(unittest.TestCase):
    def test_zero_match_pattern_errors(self):
        # a pattern matching no nonant is almost always a typo -> hard error
        ds = [SlamDirective("typo[*]", True, ("lb",), 1.0)]
        with self.assertRaises(ValueError):
            _build(ds, [[{"lb": 1.0, "ub": 9.0, "value": 5.0}]])

    def test_error_names_the_directives_file(self):
        path = _write_csv("name,directions\ntypo[*],lb\n")
        try:
            _m, sc = _make_scenario([{"lb": 1.0, "ub": 9.0, "value": 5.0}])
            opt = _make_opt([sc], {"directives_file": path})
            ext = Slammer(opt)
            with self.assertRaises(ValueError) as cm:
                ext.pre_iter0()
            self.assertIn(path, str(cm.exception))
        finally:
            os.remove(path)

    def test_pattern_matched_on_another_rank_is_ok(self):
        # a pattern that matches no *local* nonant but is reported as matched by
        # the cross-rank reduction must not error (simulated via the fake comm)
        class _OrComm:
            def Allreduce(self, sendbuf, recvbuf, op=None):
                recvbuf[0][:] = 1  # some other rank matched everything
        ds = [SlamDirective("only_elsewhere[*]", True, ("lb",), 1.0)]
        _m, sc = _make_scenario([{"lb": 1.0, "ub": 9.0, "value": 5.0}])
        opt = _make_opt([sc], {"directives": ds})
        opt.comms[NDN] = _OrComm()
        ext = Slammer(opt)
        ext.pre_iter0()  # must not raise


# --------------------------------------------------------------------------- #
# trigger + config (backward compatibility contract)
# --------------------------------------------------------------------------- #
class Test_trigger(unittest.TestCase):
    def test_iteration_for_slam_predicate(self):
        ds = [SlamDirective("v[*]", True, ("lb",), 1.0)]
        ext, _, _ = _build(ds, [[{"lb": 0.0, "ub": 1.0, "value": 0.0}]],
                           slam_start_iter=3, iters_between_slams=2)
        self.assertFalse(ext.iteration_for_slam(1))
        self.assertFalse(ext.iteration_for_slam(2))
        self.assertTrue(ext.iteration_for_slam(3))
        self.assertFalse(ext.iteration_for_slam(4))
        self.assertTrue(ext.iteration_for_slam(5))

    def test_miditer_respects_trigger(self):
        ds = [SlamDirective("v[*]", True, ("lb",), 1.0)]
        ext, opt, models = _build(ds, [[{"lb": 2.0, "ub": 9.0, "value": 5.0}]],
                                  slam_start_iter=5, iters_between_slams=1)
        opt._PHIter = 2
        ext.miditer()
        self.assertFalse(_var(models, 0, 0).fixed)  # before start
        opt._PHIter = 5
        ext.miditer()
        self.assertTrue(_var(models, 0, 0).fixed)


def _cfg_with_rho_seeded():
    """A Config with slamming_args plus the rho flags Config.checker() sums, so
    the checker can run in isolation."""
    cfg = config.Config()
    cfg.slamming_args()
    for n in ("grad_rho", "sensi_rho", "coeff_rho", "sep_rho"):
        cfg.quick_assign(n, bool, False)
    return cfg


class Test_backward_compat(unittest.TestCase):
    def test_no_slamming_options_is_fine(self):
        cfg = _cfg_with_rho_seeded()
        cfg.checker()  # must not raise

    def test_slam_option_without_file_errors(self):
        cfg = _cfg_with_rho_seeded()
        cfg.slam_start_iter = 5
        with self.assertRaises(ValueError):
            cfg.checker()

    def test_iters_between_without_file_errors(self):
        cfg = _cfg_with_rho_seeded()
        cfg.iters_between_slams = 3
        with self.assertRaises(ValueError):
            cfg.checker()

    def test_file_with_options_is_fine(self):
        cfg = _cfg_with_rho_seeded()
        cfg.slamming_directives_file = "directives.csv"
        cfg.slam_start_iter = 5
        cfg.checker()  # must not raise


class Test_valid_directions(unittest.TestCase):
    def test_all_tokens_present(self):
        self.assertEqual(
            set(VALID_DIRECTIONS),
            {"lb", "ub", "nearest", "anywhere", "min", "max"})


if __name__ == "__main__":
    unittest.main()
