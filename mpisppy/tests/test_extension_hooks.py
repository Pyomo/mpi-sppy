###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for the hooks of individual extensions, as opposed to
test_extensions.py, which covers the extension base classes.

These build a real PH object for its scenarios and Params, but never solve, so
no solver is required.
"""

import unittest

import mpisppy.opt.ph
import mpisppy.tests.examples.farmer as farmer


def _make_ph(**option_overrides):
    options = {
        "solver_name": "gurobi",     # never used; nothing here solves
        "PHIterLimit": 10,
        "defaultPHrho": 1,
        "convthresh": 1e-8,
        "verbose": False,
        "display_timing": False,
        "display_progress": False,
        "smoothed": 0,
        "toc": False,
    }
    options.update(option_overrides)
    return mpisppy.opt.ph.PH(
        options, [f"Scenario{i+1}" for i in range(3)], farmer.scenario_creator,
        farmer.scenario_denouement,
        scenario_creator_kwargs={"crops_multiplier": 1})


class TestIntegerRelaxThenEnforce(unittest.TestCase):

    def test_unrelaxes_at_most_once_per_pass(self):
        # The three conditions in miditer are independent, and more than one
        # can hold in the same pass -- past the time fraction and past the
        # iteration fraction. Pyomo's undo deletes _relaxed_integer_vars, so
        # the second call in one pass raises.
        from mpisppy.extensions.integer_relax_then_enforce import (
            IntegerRelaxThenEnforce)
        ph = _make_ph(time_limit=600, PHIterLimit=100)
        ph.PH_Prep()
        for s in ph.local_scenarios.values():
            s._solver_plugin = None       # not persistent; no solver needed
        irte = IntegerRelaxThenEnforce(ph)
        irte.pre_iter0()
        self.assertTrue(irte._integers_relaxed)

        ph.start_time -= 320              # past 600 * 0.5
        ph._PHIter = 51                   # past 100 * 0.5
        ph.conv = 1.0
        irte.miditer()                    # must not raise
        self.assertFalse(irte._integers_relaxed)

    def test_the_time_condition_is_decided_on_every_rank(self):
        # The time fraction is a per-process clock and _unrelax_integers is
        # not collective, so deciding it per rank leaves some ranks solving
        # MIPs and the rest LPs in the same iteration.
        from mpisppy.extensions.integer_relax_then_enforce import (
            IntegerRelaxThenEnforce)
        ph = _make_ph(time_limit=600, PHIterLimit=100)
        ph.PH_Prep()
        for s in ph.local_scenarios.values():
            s._solver_plugin = None
        irte = IntegerRelaxThenEnforce(ph)
        irte.pre_iter0()

        reduced = []
        real = ph.allreduce_or

        def counting(val):
            reduced.append(val)
            return real(val)

        ph.allreduce_or = counting
        ph.start_time -= 320
        ph._PHIter = 1
        ph.conv = 1.0
        irte.miditer()
        self.assertEqual(reduced, [True],
                         "the time condition was decided without a reduction")

    def test_each_condition_unrelaxes_on_its_own(self):
        # Three conditions, three returns. The time one is covered above; a
        # run can also reach the fraction of its iteration limit, or come
        # near convergence, with the clock nowhere near the limit.
        from mpisppy.extensions.integer_relax_then_enforce import (
            IntegerRelaxThenEnforce)
        for label, phiter, conv in (("iterations", 51, 1.0),
                                    ("convergence", 1, 1e-9)):
            with self.subTest(condition=label):
                ph = _make_ph(time_limit=600, PHIterLimit=100)
                ph.PH_Prep()
                for s in ph.local_scenarios.values():
                    s._solver_plugin = None
                irte = IntegerRelaxThenEnforce(ph)
                irte.pre_iter0()
                ph._PHIter = phiter
                ph.conv = conv
                irte.miditer()
                self.assertFalse(irte._integers_relaxed,
                                 f"the {label} condition did not unrelax")

    def test_no_reduction_when_there_is_no_time_limit(self):
        # time_limit is rank-identical, so an unset one is False everywhere
        # and reducing it every iteration of the hub's loop is an Allreduce
        # for a known answer.
        from mpisppy.extensions.integer_relax_then_enforce import (
            IntegerRelaxThenEnforce)
        ph = _make_ph(PHIterLimit=100)
        ph.PH_Prep()
        for s in ph.local_scenarios.values():
            s._solver_plugin = None
        irte = IntegerRelaxThenEnforce(ph)
        irte.pre_iter0()

        reduced = []
        ph.allreduce_or = lambda val: reduced.append(val) or val
        ph._PHIter = 1
        ph.conv = 1.0
        irte.miditer()
        self.assertEqual(reduced, [])


class TestCrossScenarioDisableEnable(unittest.TestCase):

    def test_w_is_put_back_when_prox_was_already_off(self):
        # _disable_W_and_prox has three branches; the one for "W on, prox
        # already off" recorded its intention on a misspelled attribute, so
        # reenable_W stayed False and _enable_W_and_prox never put W back --
        # every later iteration solved without the dual term while W_disabled
        # reported the run as normal.
        from mpisppy.extensions.cross_scen_extension import (
            CrossScenarioExtension)
        ph = _make_ph()
        ph.PH_Prep()
        for s in ph.local_scenarios.values():
            s._mpisppy_model.W_on = 1
            s._mpisppy_model.prox_on = 0
        ext = CrossScenarioExtension(ph)
        ext._disable_W_and_prox()
        self.assertTrue(ph.W_disabled, "W was not turned off")
        ext._enable_W_and_prox()
        self.assertFalse(ph.W_disabled, "W was never put back")


class TestWtrackerReportNames(unittest.TestCase):

    def test_an_unset_prefix_is_empty_not_None(self):
        # The three report names are built from file_prefix; unset, they were
        # named after a stringified None ("None_summary_iter5_rank0.txt").
        from mpisppy.extensions.wtracker_extension import Wtracker_extension
        ph = _make_ph()
        ph.options["wtracker_options"] = {"wlen": 2}
        ext = Wtracker_extension(ph)
        seen = {}

        def fake_report(wlen, reportlen=None, stdevthresh=None, file_prefix=''):
            seen["prefix"] = file_prefix

        ext.wtracker.report_by_moving_stats = fake_report
        ext.post_everything()
        self.assertEqual(seen["prefix"], "")


class TestFWPHSmoothing(unittest.TestCase):

    def test_fwph_refuses_smoothing(self):
        # FWPH attaches no proximal term, and smoothing rides on that term, so
        # PH_Prep creates neither p nor beta. At smoothed == 2 Iter0's rescale
        # then dies on the missing p; at 1 the run finishes with the smoothing
        # silently doing nothing. Subgradient refuses the same combination.
        from mpisppy.opt.fwph import FWPH
        for smoothed in (1, 2):
            with self.subTest(smoothed=smoothed):
                options = {
                    "solver_name": "gurobi", "PHIterLimit": 2,
                    "defaultPHrho": 1, "convthresh": 1e-8, "verbose": False,
                    "display_timing": False, "display_progress": False,
                    "smoothed": smoothed, "toc": False,
                    "FW_iter_limit": 5, "FW_weight": 0.0,
                    "FW_conv_thresh": 1e-4, "stop_check_tol": 1e-5,
                    "FW_LP_start_iterations": 0, "FW_verbose": False,
                    "mip_solver_options": {}, "qp_solver_options": {},
                    "iter0_solver_options": None, "iterk_solver_options": None,
                }
                fw = FWPH(options, [f"Scenario{i+1}" for i in range(3)],
                          farmer.scenario_creator, farmer.scenario_denouement,
                          scenario_creator_kwargs={"crops_multiplier": 1})
                with self.assertRaisesRegex(RuntimeError, "smoothing"):
                    fw.fwph_main()

    def test_the_fwph_spoke_zeroes_smoothing(self):
        # A caller reusing a PH hub's options dict brings smoothed along, and
        # options_check only fills in a missing key -- so the spoke has to
        # zero it the way fwph_hub does, or the run aborts at fwph_main.
        from mpisppy.utils import config
        import mpisppy.utils.cfg_vanilla as vanilla
        cfg = config.Config()
        cfg.popular_args()
        cfg.ph_args()
        cfg.two_sided_args()
        cfg.fwph_args()
        cfg.num_scens_required()
        cfg.num_scens = 3
        cfg.solver_name = "gurobi"
        cfg.default_rho = 1
        beans = (cfg, farmer.scenario_creator, farmer.scenario_denouement,
                 [f"Scenario{i+1}" for i in range(3)])
        spoke = vanilla.fwph_spoke(*beans)
        self.assertEqual(spoke["opt_kwargs"]["options"]["smoothed"], 0)


class TestAgnosticGuestCallouts(unittest.TestCase):

    def test_the_gurobipy_guests_take_the_parameter_the_host_passes(self):
        # SPOpt passes {"s": s} for these two callouts, and callout_agnostic
        # calls fct(Ag=self, **kwargs), so a guest declaring anything else is
        # a TypeError as soon as the host calls it. Read from the source
        # rather than imported: these example modules pull in a farmer module
        # that is not importable from the test environment.
        import ast
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[2]
        guests = [
            root / "mpisppy" / "agnostic" / "examples"
                 / "farmer_gurobipy_model.py",
            root / "examples" / "farmer" / "agnostic"
                 / "farmer_gurobipy_agnostic.py",
        ]
        wanted = {"_restore_original_fixedness", "_fix_root_nonants"}
        for guest in guests:
            if not guest.exists():          # examples/ is not always shipped
                continue
            tree = ast.parse(guest.read_text())
            found = {n.name: [a.arg for a in n.args.args]
                     for n in tree.body
                     if isinstance(n, ast.FunctionDef) and n.name in wanted}
            self.assertEqual(set(found), wanted, f"{guest.name}: {found}")
            for fname, params in found.items():
                with self.subTest(guest=guest.name, callout=fname):
                    self.assertEqual(params, ["Ag", "s"])


if __name__ == "__main__":
    unittest.main()
