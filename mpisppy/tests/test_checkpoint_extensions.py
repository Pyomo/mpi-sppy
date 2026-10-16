###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Extension and converger state across a checkpoint (design phase 3).

Everything checkpointed before this phase lived on a scenario model, so it came
back with the dilled models and came back consistent with the variable values
it pairs with. This phase covers the state extensions keep on *themselves*,
which no model carries and which was therefore lost outright.

That loss is the quiet kind. Nothing is missing, nothing raises, the run
continues and reports numbers -- but an extension whose behavior depends on
what it did in earlier iterations takes a *different action* at the first
iteration after the stop than the uninterrupted run would have, and the two
diverge from there. A rho updater with no record of the previous xbar skips a
rho update. A fixer whose per-variable countdowns were reset waits longer to
fix. A converger with no previous xbar measures a dual residual of zero and can
stop the run early.

So each case here is an A/B comparison in the shape design section 11.1 lays
out -- an uninterrupted run of N iterations against one stopped at k and
resumed -- with the extension attached to both. Farmer is a deterministic LP,
so "the extension state was carried" and "the runs are bit-identical" are the
same statement, and the assertions say so directly rather than inferring it
from a summary number.

Two of these are regressions rather than divergences: ``--sep-rho`` and its
siblings *crashed* on the first iteration after a resume, and the fixer's
per-variable counts were being zeroed by its own setup hook on the way back in.
"""

import contextlib
import io
import json
import os
import pickle
import tempfile
import types
import unittest

import mpisppy.tests.examples.farmer as farmer
import mpisppy.tests.examples.sizes.sizes as sizes
import mpisppy.utils.checkpointing as checkpointing
from mpisppy.extensions.checkpointer import Checkpointer
from mpisppy.extensions.extension import Extension, MultiExtension
from mpisppy.opt.ph import PH
from mpisppy.tests.utils import get_solver

solver_available, solver_name, persistent_available, persistent_solver_name = \
    get_solver()

FARMER_SCENARIOS = ["scen0", "scen1", "scen2"]
FARMER_KWARGS = {"use_integer": False, "crops_multiplier": 1}
SIZES_SCENARIOS = ["Scenario1", "Scenario2", "Scenario3"]
SIZES_KWARGS = {"scenario_count": 3}


def _options(max_iters, ckpt_dir=None, resume_from=None, **overrides):
    options = {
        "solver_name": solver_name,
        "PHIterLimit": max_iters,
        "defaultPHrho": 1.0,
        # Never converge early: the A/B comparison needs both sides to run the
        # same iterations.
        "convthresh": -1.0,
        "verbose": False,
        "display_progress": False,
        "display_timing": False,
        "display_convergence_detail": False,
        "iter0_solver_options": None,
        "iterk_solver_options": None,
        "tee-rank0-solves": False,
        "smoothed": 0,
        "time_limit": None,
    }
    if ckpt_dir is not None:
        options["checkpoint_dir"] = ckpt_dir
        options["checkpoint_backend"] = checkpointing.DILL_RELOAD_BACKEND
        options["checkpoint_every_iterations"] = 1
    if resume_from is not None:
        options["resume_from"] = resume_from
    options.update(overrides)
    return options


def _make_ph(options, ext_classes, model=farmer, scenario_names=None,
             creator_kwargs=None, **ph_kwargs):
    """A PH hub with the checkpointer and the extensions under test attached."""
    classes = list(ext_classes)
    if "checkpoint_dir" in options or "resume_from" in options:
        classes.insert(0, Checkpointer)
    return PH(
        options,
        scenario_names if scenario_names is not None else FARMER_SCENARIOS,
        model.scenario_creator,
        model.scenario_denouement,
        scenario_creator_kwargs=(creator_kwargs if creator_kwargs is not None
                                 else FARMER_KWARGS),
        extensions=MultiExtension,
        extension_kwargs={"ext_classes": classes},
        **ph_kwargs,
    )


def _extension(ph, cls):
    for candidate in ph.extobject.extdict.values():
        if isinstance(candidate, cls):
            return candidate
    raise AssertionError(f"no {cls.__name__} attached")


def _primal_snapshot(ph):
    """Nonant values, fixedness and the per-nonant Params, keyed by name."""
    snap = {}
    for sname, s in ph.local_scenarios.items():
        for ndn_i, v in s._mpisppy_data.nonant_indices.items():
            snap[f"{sname}|x|{v.name}"] = v._value
            snap[f"{sname}|fixed|{v.name}"] = float(v.is_fixed())
            for pname in ("W", "rho", "xbars"):
                param = getattr(s._mpisppy_model, pname, None)
                if param is not None:
                    snap[f"{sname}|{pname}|{ndn_i}"] = float(param[ndn_i]._value)
    return snap


class _ABMixin:
    """Uninterrupted vs stop-and-resume, with the extension under test.

    The legs run in one process, which is enough here: what is under test is
    whether state survives the *checkpoint*, and a fresh interpreter is what
    ``test_checkpoint.py`` and the cylinders harnesses already pin.
    """

    N = 5
    STOP = 2
    MODEL = farmer
    SCENARIOS = None
    CREATOR_KWARGS = None
    #: Extra options every leg needs to configure the extension under test.
    EXTRA_OPTIONS = {}
    PH_KWARGS = {}

    def ext_classes(self):
        raise NotImplementedError

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _ph(self, max_iters, **ckpt_kwargs):
        options = _options(max_iters, **ckpt_kwargs, **self.EXTRA_OPTIONS)
        return _make_ph(options, self.ext_classes(), model=self.MODEL,
                        scenario_names=self.SCENARIOS,
                        creator_kwargs=self.CREATOR_KWARGS, **self.PH_KWARGS)

    def run_ab(self):
        """Returns (reference, stopped, resumed) PH objects, all run."""
        reference = self._ph(self.N)
        reference.ph_main()
        stopped = self._ph(self.STOP, ckpt_dir=self.ckpt_dir)
        stopped.ph_main()
        # --max-iterations counts this run's iterations, so the resumed leg
        # asks for the ones that are left rather than for the study total.
        resumed = self._ph(self.N - self.STOP, resume_from=self.ckpt_dir)
        resumed.ph_main()
        self.assertTrue(resumed._resumed_from_checkpoint,
                        msg="the third leg started from scratch")
        return reference, stopped, resumed

    def assert_bit_identical(self, reference, resumed):
        want, got = _primal_snapshot(reference), _primal_snapshot(resumed)
        self.assertEqual(set(want), set(got))
        worst = max((abs(want[k] - got[k]) for k in want), default=0.0)
        self.assertEqual(
            worst, 0.0,
            msg=f"the resumed run differs from the uninterrupted one by "
                f"{worst}; this instance is deterministic, so a resume that "
                f"carried the extension's state must land bit-identically")


@unittest.skipIf(not solver_available, "no solver is available")
class TestNormRhoUpdaterResume(_ABMixin, unittest.TestCase):
    """A rho updater that compares against the previous iteration's xbar.

    Without the restore this does not merely differ in a statistic. `miditer`
    branches on whether it has a previous xbar at all, so a resumed run takes
    the "first time through" branch: it snapshots and performs **no rho update
    that iteration**, then carries the resulting rho -- different from the
    uninterrupted run's -- for the rest of the run.
    """

    #: The default factor of 100 never changes rho on farmer, on either leg,
    #: and a restore that did nothing then passed. At 1.0 rho moves.
    EXTRA_OPTIONS = {"norm_rho_options": {"verbose": False,
                                          "primal_dual_difference_factor": 1.0}}

    def ext_classes(self):
        from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
        return [NormRhoUpdater]

    def test_resume_is_bit_identical(self):
        reference, _, resumed = self.run_ab()
        self.assert_bit_identical(reference, resumed)

    def test_rho_really_changes_after_the_stop(self):
        """Otherwise the comparison above cannot see the restore."""
        _, stopped, resumed = self.run_ab()
        before = {k: v for k, v in _primal_snapshot(stopped).items()
                  if "|rho|" in k}
        after = {k: v for k, v in _primal_snapshot(resumed).items()
                 if "|rho|" in k}
        self.assertNotEqual(before, after,
                            msg="no rho changed after the resume")

    def test_the_previous_xbar_is_actually_restored(self):
        """Named directly, so a passing comparison cannot be a coincidence."""
        from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
        _, stopped, resumed = self.run_ab()
        saved = _extension(stopped, NormRhoUpdater)._prev_avg
        # The resumed run has moved on by the time it finishes, so compare
        # what the checkpoint holds rather than the extension's final state.
        with open(os.path.join(self.ckpt_dir, "manifest.json")) as f:
            generation = json.load(f)["generation"]
        leaf = _read_leaf(self.ckpt_dir, generation)
        carried = leaf["extension_state"]["extensions"]["NormRhoUpdater"]
        self.assertEqual(carried["prev_avg"], saved)
        self.assertTrue(saved, msg="the extension recorded no previous xbar, "
                                   "so this test proves nothing")
        self.assertTrue(_extension(resumed, NormRhoUpdater)._prev_avg)


@unittest.skipIf(not solver_available, "no solver is available")
class TestPrimalDualRhoResume(_ABMixin, unittest.TestCase):
    """Compares each iteration's xbars against the previous ones.

    Without the restore a resumed run takes the "first time through" branch
    of miditer: it records the xbars and makes no rho update that iteration,
    which an uninterrupted run does make.
    """

    N = 6
    STOP = 3
    EXTRA_OPTIONS = {"primal_dual_rho_options": {"verbose": False,
                                                 "rho_update_threshold": 1.5}}

    def ext_classes(self):
        from mpisppy.extensions.primal_dual_rho import PrimalDualRho
        return [PrimalDualRho]

    def test_resume_is_bit_identical(self):
        reference, _, resumed = self.run_ab()
        self.assert_bit_identical(reference, resumed)

    def test_rho_really_changes_after_the_stop(self):
        """Otherwise the comparison above cannot see the restore."""
        _, stopped, resumed = self.run_ab()
        before = {k: v for k, v in _primal_snapshot(stopped).items()
                  if "|rho|" in k}
        after = {k: v for k, v in _primal_snapshot(resumed).items()
                 if "|rho|" in k}
        self.assertNotEqual(before, after,
                            msg="no rho changed after the resume")


@unittest.skipIf(not solver_available, "no solver is available")
class TestMultRhoUpdaterResume(_ABMixin, unittest.TestCase):
    """A rho updater that anchors a ratio once and scales from it forever.

    A resumed run that forgot the anchor re-derives it from the *checkpointed*
    rho and the current convergence metric, so every later rho is scaled from
    a baseline the uninterrupted run never had.
    """

    EXTRA_OPTIONS = {"mult_rho_options": {"verbose": False}}

    def ext_classes(self):
        from mpisppy.extensions.mult_rho_updater import MultRhoUpdater
        return [MultRhoUpdater]

    def test_resume_is_bit_identical(self):
        reference, _, resumed = self.run_ab()
        self.assert_bit_identical(reference, resumed)

    def test_the_anchor_is_restored_not_re_derived(self):
        from mpisppy.extensions.mult_rho_updater import MultRhoUpdater
        _, stopped, resumed = self.run_ab()
        before = _extension(stopped, MultRhoUpdater)
        after = _extension(resumed, MultRhoUpdater)
        self.assertIsNotNone(before._first_rho,
                             msg="the anchor was never set, so this test "
                                 "proves nothing")
        self.assertEqual(after.first_c, before.first_c)
        self.assertEqual(after._first_rho, before._first_rho)


@unittest.skipIf(not solver_available, "no solver is available")
class TestSepRhoResume(_ABMixin, unittest.TestCase):
    """A regression: this configuration used to *crash* on resume.

    The dynamic-rho extensions track a W history through a `WTracker`, and
    `W_diff` indexes it at the two iterations before the current one. A resumed
    run had none of them, so the first iteration after the stop died with a
    bare `KeyError` -- no warning, no partial result, just a traceback out of a
    utility three call levels below the extension.

    Its own rho is *not* recomputed at the resume (the checkpointed rho, with
    whatever adaptation it carries, is the right starting point and the
    extension already knew that), so the assertion here is the ordinary one:
    the run continues, and continues identically.
    """

    #: With both criteria off, rho is never recomputed after iteration 0, so
    #: this class exercises the W history alone. The subclass below turns them
    #: on.
    PRIMAL_CRIT = False
    DUAL_CRIT = False
    THRESH = 0.5

    def ext_classes(self):
        from mpisppy.extensions.sep_rho import SepRho
        return [SepRho]

    def _ph(self, max_iters, **ckpt_kwargs):
        # SepRho reads its settings off a cfg handed to it in the options.
        from mpisppy.utils.config import Config
        cfg = Config()
        cfg.add_to_config("sep_rho_multiplier", description="", domain=float,
                          default=1.0)
        cfg.add_to_config("dynamic_rho_primal_crit", description="",
                          domain=bool, default=self.PRIMAL_CRIT)
        cfg.add_to_config("dynamic_rho_dual_crit", description="",
                          domain=bool, default=self.DUAL_CRIT)
        for thresh in ("dynamic_rho_primal_thresh", "dynamic_rho_dual_thresh"):
            cfg.add_to_config(thresh, description="", domain=float,
                              default=self.THRESH)
        options = _options(max_iters, **ckpt_kwargs)
        options["sep_rho_options"] = {"cfg": cfg}
        return _make_ph(options, self.ext_classes())

    def test_a_resumed_run_does_not_crash(self):
        """The crash regression, stated as plainly as it can be."""
        reference, _, resumed = self.run_ab()
        self.assertEqual(resumed._PHIter, self.N)
        self.assert_bit_identical(reference, resumed)

    def test_the_w_history_the_next_diff_reads_is_carried(self):
        from mpisppy.extensions.sep_rho import SepRho
        _, stopped, resumed = self.run_ab()
        tracker = _extension(stopped, SepRho).wt
        carried = _extension(resumed, SepRho).wt
        # W_diff reads its own ph_iter and one before it; those are the two
        # the checkpoint has to carry, and are what used to be missing.
        for wanted in (tracker.ph_iter, tracker.ph_iter - 1):
            self.assertIn(wanted, carried.local_Ws,
                          msg=f"local_Ws[{wanted}] was not carried; the next "
                              f"W_diff would raise KeyError")


@unittest.skipIf(not solver_available, "no solver is available")
class TestSepRhoResumeWithRhoUpdates(TestSepRhoResume):
    """The same, with rho recomputed after the stop.

    The first recompute after a resume used to read the cost coefficients off
    the resumed objective -- which by then holds W and the quadratic prox --
    and died with "nonant_cost_coefficient found nonlinear variables".
    """

    PRIMAL_CRIT = True
    DUAL_CRIT = True
    N = 8
    STOP = 3


@unittest.skipIf(not solver_available, "no solver is available")
class TestSepRhoResumeWithDualCriterion(TestSepRhoResumeWithRhoUpdates):
    """The criterion that reads the convergence history.

    The dual criterion waits for four entries in its cache, so a resume that
    started the caches empty skipped the rho updates of the first resumed
    iterations and then carried a different rho for the rest of the run. With
    a threshold every difference passes, it updates whenever it has the
    history, which is what makes the carried history visible.
    """

    PRIMAL_CRIT = False
    DUAL_CRIT = True
    THRESH = 100.0

    def test_rho_really_is_recomputed_after_the_stop(self):
        """Otherwise this class guards nothing its parent does not."""
        _, stopped, resumed = self.run_ab()
        before = {k: v for k, v in _primal_snapshot(stopped).items()
                  if "|rho|" in k}
        after = {k: v for k, v in _primal_snapshot(resumed).items()
                 if "|rho|" in k}
        self.assertTrue(before)
        self.assertNotEqual(before, after,
                            msg="no rho changed after the resume, so the "
                                "resumed leg never recomputed it")


@unittest.skipIf(not solver_available, "no solver is available")
class TestWtrackerResume(_ABMixin, unittest.TestCase):
    """The W tracker reports moving statistics over the last ``wlen``
    iterations when the run ends. A resumed run holds only the W sets it
    grabbed itself, so a window wider than the resumed leg reached back past
    the stop and the run died with a KeyError in ``post_everything`` -- after
    every solve had been paid for. The extension now carries the window's
    worth of W sets, and the resumed run's reports match the uninterrupted
    run's.
    """

    N = 5
    STOP = 3
    #: Wider than the resumed leg (two iterations), so the report reads
    #: iterations the resumed run never saw.
    WLEN = 3

    def ext_classes(self):
        from mpisppy.extensions.wtracker_extension import Wtracker_extension
        return [Wtracker_extension]

    def setUp(self):
        super().setUp()
        self._legs = 0

    def _ph(self, max_iters, **ckpt_kwargs):
        # The report files are named for the iteration they close at, which
        # is the same for the reference and the resumed leg; a prefix per leg
        # keeps them apart.
        self._legs += 1
        prefix = os.path.join(self._tmp.name, f"leg{self._legs}")
        options = _options(max_iters, **ckpt_kwargs)
        options["wtracker_options"] = {"wlen": self.WLEN,
                                       "file_prefix": prefix}
        return _make_ph(options, self.ext_classes())

    def _report(self, leg, kind, ph):
        path = os.path.join(self._tmp.name,
                            f"leg{leg}_{kind}_iter{self.N}_rank{ph.global_rank}.csv")
        with open(path) as f:
            return f.read()

    def test_the_window_reaches_back_past_the_stop(self):
        self.assertGreater(self.WLEN + 1, self.N - self.STOP,
                           msg="the window must be wider than the resumed "
                               "leg for this test to say anything")
        reference, _, resumed = self.run_ab()
        for kind in ("stdev", "cv"):
            self.assertEqual(self._report(3, kind, resumed),
                             self._report(1, kind, reference),
                             msg=f"the resumed run's {kind} report differs "
                                 f"from the uninterrupted run's")

    def test_the_carried_wlen_is_kept_for_the_report_to_judge(self):
        """The restore records what the checkpoint was written with and says
        nothing about it. Whether a wider window can be filled depends on how
        far this leg runs, which this hook -- at the end of Iter0 -- cannot
        know; the two classes below run that out."""
        from mpisppy.extensions.wtracker_extension import Wtracker_extension
        _, stopped, _ = self.run_ab()
        ext = _extension(stopped, Wtracker_extension)
        state = ext.checkpoint_state()
        ext.wlen = self.WLEN + 5        # what a wider --wlen would ask for
        self.assertIsNone(
            ext.restore_state(state),
            msg="the restore judged a window it cannot yet see the end of")
        self.assertEqual(ext._carried_wlen, self.WLEN)

    def test_what_it_costs_is_the_windows_worth_not_the_first_write(self):
        """This is said at the first write, where the window holds one set of
        the wlen+1 that every write after it carries. Reporting what happened
        to be in hand there reported a fraction of what the option costs."""
        from mpisppy.extensions.wtracker_extension import Wtracker_extension
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _, stopped, _ = self.run_ab()
        said = [line for line in out.getvalue().splitlines()
                if "Wtracker carries" in line]
        self.assertEqual(len(said), 1, msg=out.getvalue())
        one_set = next(iter(
            _extension(stopped, Wtracker_extension).wtracker.local_Ws.values()))
        per_set = sum(len(w) for w in one_set.values())
        self.assertIn(f"{self.WLEN + 1} W set(s)", said[0])
        self.assertIn(f"{(self.WLEN + 1) * per_set} values", said[0])

    def test_what_it_costs_is_said_once(self):
        """The window is the user's option, so its cost is theirs to see --
        and it does not change between writes, so it is said once."""
        from mpisppy.extensions.wtracker_extension import Wtracker_extension
        _, stopped, _ = self.run_ab()
        ext = _extension(stopped, Wtracker_extension)
        ext._size_tocced = False
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ext.checkpoint_state()
            ext.checkpoint_state()
        said = [line for line in out.getvalue().splitlines()
                if "Wtracker carries" in line]
        self.assertEqual(len(said), 1, msg=out.getvalue())
        self.assertIn(f"wlen {self.WLEN}", said[0])

    def test_the_windows_worth_is_carried_and_no_more(self):
        from mpisppy.extensions.wtracker_extension import Wtracker_extension
        _, stopped, resumed = self.run_ab()
        state = _extension(stopped, Wtracker_extension).checkpoint_state()
        self.assertEqual(sorted(state["local_Ws"]),
                         list(range(1, self.STOP + 1))[-(self.WLEN + 1):])
        carried = _extension(resumed, Wtracker_extension).wtracker.local_Ws
        self.assertEqual(sorted(carried), list(range(1, self.N + 1)))


class _WiderWindowMixin(_ABMixin):
    """The two legs are asked for different windows.

    The leg that stops tracks a narrow one; the leg that resumes asks for a
    wider one. That is the case that used to be judged in restore_state, at
    the end of Iter0 -- before the resumed leg had run an iteration, and so
    before anything could know whether it would fill the wider window itself.
    """

    #: What the checkpoint is written with, and what it therefore carries.
    STOPPED_WLEN = 1
    #: What the resumed leg -- and the uninterrupted reference -- ask for.
    RESUMED_WLEN = 5

    def ext_classes(self):
        from mpisppy.extensions.wtracker_extension import Wtracker_extension
        return [Wtracker_extension]

    def setUp(self):
        super().setUp()
        self._legs = 0

    def _ph(self, max_iters, **ckpt_kwargs):
        self._legs += 1
        # The leg that writes the checkpoint is the one asked for the narrow
        # window; the reference leg asks for the wide one, so its report is
        # the one the resumed leg's is compared against.
        wlen = (self.STOPPED_WLEN if "ckpt_dir" in ckpt_kwargs
                else self.RESUMED_WLEN)
        prefix = os.path.join(self._tmp.name, f"leg{self._legs}")
        options = _options(max_iters, **ckpt_kwargs)
        options["wtracker_options"] = {"wlen": wlen, "file_prefix": prefix}
        return _make_ph(options, self.ext_classes())

    def _leg_file(self, leg, kind, ph, ext="csv"):
        return os.path.join(
            self._tmp.name,
            f"leg{leg}_{kind}_iter{self.N}_rank{ph.global_rank}.{ext}")

    def run_ab_and_capture(self):
        """run_ab, plus everything the three legs said."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            reference, stopped, resumed = self.run_ab()
        return reference, stopped, resumed, out.getvalue()


@unittest.skipIf(not solver_available, "no solver is available")
class TestALegThatFillsTheWiderWindowIsNotWarnedAbout(_WiderWindowMixin,
                                                      unittest.TestCase):
    """A resumed leg long enough to fill the wider window on its own writes
    the report an uninterrupted run writes, and there is nothing to say about
    it. Judged at the restore, every such run was told its report covered
    less than an uninterrupted one -- while the report it went on to write
    was identical to the uninterrupted one's.
    """

    N = 8
    STOP = 3

    def test_the_wider_window_is_filled_and_nothing_is_said(self):
        reference, _, resumed, said = self.run_ab_and_capture()
        for kind in ("stdev", "cv"):
            with open(self._leg_file(3, kind, resumed)) as f:
                got = f.read()
            with open(self._leg_file(1, kind, reference)) as f:
                want = f.read()
            self.assertEqual(
                got, want,
                msg=f"the resumed run's {kind} report differs from the "
                    f"uninterrupted run's, so this leg did not fill the "
                    f"wider window and the case under test is not running")
        self.assertNotIn(
            "Wtracker_extension", said,
            msg="the resumed leg wrote the report an uninterrupted run "
                "writes, and was warned that its report was short anyway")


@unittest.skipIf(not solver_available, "no solver is available")
class TestALegThatCannotFillTheWiderWindowIsToldWhy(_WiderWindowMixin,
                                                    unittest.TestCase):
    """The other side of it: a resumed leg too short to fill the wider window
    gets no report, and the reason is the earlier leg's narrower one rather
    than the study having gone quiet."""

    N = 5
    STOP = 3

    def test_the_missing_report_names_the_window_the_checkpoint_carried(self):
        _, _, resumed, said = self.run_ab_and_capture()
        with open(self._leg_file(3, "summary", resumed, ext="txt")) as f:
            self.assertIn("Not enough iterations tracked", f.read(),
                          msg="this leg did fill the window, so there is "
                              "nothing for the warning to explain")
        self.assertIn(f"no report for window len {self.RESUMED_WLEN}", said)
        self.assertIn(f"written with wlen {self.STOPPED_WLEN}", said)


@unittest.skipIf(not solver_available, "no solver is available")
class TestTheSameWindowOnBothLegsStaysSilent(_WiderWindowMixin,
                                             unittest.TestCase):
    """A window the study has simply not run long enough for is the report's
    own business -- it says "not enough iterations tracked" for itself, and
    the two legs were asked the same question."""

    N = 5
    STOP = 3
    STOPPED_WLEN = 5
    RESUMED_WLEN = 5

    def test_a_short_window_both_legs_asked_for_is_not_explained_away(self):
        _, _, resumed, said = self.run_ab_and_capture()
        with open(self._leg_file(3, "summary", resumed, ext="txt")) as f:
            self.assertIn("Not enough iterations tracked", f.read())
        self.assertNotIn("Wtracker_extension", said)


class TestARestoreStateSentenceReachesTheUser(unittest.TestCase):
    """The plumbing: a sentence an extension returns from restore_state comes
    out with the resume's other warnings rather than being dropped.

    Wtracker_extension returned the only shipped one until its message moved
    to the end of the run, where what it reports is knowable, so this drives
    the mechanism with a stand-in.
    """

    class _HasSomethingToSay(Extension):
        def checkpoint_state(self):
            return {"n": 1}

        def restore_state(self, state):
            return "the counter it kept could not be put back exactly"

    def test_the_sentence_comes_out_with_the_resumes_warnings(self):
        ext = self._HasSomethingToSay.__new__(self._HasSomethingToSay)
        opt = types.SimpleNamespace(
            extobject=types.SimpleNamespace(
                extdict={"_HasSomethingToSay": ext}))
        warnings = checkpointing.restore_extension_state(
            opt, {"extensions": {"_HasSomethingToSay": {"n": 1}}})
        self.assertTrue(
            any("could not be put back exactly" in w for w in warnings),
            msg=f"the extension's message never reached the user: {warnings}")
        self.assertTrue(any("_HasSomethingToSay" in w for w in warnings),
                        msg=f"and nothing said which extension: {warnings}")


@unittest.skipIf(not solver_available, "no solver is available")
class TestFixerResume(_ABMixin, unittest.TestCase):
    """The fixer's per-variable countdowns, on a MIP that actually fixes.

    Those counts live on the scenario models, so they ride in the dill for
    free -- and then the fixer's own `post_iter0` hook, which runs on a resumed
    run too, zeroed every one of them on the way back in. A nonant one
    iteration short of its threshold restarted its countdown, so a resumed run
    fixed strictly later than the uninterrupted one.
    """

    MODEL = sizes
    SCENARIOS = SIZES_SCENARIOS
    CREATOR_KWARGS = SIZES_KWARGS
    N = 4
    STOP = 2

    def setUp(self):
        super().setUp()
        self.EXTRA_OPTIONS = {
            "fixeroptions": {
                "verbose": False,
                "boundtol": 0.01,
                "id_fix_list_fct": sizes.id_fix_list_fct,
            },
        }

    def ext_classes(self):
        from mpisppy.extensions.fixer import Fixer
        return [Fixer]

    def test_the_countdowns_survive_the_resume(self):
        from mpisppy.extensions.fixer import Fixer
        _, stopped, resumed = self.run_ab()
        # Compare what the checkpoint's models hold against what the resumed
        # run started from, per scenario and per nonant.
        for sname, s in stopped.local_scenarios.items():
            saved = dict(s._mpisppy_data.conv_iter_count)
            self.assertTrue(saved, msg="the fixer tracked nothing, so this "
                                       "test proves nothing")
        counts = {sname: dict(s._mpisppy_data.conv_iter_count)
                  for sname, s in resumed.local_scenarios.items()}
        self.assertTrue(any(counts[sname] for sname in counts))
        self.assertIsNotNone(_extension(resumed, Fixer))

    def test_the_same_variables_end_up_fixed(self):
        """The observable consequence: fixing happens at the same iteration.

        A resumed run whose countdowns restarted would still fix these
        variables eventually -- just later -- so comparing the final fixed set
        against the uninterrupted run's is what catches the delay.
        """
        reference, _, resumed = self.run_ab()
        want = _fixed_nonant_names(reference)
        got = _fixed_nonant_names(resumed)
        self.assertEqual(want, got)

    def test_the_running_totals_are_carried(self):
        from mpisppy.extensions.fixer import Fixer
        reference, _, resumed = self.run_ab()
        self.assertEqual(_extension(resumed, Fixer).fixed_so_far,
                         _extension(reference, Fixer).fixed_so_far)


def _fixed_nonant_names(ph):
    return {(sname, v.name)
            for sname, s in ph.local_scenarios.items()
            for v in s._mpisppy_data.nonant_indices.values()
            if v.is_fixed()}


def _relaxed_models(ph):
    """Which scenarios currently carry the integer relaxation.

    ``core.relax_integer_vars`` leaves ``_relaxed_integer_vars`` on the model
    and its undo deletes it, so the attribute is the model's own answer to
    "are my integers relaxed right now" -- independent of what any extension
    believes.
    """
    return {sname for sname, s in ph.local_scenarios.items()
            if hasattr(s, "_relaxed_integer_vars")}


class _RelaxationProbe(Extension):
    """Records which subproblems are relaxed, as the run goes.

    The end of a run does not answer the question this file is about: a
    resumed leg that re-relaxed at ``pre_iter0`` and then enforced again a
    couple of iterations later ends unrelaxed, exactly as an uninterrupted one
    does. What separates them is what the subproblems looked like *while the
    iterations were being solved*, so that is what this records.
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.after_iter0 = None
        self.per_iteration = []

    def post_iter0(self):
        self.after_iter0 = _relaxed_models(self.opt)

    def enditer(self):
        self.per_iteration.append(_relaxed_models(self.opt))

    # A test probe; nothing of its own to carry across a resume.
    def checkpoint_state(self):
        return None

    def restore_state(self, state):
        pass


class _IntegerRelaxMixin(_ABMixin):
    """Relax-then-enforce across a stop, on a MIP.

    The extension keeps no checkpointed state: the relaxation is a model
    transformation, so it rides in the dill, and ``pre_iter0`` reads which
    state the reloaded models are in from them. These classes pin that a
    resumed run neither relaxes again what the study had enforced nor
    forgets that it is still relaxed.

    The ratio decides which state the checkpoint is taken in, and both are
    worth pinning: ``0.1`` enforces before the stop, ``1.1`` puts both the
    iteration and the time condition out of reach so the run is still relaxed
    when it writes.
    """

    MODEL = sizes
    SCENARIOS = SIZES_SCENARIOS
    CREATOR_KWARGS = SIZES_KWARGS
    N = 4
    STOP = 2
    RATIO = None

    def setUp(self):
        super().setUp()
        self.EXTRA_OPTIONS = {
            "integer_relax_then_enforce_options": {"ratio": self.RATIO},
        }

    def ext_classes(self):
        from mpisppy.extensions.integer_relax_then_enforce import (
            IntegerRelaxThenEnforce)
        return [IntegerRelaxThenEnforce, _RelaxationProbe]

    def assert_objective_agrees(self, reference, _stopped, resumed):
        """The comparison a MIP supports, in place of bit-identity.

        These are the only classes in this file on a MIP rather than on
        farmer: alternate optima mean a resumed run can land on a different
        solution of equal value than the uninterrupted one found, which moves
        the iterate without anything having gone wrong -- and which solver is
        installed decides whether it does. The objective walking away is what
        would mean something went wrong, so that is what is pinned, at the
        tolerance `test_checkpoint_multirank.py` already uses for `sizes`.
        """
        want, got = reference.Eobjective(), resumed.Eobjective()
        scale = max(1.0, abs(want))
        self.assertLessEqual(
            abs(want - got), 1e-3 * scale,
            msg=f"the resumed run's expected objective is {got}, the "
                f"uninterrupted run's is {want}")

    def _flag(self, ph):
        from mpisppy.extensions.integer_relax_then_enforce import (
            IntegerRelaxThenEnforce)
        return _extension(ph, IntegerRelaxThenEnforce)._integers_relaxed


@unittest.skipIf(not solver_available, "no solver is available")
class TestIntegerRelaxThenEnforceEnforcedAtTheStop(_IntegerRelaxMixin,
                                                   unittest.TestCase):
    """The stop lands after integrality has been enforced.

    The ratio is a fraction of *this run's* iteration budget, so a leg told to
    run fewer iterations reaches its fraction sooner. At 0.1 both the
    uninterrupted leg and the stopped leg enforce at iteration 1, which is
    what lets the two be compared at all: a ratio where they enforced at
    different iterations would be measuring the window rather than the
    checkpoint.
    """

    RATIO = 0.1

    def test_the_stop_really_did_land_after_enforcement(self):
        """Otherwise this class is silently testing the other case."""
        _, stopped, _ = self.run_ab()
        self.assertEqual(_relaxed_models(stopped), set(),
                         msg="the stopped run was still relaxed, so the "
                             "checkpoint was not taken after enforcement")

    def test_a_resumed_run_does_not_re_relax_what_was_enforced(self):
        """Every iteration of the resumed leg solves what the study enforced.

        Checking the end of the run would not catch this: a leg that relaxed
        at pre_iter0 and enforced again two iterations later also ends
        unrelaxed.
        """
        reference, _, resumed = self.run_ab()
        probe = _extension(resumed, _RelaxationProbe)
        self.assertEqual(
            probe.after_iter0, set(),
            msg="the resumed run relaxed integrality that the study had "
                "already enforced")
        self.assertEqual(
            probe.per_iteration, [set()] * (self.N - self.STOP),
            msg=f"the resumed run solved relaxed subproblems: "
                f"{probe.per_iteration}")
        self.assertFalse(self._flag(resumed))
        # The uninterrupted leg is the standard: it, too, was enforced by the
        # time these iterations came around.
        self.assertEqual(
            _extension(reference, _RelaxationProbe).per_iteration[self.STOP:],
            [set()] * (self.N - self.STOP))

    def test_resume_matches_the_uninterrupted_run(self):
        self.assert_objective_agrees(*self.run_ab())


@unittest.skipIf(not solver_available, "no solver is available")
class TestIntegerRelaxThenEnforceRelaxedAtTheStop(_IntegerRelaxMixin,
                                                  unittest.TestCase):
    """The stop lands while the integers are still relaxed.

    The state a run using the profile's own ratio is overwhelmingly likely to
    be stopped in, since above 1 neither the iteration nor the time condition
    can fire inside a run.
    """

    RATIO = 1.1

    def test_the_stop_really_did_land_while_relaxed(self):
        _, stopped, _ = self.run_ab()
        self.assertEqual(_relaxed_models(stopped), set(stopped.local_scenarios))

    def test_a_resumed_run_comes_back_relaxed_and_knows_it(self):
        """The models keep the relaxation; the extension has to agree.

        An extension that came back believing the integers were enforced
        would return from `miditer` without ever undoing the relaxation, and
        the study would finish on a solution to the wrong problem.
        """
        _, _, resumed = self.run_ab()
        probe = _extension(resumed, _RelaxationProbe)
        self.assertEqual(probe.after_iter0, set(resumed.local_scenarios))
        self.assertEqual(probe.per_iteration,
                         [set(resumed.local_scenarios)] * (self.N - self.STOP))
        self.assertTrue(self._flag(resumed))

    def test_resume_matches_the_uninterrupted_run(self):
        self.assert_objective_agrees(*self.run_ab())


@unittest.skipIf(not solver_available, "no solver is available")
class TestIntegerRelaxThenEnforceOnTheStudysSchedule(_IntegerRelaxMixin,
                                                     unittest.TestCase):
    """With --stop-at-iteration-number, enforcement follows the study.

    Every leg is given the study bound, so the ratio is a fraction of the
    study rather than of the leg. At 0.5 over a 4-iteration study the
    uninterrupted run enforces at iteration 3. Legs that counted their own
    budgets would not, at either end of the stop: the stopped leg, 2
    iterations long, would enforce at iteration 2, one early; and a resumed
    leg with 2 iterations left would solve iteration 3 relaxed and enforce at
    4, one late.
    """

    RATIO = 0.5

    def setUp(self):
        super().setUp()
        self.EXTRA_OPTIONS = dict(self.EXTRA_OPTIONS,
                                  stop_at_iteration_number=self.N)

    def test_the_resumed_run_enforces_where_the_uninterrupted_one_did(self):
        reference, stopped, resumed = self.run_ab()
        want = _extension(reference, _RelaxationProbe).per_iteration
        relaxed = set(reference.local_scenarios)
        self.assertEqual(want, [relaxed, relaxed, set(), set()],
                         msg="the uninterrupted run did not enforce at "
                             "iteration 3, so this test proves nothing")
        self.assertEqual(_relaxed_models(stopped), relaxed,
                         msg="the stopped leg enforced before the study's "
                             "fraction, as one counting its own budget would")
        self.assertEqual(
            _extension(resumed, _RelaxationProbe).per_iteration,
            want[self.STOP:],
            msg="the resumed run enforced at a different iteration than "
                "the uninterrupted one")

    def test_resume_matches_the_uninterrupted_run(self):
        self.assert_objective_agrees(*self.run_ab())


def _read_leaf(ckpt_dir, generation):
    path = os.path.join(ckpt_dir, "hub", f"gen_{generation:04d}",
                        "hub_rank_0000.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


@unittest.skipIf(not solver_available, "no solver is available")
class TestSlammerResume(_ABMixin, unittest.TestCase):
    """What a slammer has already pinned, and to what value.

    Slams are sticky, so `_slammed` is both the record of what was done and
    the list of what would have to be released to undo it. A resumed run that
    forgot it reports the wrong totals and would slam a nonant a second time
    the moment anything else unfixed it.

    There is a second, subtler failure here that has nothing to do with
    `_slammed`: `pre_iter0` classifies every nonant that is fixed *right now*
    as the modeler's and drops it from the eligibility map. On a resumed run
    every mid-run fixing is already applied, so the extension would file its
    own earlier slams -- and the fixer's fixings -- as untouchable.
    """

    N = 5
    STOP = 3
    #: Farmer's nonants are continuous, so slam them to a bound.
    DIRECTIVE_PATTERN = "DevotedAcreage[*]"

    def setUp(self):
        super().setUp()
        from mpisppy.extensions.slammer import SlamDirective
        self.EXTRA_OPTIONS = {
            "slammer_options": {
                "directives": [SlamDirective(self.DIRECTIVE_PATTERN, True,
                                             ("lb",), 1.0)],
                "slam_start_iter": 1,
                "iters_between_slams": 1,
                "verbose": False,
            },
        }

    def ext_classes(self):
        from mpisppy.extensions.slammer import Slammer
        return [Slammer]

    def test_the_slam_record_is_restored(self):
        from mpisppy.extensions.slammer import Slammer
        _, stopped, resumed = self.run_ab()
        before = _extension(stopped, Slammer)._slammed
        self.assertTrue(before, msg="nothing was slammed before the stop, so "
                                    "this test proves nothing")
        after = _extension(resumed, Slammer)._slammed
        for ndn_i, value in before.items():
            self.assertIn(ndn_i, after,
                          msg="the resumed run forgot a nonant it had slammed")
            self.assertEqual(after[ndn_i], value)

    def test_previous_slams_are_not_filed_as_modeler_fixed(self):
        from mpisppy.extensions.slammer import Slammer
        _, stopped, resumed = self.run_ab()
        before = _extension(stopped, Slammer)._slammed
        ext = _extension(resumed, Slammer)
        for ndn_i in before:
            self.assertNotIn(
                ndn_i, ext._modeler_fixed,
                msg="a nonant this extension slammed itself came back "
                    "classified as fixed by the modeler")

    def test_resume_matches_the_uninterrupted_run(self):
        reference, _, resumed = self.run_ab()
        from mpisppy.extensions.slammer import Slammer
        self.assertEqual(
            set(_extension(resumed, Slammer)._slammed),
            set(_extension(reference, Slammer)._slammed),
            msg="the resumed run slammed a different set of nonants")
        self.assert_bit_identical(reference, resumed)


@unittest.skipIf(not solver_available, "no solver is available")
class TestPrimalDualConvergerResume(_ABMixin, unittest.TestCase):
    """A converger decides when the run *stops*, so the test is where.

    Its dual residual is rho * ||xbar_t - xbar_{t-1}||, and prev_xbars is its
    one piece of history. It needs no checkpoint entry: a resumed run builds
    it after the checkpointed models are spliced in, so it reads the xbars of
    the last checkpointed iteration, which is what prev_xbars held. So it is
    declared stateless, and what has to hold is that the resumed run stops
    at the iteration the uninterrupted one does, without warning.
    """

    #: Far enough that the reference converges on its own, before N.
    N = 60
    STOP = 5

    def ext_classes(self):
        return []

    def _ph(self, max_iters, **ckpt_kwargs):
        from mpisppy.convergers.primal_dual_converger import (
            PrimalDualConverger)
        options = _options(max_iters, **ckpt_kwargs)
        options["primal_dual_converger_options"] = {"tol": 1.0,
                                                    "verbose": False}
        return _make_ph(options, self.ext_classes(),
                        ph_converger=PrimalDualConverger)

    def test_the_resumed_run_stops_where_the_uninterrupted_one_does(self):
        """Stopped one iteration short of where the reference converges.

        Anywhere earlier, a wrong prev_xbars only changes the decision at the
        first resumed iteration, which is nowhere near converging on either
        leg, and the test passes regardless.
        """
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            reference = self._ph(self.N)
            reference.ph_main()
            self.assertLess(reference._PHIter, self.N,
                            msg="the reference never converged, so this "
                                "checks nothing about the converger")
            self.STOP = reference._PHIter - 1
            self.assertGreater(self.STOP, 1)
            _, _, resumed = self.run_ab()
        self.assertEqual(resumed._PHIter, reference._PHIter)
        self.assertNotIn("converger does not carry state", out.getvalue())

    def test_resume_is_bit_identical(self):
        reference, _, resumed = self.run_ab()
        self.assert_bit_identical(reference, resumed)


class _StatefulExtension(Extension):
    """Minimal extension with state, for the contract tests."""

    def __init__(self, opt):
        super().__init__(opt)
        self.counter = 0
        self.restored = None

    def checkpoint_state(self):
        return {"counter": self.counter}

    def restore_state(self, state):
        self.restored = state
        self.counter = state["counter"]


class _StatelessExtension(Extension):
    # Says so, rather than merely being so: an extension that overrides
    # neither hook is refused at the start of a checkpointed run.
    def checkpoint_state(self):
        return None

    def restore_state(self, state):
        pass


class _FakeOpt:
    def __init__(self, extobject=None, convobject=None):
        self.extobject = extobject
        self.convobject = convobject


class TestExtensionStateContract(unittest.TestCase):
    """The aggregation itself, without a solver in the way."""

    def test_the_base_extension_hooks_raise(self):
        """Not a no-op: having no state is a decision a subclass makes."""
        ext = Extension(None)
        with self.assertRaises(NotImplementedError):
            ext.checkpoint_state()
        with self.assertRaises(NotImplementedError):
            ext.restore_state(None)

    def test_the_base_converger_hooks_raise(self):
        from mpisppy.convergers.converger import Converger

        class _Bare(Converger):
            def is_converged(self):
                return False

        conv = _Bare(None)
        with self.assertRaises(NotImplementedError):
            conv.checkpoint_state()
        with self.assertRaises(NotImplementedError):
            conv.restore_state(None)

    def test_stateless_extensions_are_left_out_entirely(self):
        """The common case must add nothing to the file."""
        opt = _FakeOpt(extobject=_StatelessExtension(None))
        self.assertIsNone(checkpointing.gather_extension_state(opt))

    def test_state_is_keyed_by_class_name(self):
        ext = _StatefulExtension(None)
        ext.counter = 7
        state = checkpointing.gather_extension_state(_FakeOpt(extobject=ext))
        self.assertEqual(state["extensions"],
                         {"_StatefulExtension": {"counter": 7}})

    def test_multiextension_is_flattened_away(self):
        """The container has no state; the extensions inside it do."""
        multi = MultiExtension(None, [_StatefulExtension, _StatelessExtension])
        multi.extdict["_StatefulExtension"].counter = 3
        state = checkpointing.gather_extension_state(_FakeOpt(extobject=multi))
        self.assertEqual(list(state["extensions"]), ["_StatefulExtension"])

    def test_restore_dispatches_by_name(self):
        multi = MultiExtension(None, [_StatefulExtension])
        warnings = checkpointing.restore_extension_state(
            _FakeOpt(extobject=multi),
            {"extensions": {"_StatefulExtension": {"counter": 9}}})
        self.assertEqual(warnings, [])
        self.assertEqual(multi.extdict["_StatefulExtension"].counter, 9)

    def test_state_for_an_extension_that_is_gone_warns(self):
        """Resuming with a different extension set is allowed, not silent.

        The hub iterate in the checkpoint is still valid, so refusing the
        whole thing would be disproportionate -- but dropping trajectory state
        without saying so is exactly the silent divergence this phase exists
        to close.
        """
        multi = MultiExtension(None, [_StatelessExtension])
        warnings = checkpointing.restore_extension_state(
            _FakeOpt(extobject=multi),
            {"extensions": {"NormRhoUpdater": {"prev_avg": {}}}})
        self.assertEqual(len(warnings), 1)
        self.assertIn("NormRhoUpdater", warnings[0])

    def test_an_extension_added_since_the_checkpoint_does_not_warn(self):
        """It is starting fresh because it never ran, which is correct."""
        multi = MultiExtension(None, [_StatefulExtension])
        warnings = checkpointing.restore_extension_state(
            _FakeOpt(extobject=multi), {"extensions": {}})
        self.assertEqual(warnings, [])

    def test_a_different_converger_is_refused_by_name(self):
        from mpisppy.convergers.converger import Converger

        class _OtherConverger(Converger):
            def is_converged(self):
                return False

        opt = _FakeOpt(extobject=None, convobject=_OtherConverger(None))
        warnings = checkpointing.restore_extension_state(
            opt, {"extensions": {},
                  "converger": {"class": "PrimalDualConverger",
                                "state": {"prev_xbars": {}}}})
        self.assertEqual(len(warnings), 1)
        self.assertIn("PrimalDualConverger", warnings[0])


@unittest.skipIf(not solver_available, "no solver is available")
class TestFixerCountsAreNotZeroedOnResume(unittest.TestCase):
    """The `populate` regression, isolated from the A/B comparison.

    `Fixer.post_iter0` calls `populate`, which builds the per-nonant counts --
    and it runs on a resumed run too, where the counts it is about to build
    already came back in the dilled models. Zeroing them there discarded the
    checkpoint's most fixer-specific state while every other part of the
    resume worked perfectly.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _ph(self, max_iters, **ckpt_kwargs):
        from mpisppy.extensions.fixer import Fixer
        options = _options(max_iters, **ckpt_kwargs)
        options["fixeroptions"] = {
            "verbose": False,
            "boundtol": 0.01,
            "id_fix_list_fct": sizes.id_fix_list_fct,
        }
        return _make_ph(options, [Fixer], model=sizes,
                        scenario_names=SIZES_SCENARIOS,
                        creator_kwargs=SIZES_KWARGS)

    def test_counts_come_back_nonzero(self):
        stopped = self._ph(3, ckpt_dir=self.ckpt_dir)
        stopped.ph_main()
        saved = {sname: dict(s._mpisppy_data.conv_iter_count)
                 for sname, s in stopped.local_scenarios.items()}
        self.assertTrue(
            any(v for counts in saved.values() for v in counts.values()),
            msg="no nonant had a nonzero count at the stop, so this test "
                "cannot tell a preserved count from a zeroed one")

        resumed = self._ph(4, resume_from=self.ckpt_dir)
        # Stop after Iter0 so the comparison is against what the resume
        # restored, before any iteration has moved the counts on.
        resumed.PH_Prep()
        resumed.Iter0()
        for sname, s in resumed.local_scenarios.items():
            self.assertEqual(dict(s._mpisppy_data.conv_iter_count),
                             saved[sname],
                             msg=f"{sname}: the fixer's counts were reset by "
                                 f"its own setup hook on the way back in")


class TestTheStateContractIsRequiredAtStartup(unittest.TestCase):
    """A checkpointed run refuses, before any solve, a class that would fail
    later.

    The base hooks raise, so an extension that overrides neither would end
    the run at its first checkpoint write, and one that overrides only
    checkpoint_state would end it at the resume. Both are visible from the
    classes, so require_state_contract refuses them up front.
    """

    class _Implements(Extension):
        def checkpoint_state(self):
            return {"n": 1}

        def restore_state(self, state):
            pass

    class _Stateless(Extension):
        def checkpoint_state(self):
            return None

        def restore_state(self, state):
            pass

    class _AnswersNeither(Extension):
        pass

    class _SavesOnly(Extension):
        def checkpoint_state(self):
            return {"n": 1}

    class _RestoresOnly(Extension):
        def restore_state(self, state):
            pass

    class _SubclassOfImplements(_Implements):
        """Inherits a real implementation."""

    def _opt(self, *classes, converger=None):
        return types.SimpleNamespace(
            extobject=types.SimpleNamespace(
                extdict={c.__name__: c.__new__(c) for c in classes}),
            ph_converger=converger)

    def _refusal(self, *classes, converger=None):
        with self.assertRaises(RuntimeError) as cm:
            checkpointing.require_state_contract(
                self._opt(*classes, converger=converger))
        return str(cm.exception)

    def test_implementing_both_hooks_passes(self):
        checkpointing.require_state_contract(
            self._opt(self._Implements, self._Stateless))

    def test_the_implementation_is_inherited(self):
        checkpointing.require_state_contract(
            self._opt(self._SubclassOfImplements))

    def test_answering_neither_is_refused_by_name(self):
        message = self._refusal(self._Implements, self._AnswersNeither)
        self.assertIn("_AnswersNeither", message)
        self.assertNotIn("_Implements", message)

    def test_half_an_implementation_is_refused(self):
        """Saving state that can never be restored would fail at the resume,
        which is the worst moment to find out."""
        message = self._refusal(self._SavesOnly, self._RestoresOnly)
        self.assertIn("_SavesOnly (implements checkpoint_state only)",
                      message)
        self.assertIn("_RestoresOnly (implements restore_state only)",
                      message)

    def test_every_offender_is_named_at_once(self):
        message = self._refusal(self._AnswersNeither, self._SavesOnly)
        self.assertIn("_AnswersNeither", message)
        self.assertIn("_SavesOnly", message)

    def test_a_converger_that_answers_neither_is_refused(self):
        from mpisppy.convergers.converger import Converger

        class _BareConverger(Converger):
            def is_converged(self):
                return False

        message = self._refusal(self._Implements, converger=_BareConverger)
        self.assertIn("_BareConverger", message)

    def test_an_xhat_spoke_refuses_before_restoring(self):
        """A spoke writes its extensions' state with its incumbent, so the
        Checkpointer checks them in pre_iter0, before the restore and long
        before the spoke's first write."""
        restored = []
        fake = types.SimpleNamespace(
            dual_spoke_mode=False, spoke_mode=True,
            opt=self._opt(self._Implements, self._AnswersNeither),
            _restore_incumbent=lambda: restored.append(1))
        with self.assertRaises(RuntimeError) as cm:
            Checkpointer.pre_iter0(fake)
        self.assertIn("_AnswersNeither", str(cm.exception))
        self.assertEqual(restored, [])

    def test_an_empty_multiextension_is_a_container_not_an_offender(self):
        """A hub resume needs no Checkpointer, so a run can legitimately have
        a MultiExtension holding nothing."""
        opt = types.SimpleNamespace(extobject=MultiExtension(None, []),
                                    ph_converger=None)
        self.assertEqual(list(checkpointing._extension_objects(opt)), [])
        checkpointing.require_state_contract(opt)

    def test_a_shipped_converger_passes(self):
        from mpisppy.convergers.primal_dual_converger import \
            PrimalDualConverger
        checkpointing.require_state_contract(
            self._opt(self._Implements, converger=PrimalDualConverger))


@unittest.skipIf(not solver_available, "no solver is available")
class TestACheckpointedRunRefusesBeforeSolving(unittest.TestCase):
    """The refusal is reached through Iter0, before the first solve."""

    class _AnswersNeither(Extension):
        pass

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def test_refused_at_iter0(self):
        ph = _make_ph(_options(2, ckpt_dir=self.ckpt_dir),
                      [self._AnswersNeither])
        ph.PH_Prep()
        solves = []
        ph.solve_loop = lambda *a, **k: solves.append(1)
        with self.assertRaises(RuntimeError) as cm:
            ph.Iter0()
        self.assertIn("_AnswersNeither", str(cm.exception))
        self.assertEqual(solves, [])

    def test_refused_on_a_resume_alone(self):
        """--resume-from without --checkpoint-dir is checked too. The refusal
        comes before the splice, so it is reached even though the directory
        it names holds no checkpoint."""
        ph = _make_ph(_options(2, resume_from=self.ckpt_dir),
                      [self._AnswersNeither])
        ph.PH_Prep()
        with self.assertRaises(RuntimeError) as cm:
            ph.Iter0()
        self.assertIn("_AnswersNeither", str(cm.exception))

    def test_a_converger_is_refused_at_iter0(self):
        from mpisppy.convergers.converger import Converger

        class _BareConverger(Converger):
            def is_converged(self):
                return False

        ph = _make_ph(_options(2, ckpt_dir=self.ckpt_dir), [],
                      ph_converger=_BareConverger)
        ph.PH_Prep()
        with self.assertRaises(RuntimeError) as cm:
            ph.Iter0()
        self.assertIn("_BareConverger", str(cm.exception))

    def test_not_checked_without_checkpointing(self):
        """A run that never checkpoints never calls the hooks, so an
        extension that skipped them is no concern of its."""
        ph = _make_ph(_options(1), [self._AnswersNeither])
        ph.ph_main()


class TestXhatFeasibilityCutKeysContinue(unittest.TestCase):
    """The cuts ride in the models, and so does the key of the next one.

    A counter on the extension restarted at 0 on a resume, and even restored
    it came back only at the end of Iter0 -- after Iter0's spoke sync, which
    can already install a cut. Reading the next key from the model leaves
    nothing to restore and nothing to race.
    """

    def _extension_with_cuts(self, n_cuts):
        import pyomo.environ as pyo
        from mpisppy.extensions.xhat_feasibility_cut_extension import \
            XhatFeasibilityCutExtension
        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], domain=pyo.Binary)
        m._mpisppy_model = pyo.Block()
        m._mpisppy_model.xhat_feasibility_cuts = pyo.Constraint(pyo.Any)
        m._mpisppy_data = types.SimpleNamespace(
            nonant_indices={("ROOT", 0): m.x[0], ("ROOT", 1): m.x[1]})
        m._solver_plugin = None
        for key in range(1, n_cuts + 1):
            m._mpisppy_model.xhat_feasibility_cuts[key] = (
                0.0, float(key) - m.x[0], None)
        ext = XhatFeasibilityCutExtension.__new__(XhatFeasibilityCutExtension)
        ext.opt = types.SimpleNamespace(local_scenarios={"s": m})
        ext._row_len = 3
        return ext, m

    def test_it_has_nothing_to_carry(self):
        ext, _ = self._extension_with_cuts(3)
        self.assertIsNone(ext.checkpoint_state())

    def test_a_cut_on_restored_models_does_not_overwrite_one_before_it(self):
        """A fresh extension object over models that already hold three cuts,
        as on a resume, with no restore_state having run."""
        ext, m = self._extension_with_cuts(3)
        cuts = m._mpisppy_model.xhat_feasibility_cuts
        before = {k: str(cuts[k].body) for k in cuts}
        ext._install_cuts([5.0, 1.0, -1.0, 1])
        self.assertEqual(sorted(cuts.keys()), [1, 2, 3, 4])
        self.assertEqual({k: str(cuts[k].body) for k in before}, before)

    def test_restored_models_without_the_component_get_one(self):
        """A resume that attaches the extension to a checkpoint written
        without it: setup_hub's component was on the replaced models."""
        ext, m = self._extension_with_cuts(0)
        m._mpisppy_model.del_component("xhat_feasibility_cuts")
        ext._install_cuts([5.0, 1.0, -1.0, 1])
        self.assertEqual(
            sorted(m._mpisppy_model.xhat_feasibility_cuts.keys()), [1])

    def test_a_zero_row_does_not_use_a_key(self):
        ext, m = self._extension_with_cuts(2)
        ext._install_cuts([0.0, 0.0, 0.0, 6.0, -1.0, 1.0, 2])
        self.assertEqual(
            sorted(m._mpisppy_model.xhat_feasibility_cuts.keys()), [1, 2, 3])

    def test_keys_start_at_one_on_a_fresh_run(self):
        ext, m = self._extension_with_cuts(0)
        ext._install_cuts([5.0, 1.0, -1.0, 6.0, -1.0, 1.0, 2])
        self.assertEqual(
            sorted(m._mpisppy_model.xhat_feasibility_cuts.keys()), [1, 2])


class TestRelaxedPHFixerResume(unittest.TestCase):
    """Its decisions need nothing saved; its one-time pass must not rerun."""

    def _fixer(self, resumed):
        from mpisppy.extensions.relaxed_ph_fixer import RelaxedPHFixer
        fixer = RelaxedPHFixer.__new__(RelaxedPHFixer)
        fixer.opt = types.SimpleNamespace(_resumed_from_checkpoint=resumed,
                                          spcomm=None)
        fixer.relaxed_nonant_buf = types.SimpleNamespace(
            id=lambda: 1, value_array=lambda: [0.0])
        fixer.passes = []
        fixer.relaxed_ph_fixing = (
            lambda sol, pre_iter0=False: fixer.passes.append(pre_iter0))
        fixer._heuristic_fixed_vars = {"s0": 0, "s1": 0}
        return fixer

    def test_a_fresh_run_makes_the_pre_iter0_pass(self):
        fixer = self._fixer(resumed=False)
        fixer.iter0_post_solver_creation()
        self.assertEqual(fixer.passes, [True])

    def test_a_resumed_run_does_not(self):
        fixer = self._fixer(resumed=True)
        fixer.iter0_post_solver_creation()
        self.assertEqual(fixer.passes, [])

    def test_the_count_round_trips(self):
        stopped = self._fixer(resumed=False)
        stopped._heuristic_fixed_vars = {"s0": 4, "s1": 2}
        state = pickle.loads(pickle.dumps(stopped.checkpoint_state()))
        resumed = self._fixer(resumed=True)
        resumed.restore_state(state)
        self.assertEqual(resumed._heuristic_fixed_vars, {"s0": 4, "s1": 2})


class TestReducedCostsFixerResume(unittest.TestCase):
    """The reduced costs it fixes from, and the bound that gates new ones."""

    def _fixer(self, resumed):
        from mpisppy.extensions.reduced_costs_fixer import ReducedCostsFixer
        fixer = ReducedCostsFixer.__new__(ReducedCostsFixer)
        fixer.opt = types.SimpleNamespace(_resumed_from_checkpoint=resumed)
        fixer._fix_fraction_target_pre_iter0 = 0.5
        fixer._fix_fraction_target_iter0 = 0.25
        fixer._best_outer_bound = -float("inf")
        fixer._outer_bound_update = lambda new, old: new > old
        fixer._current_reduced_costs = None
        fixer._heuristic_fixed_vars = 0
        return fixer

    def test_a_resumed_run_skips_the_pre_iter0_pass(self):
        fixer = self._fixer(resumed=True)
        # Reaching the spoke wait would fail here: there is no buffer.
        fixer.iter0_post_solver_creation()
        self.assertEqual(fixer.fix_fraction_target, 0.0)

    def test_iter0s_sync_on_a_resume_fixes_nothing(self):
        """The reduced-costs spoke restarts on a resume, and Iter0's sync runs
        before restore_state, while the best bound is still -inf. Reduced
        costs it sends then are recorded, not fixed from; the restore keeps
        the checkpoint's when its bound is better, and fixing resumes at the
        next iteration from those."""
        import numpy as np
        fixer = self._fixer(resumed=True)
        fixer._fix_fraction_target_iterK = 0.75
        fixer._rc_fixer_require_improving_lagrangian = True
        fixer.verbose = False
        fixer.opt.cylinder_rank = 0
        fixer.opt.spcomm = types.SimpleNamespace(
            get_receive_buffer=lambda *a, **k: True)
        fixer.reduced_costs_spoke_index = 0
        fixer.reduced_cost_buf = types.SimpleNamespace(
            is_new=lambda: True, id=lambda: 1,
            value_array=lambda: np.array([7.0, 7.0]))
        fixer.outer_bound_buf = types.SimpleNamespace(
            id=lambda: 1, value_array=lambda: np.array([-1000.0]))
        fixings = []
        fixer.reduced_costs_fixing = lambda rc, **k: fixings.append(rc)

        fixer.iter0_post_solver_creation()
        fixer.sync_with_spokes()
        self.assertEqual(fixings, [],
                         msg="Iter0's sync fixed from a restarted spoke")
        fixer.post_iter0_after_sync()
        fixer.restore_state({"best_outer_bound": 12.5,
                             "current_reduced_costs": [1.0, 2.0],
                             "heuristic_fixed_vars": 3.0})
        self.assertEqual(fixer._best_outer_bound, 12.5)
        self.assertEqual(fixer._current_reduced_costs.tolist(), [1.0, 2.0])
        self.assertEqual(fixer.fix_fraction_target, 0.75)

    def test_the_state_round_trips(self):
        import numpy as np
        stopped = self._fixer(resumed=False)
        stopped._best_outer_bound = 12.5
        stopped._current_reduced_costs = np.array([0.0, 3.0, -1.0])
        stopped._heuristic_fixed_vars = 7.0
        state = pickle.loads(pickle.dumps(stopped.checkpoint_state()))
        resumed = self._fixer(resumed=True)
        resumed.restore_state(state)
        self.assertEqual(resumed._best_outer_bound, 12.5)
        self.assertEqual(resumed._current_reduced_costs.tolist(),
                         [0.0, 3.0, -1.0])
        self.assertEqual(resumed._heuristic_fixed_vars, 7.0)

    def test_newer_reduced_costs_from_iter0_are_kept(self):
        """Iter0's spoke sync runs before the restore; reduced costs it took
        at a better bound than the checkpoint's are newer than the saved ones."""
        import numpy as np
        resumed = self._fixer(resumed=True)
        resumed._best_outer_bound = 20.0
        resumed._current_reduced_costs = np.array([9.0])
        resumed.restore_state({"best_outer_bound": 12.5,
                               "current_reduced_costs": [1.0],
                               "heuristic_fixed_vars": 3.0})
        self.assertEqual(resumed._best_outer_bound, 20.0)
        self.assertEqual(resumed._current_reduced_costs.tolist(), [9.0])
        self.assertEqual(resumed._heuristic_fixed_vars, 3.0)


class TestPHTrackerFilesContinue(unittest.TestCase):
    """A resumed run continues the tracker's files instead of truncating them."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _tracked(self):
        from mpisppy.extensions.phtracker import TrackedData
        tracked = TrackedData("xbars", self._tmp.name)
        tracked.initialize_fnames()
        return tracked

    def _rows(self, tracked):
        import pandas as pd
        return pd.read_csv(tracked.fname)["iteration"].tolist()

    def test_rows_after_the_checkpoint_are_dropped(self):
        tracked = self._tracked()
        tracked.initialize_df(["iteration", "x"])
        for it in range(6):
            tracked.add_row([it, float(it)])
        tracked.write_out_data()
        # The resumed run, continuing from iteration 3.
        resumed = self._tracked()
        resumed.initialize_df(["iteration", "x"], keep_through=3)
        self.assertEqual(self._rows(resumed), [0, 1, 2, 3])
        self.assertEqual(resumed.seen_iters, {0, 1, 2, 3})
        resumed.add_row([4, 4.0])
        resumed.write_out_data()
        self.assertEqual(self._rows(resumed), [0, 1, 2, 3, 4])

    def test_a_fresh_run_still_starts_over(self):
        tracked = self._tracked()
        tracked.initialize_df(["iteration", "x"])
        tracked.add_row([0, 0.0])
        tracked.write_out_data()
        again = self._tracked()
        again.initialize_df(["iteration", "x"])
        self.assertEqual(self._rows(again), [])

    def test_a_checkpoint_writes_out_what_is_buffered(self):
        from mpisppy.extensions.phtracker import PHTracker
        tracker = PHTracker.__new__(PHTracker)
        tracker.finished_init = True
        tracker._rank = 0
        tracker.spcomm = None
        tracker.opt = types.SimpleNamespace(_PHIter=2)
        tracked = self._tracked()
        tracked.initialize_df(["iteration", "x"])
        tracked.add_row([1, 1.0])
        tracked.add_row([2, 2.0])
        tracker.track_dict = {"xbars": tracked}
        self.assertEqual(tracker.checkpoint_state(), {"through_iteration": 2})
        self.assertEqual(self._rows(tracked), [1, 2])


@unittest.skipIf(not solver_available, "no solver is available")
class TestPHTrackerResume(_ABMixin, unittest.TestCase):
    """End to end: the stopped-and-resumed run's tracker file matches the
    uninterrupted run's, rather than holding only the resumed iterations."""

    N = 5
    STOP = 2

    def ext_classes(self):
        from mpisppy.extensions.phtracker import PHTracker
        return [PHTracker]

    def _ph(self, max_iters, **ckpt_kwargs):
        folder = os.path.join(
            self._tmp.name, "ab" if ckpt_kwargs else "reference")
        self.EXTRA_OPTIONS = {"phtracker_options": {
            "results_folder": folder, "track_xbars": True, "write_every": 3}}
        return super()._ph(max_iters, **ckpt_kwargs)

    def _rows(self, which):
        import pandas as pd
        folder = os.path.join(self._tmp.name, which)
        (cylinder,) = os.listdir(folder)
        return pd.read_csv(os.path.join(folder, cylinder, "xbars.csv"))

    def test_a_resume_that_runs_no_iterations_finishes(self):
        """A resumed run skips Iter0's solve loop, so with no iterations left
        the tracker was never set up when post_everything ran."""
        stopped = self._ph(self.STOP, ckpt_dir=self.ckpt_dir)
        stopped.ph_main()
        resumed = self._ph(0, resume_from=self.ckpt_dir)
        resumed.ph_main()
        self.assertEqual(self._rows("ab")["iteration"].tolist(),
                         list(range(self.STOP + 1)))

    def test_the_file_matches_the_uninterrupted_run(self):
        self.run_ab()
        want, got = self._rows("reference"), self._rows("ab")
        self.assertEqual(got["iteration"].tolist(),
                         want["iteration"].tolist())
        self.assertTrue(want.equals(got),
                        msg="the resumed run's tracked xbars differ")


class TestShippedExtensionsAnswerTheQuestion(unittest.TestCase):
    """Every shipped extension either carries its state or says it has none.

    The point of the base hooks raising is that adding an extension makes
    somebody decide. This pins the ones already decided, so a new one cannot
    join the unanswered set unnoticed -- and names the ones still unanswered,
    each of which is a known open question rather than an oversight.
    """

    #: Extensions that keep state across iterations and do not yet carry it.
    #: A checkpointed run refuses each of them at startup, since a resume
    #: with one would not retrace an uninterrupted run.
    UNANSWERED = {"CrossScenarioExtension", "WOscillationMonitor"}

    #: Where shipped extensions and convergers live. Not only
    #: mpisppy.extensions: the W and xbar file extensions are in utils, and
    #: were unanswered while this looked only in extensions.
    PACKAGES = ("mpisppy.extensions", "mpisppy.utils", "mpisppy.convergers")

    def _all_classes(self, base, excluded):
        import importlib
        import inspect
        import pkgutil
        found = {}
        for package_name in self.PACKAGES:
            package = importlib.import_module(package_name)
            for mod_info in pkgutil.walk_packages(package.__path__,
                                                  f"{package_name}."):
                try:
                    mod = importlib.import_module(mod_info.name)
                except ImportError:
                    continue    # an optional dependency this env lacks
                except Exception:
                    # utils holds a few scripts that refuse to be imported
                    # without their command line; none defines an extension.
                    # The extensions package itself has to import.
                    if package_name == "mpisppy.extensions":
                        raise
                    continue
                for name, cls in inspect.getmembers(mod, inspect.isclass):
                    if (issubclass(cls, base) and cls not in excluded
                            and cls.__module__ == mod.__name__):
                        found[name] = cls
        return found

    def _all_extension_classes(self):
        return self._all_classes(Extension, (Extension, MultiExtension))

    def test_every_shipped_converger_answers(self):
        from mpisppy.convergers.converger import Converger
        found = self._all_classes(Converger, (Converger,))
        self.assertIn("PrimalDualConverger", found)
        unanswered = {
            name for name, cls in found.items()
            if checkpointing._overrides_state_hooks(cls, Converger)
            != (True, True)}
        self.assertEqual(unanswered, set(),
                         msg="a converger does not implement both state "
                             "hooks, so every checkpointed run with it is "
                             "refused")

    def test_the_unanswered_set_is_exactly_what_is_recorded(self):
        classes = self._all_extension_classes()
        half = {
            name for name, cls in classes.items()
            if len(set(checkpointing._overrides_state_hooks(cls, Extension)))
            == 2}
        self.assertEqual(half, set(),
                         msg="a shipped extension implements one state hook "
                             "without the other")
        unanswered = {
            name for name, cls in classes.items()
            if checkpointing._overrides_state_hooks(cls, Extension)
            == (False, False)}
        self.assertEqual(
            unanswered, self.UNANSWERED,
            msg="an extension joined or left the set that implements neither "
                "state hook. If you added one, implement checkpoint_state and "
                "restore_state (returning None and doing nothing if it has no "
                "state); if you answered one, drop it from UNANSWERED here.")


if __name__ == "__main__":
    unittest.main()
