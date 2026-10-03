###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""The xhatshuffle spoke's loop cursor across a checkpoint.

A spoke's incumbent file gives it back its best solution; the cursor gives it
back its *place*. An xhatshuffle spoke walks a shuffled list of scenarios,
trying each as a candidate xhat. A resume always begins a new epoch, so
nothing is tried twice either way; what the cursor carries is where that
epoch starts (the best scenario so far), the position in the order, and the
direction, so the resumed spoke walks on the way the uninterrupted one would
rather than from the start of the order.

Two things make this cheaper than it sounds, and both are worth stating
because they are what the design leans on:

* **The scenario order is not carried.** The shuffle is seeded to a fixed
  value and drawn once, so a resumed spoke reproduces it exactly. Only the
  *position* in it is checkpointed -- and a position is only meaningful
  against the order it indexes, so the file carries a fingerprint of that
  order and the cursor is discarded if it no longer matches.
* **A cursor move costs a subproblem solve.** So writing the small spoke file
  whenever the cursor moves is negligible against what caused the move, while
  a pass that solves nothing writes nothing.

The dual cylinders' own W also crosses the checkpoint, and the serial half
of that lives here too: what a restored set of weights has to satisfy
before this cylinder publishes it. The half that needs several ranks -- every
rank restoring the same iteration -- is in ``test_checkpoint_multirank.py``.
"""

import os
import tempfile
import types
import unittest

import mpisppy.utils.checkpointing as checkpointing
import mpisppy.tests.examples.farmer as farmer
from mpisppy.cylinders.xhatshufflelooper_bounder import ScenarioCycler
from mpisppy.opt.ph import PH
from mpisppy.tests.utils import get_solver

solver_available, solver_name, persistent_available, persistent_solver_name = \
    get_solver()


def _cycler(names, iter_step=1):
    """A two-stage cycler over ``names``, in that order.

    ``nonleaves={}`` is what makes it two-stage: the constructor looks for a
    'ROOT' entry with children and finds none, which is the same branch a real
    two-stage problem takes.
    """
    shuffled = list(enumerate(names))
    return ScenarioCycler(shuffled, {}, False, iter_step)


class TestScenarioCyclerState(unittest.TestCase):
    """The cursor round trip, without a solver or an MPI job in the way."""

    NAMES = ["scen0", "scen1", "scen2", "scen3", "scen4"]

    def test_a_fresh_cycler_starts_at_the_beginning(self):
        cycler = _cycler(self.NAMES)
        state = cycler.checkpoint_state()
        self.assertEqual(state["cycle_idx"], 0)
        self.assertEqual(state["cur_root_scen"], "scen0")
        self.assertEqual(state["scenarios_this_epoch"], [])

    def test_the_position_survives_a_round_trip(self):
        cycler = _cycler(self.NAMES)
        for _ in range(3):
            cycler.get_next()
        saved = cycler.checkpoint_state()

        restored = _cycler(self.NAMES)
        self.assertEqual(restored.restore_state(saved), [])
        self.assertEqual(restored.checkpoint_state(), saved)

    def test_a_restored_cycler_continues_where_the_other_left_off(self):
        """The property that matters, in the order the loop makes the calls.

        The xhatshuffle loop begins a new epoch whenever the hub sends new
        nonants, and the first pass after a resume always has new ones, so a
        restore is always followed by ``begin_epoch`` before ``get_next``.
        That resets the tried set and the current scenario; what carries over
        is ``best`` (where the epoch starts), the cycle position and the
        direction. So the comparison is of what comes after ``begin_epoch``
        -- and against a fresh cycler too, which would pass a comparison that
        a restore doing nothing also passes.
        """
        def walk(cycler):
            cycler.begin_epoch()
            return [cycler.get_next() for _ in range(len(self.NAMES))]

        uninterrupted = _cycler(self.NAMES)
        for _ in range(2):
            uninterrupted.get_next()
        uninterrupted.best = "scen3"
        saved = uninterrupted.checkpoint_state()

        resumed = _cycler(self.NAMES)
        self.assertEqual(resumed.restore_state(saved), [])
        fresh = _cycler(self.NAMES)
        fresh.best = "scen3"

        want = walk(uninterrupted)
        self.assertEqual(walk(resumed), want)
        self.assertNotEqual(walk(fresh), want,
                            msg="a fresh cycler walks the same way, so this "
                                "cannot tell a restore from none")

    def test_the_best_scenario_survives(self):
        """`best` decides where the next epoch starts, so it is trajectory."""
        cycler = _cycler(self.NAMES)
        cycler.get_next()
        cycler.best = "scen3"
        restored = _cycler(self.NAMES)
        restored.restore_state(cycler.checkpoint_state())
        self.assertEqual(restored.best, "scen3")
        restored.begin_epoch()
        self.assertEqual(restored.get_next()["ROOT"], "scen3")

    def test_a_changed_scenario_order_is_refused_and_reported(self):
        """A position means nothing against a different list.

        The cursor is an index. If the model's scenario list changed, index 3
        is a different scenario, and restoring it would send the spoke
        somewhere it never meant to go -- silently, since an index is always a
        valid index. So the file carries a fingerprint of the order it was
        taken against.
        """
        cycler = _cycler(self.NAMES)
        for _ in range(3):
            cycler.get_next()
        saved = cycler.checkpoint_state()

        other = _cycler(self.NAMES + ["scen5"])
        warnings = other.restore_state(saved)
        self.assertEqual(len(warnings), 1)
        self.assertIn("different scenario order", warnings[0])

    def test_a_refused_cursor_leaves_a_usable_cycler(self):
        """Refusing the cursor must not break the spoke.

        The same file carries the incumbent, which is the part worth keeping.
        A cursor that no longer fits is a reason to explore from the start
        again, not a reason to fail the resume.
        """
        cycler = _cycler(self.NAMES)
        for _ in range(3):
            cycler.get_next()

        other = _cycler(self.NAMES + ["scen5"])
        other.restore_state(cycler.checkpoint_state())
        self.assertEqual(other.get_next()["ROOT"], "scen0")

    def _saved_away_from_a_fresh_cycler(self):
        """A cursor that differs from a fresh cycler's in every field a
        two-stage cycler moves: position, best, current scenario, and the
        scenarios tried.

        So a refusal that wrote one of them before deciding shows up as a
        changed cycler; a cursor that matched a fresh cycler on best, say,
        could not tell a half-applied restore from an untouched one.
        """
        cycler = _cycler(self.NAMES)
        for _ in range(3):
            cycler.get_next()
        cycler.best = "scen2"
        return cycler.checkpoint_state()

    def assert_refused_and_untouched(self, cycler, before, warnings, name):
        self.assertEqual(len(warnings), 1)
        self.assertIn(name, warnings[0])
        self.assertEqual(cycler.checkpoint_state(), before,
                         msg="a refused cursor was half applied")
        self.assertEqual(cycler.get_next()["ROOT"], "scen0")

    def test_a_cursor_missing_a_key_is_refused_and_changes_nothing(self):
        """A same-version file whose cursor lacks a key is discarded whole.

        Hand-edited, or written before a change to checkpoint_state() that
        did not raise the format version. It goes the way of a changed
        scenario order, with a warning, rather than raising a KeyError out of
        the spoke's main() and taking the incumbent down with it.
        """
        saved = self._saved_away_from_a_fresh_cycler()
        for key in saved:
            with self.subTest(missing=key):
                damaged = {k: v for k, v in saved.items() if k != key}
                cycler = _cycler(self.NAMES)
                before = cycler.checkpoint_state()
                warnings = cycler.restore_state(damaged)
                self.assert_refused_and_untouched(
                    cycler, before, warnings,
                    "different scenario order" if key == "order_fingerprint"
                    else key)

    def test_a_cursor_with_a_bad_value_is_refused_and_changes_nothing(self):
        saved = self._saved_away_from_a_fresh_cycler()
        for key, value in (("cycle_idx", len(self.NAMES)),
                           ("cycle_idx", "2"), ("cycle_idx", True),
                           ("best", 3), ("cur_root_scen", 3),
                           ("scenarios_this_epoch", None),
                           ("scenarios_this_epoch", [1]),
                           ("reversed", "no"), ("nodescen_dict", None),
                           ("nodescen_dict", {"ROOT": 3})):
            with self.subTest(key=key, value=value):
                cycler = _cycler(self.NAMES)
                before = cycler.checkpoint_state()
                warnings = cycler.restore_state(dict(saved, **{key: value}))
                self.assert_refused_and_untouched(cycler, before, warnings,
                                                  key)

    def test_loop_state_missing_a_key_or_bad_is_refused_with_a_warning(self):
        from mpisppy.cylinders.xhatshufflelooper_bounder import (
            XhatShuffleInnerBound)
        saved = {"xh_iter": 4, "cursor": self._saved_away_from_a_fresh_cycler()}
        cases = [(f"no {key}", key, {k: v for k, v in saved.items()
                                     if k != key}) for key in saved]
        cases += [(f"xh_iter={value!r}", "xh_iter", dict(saved, xh_iter=value))
                  for value in (None, "4", -1, True)]
        cases.append(("not a dict", "cursor", ["not", "a", "dict"]))
        for label, name, state in cases:
            with self.subTest(case=label):
                spoke = types.SimpleNamespace(
                    scenario_cycler=_cycler(self.NAMES), xh_iter=1)
                before = spoke.scenario_cycler.checkpoint_state()
                warnings = XhatShuffleInnerBound.restore_loop_state(spoke,
                                                                    state)
                self.assertEqual(len(warnings), 1)
                self.assertIn(name, warnings[0])
                self.assertEqual(spoke.xh_iter, 1)
                self.assertEqual(spoke.scenario_cycler.checkpoint_state(),
                                 before,
                                 msg="the cursor was restored although the "
                                     "loop state was refused")

    def test_progress_of_a_loop_state_that_is_not_a_dict_is_none(self):
        """Read on a resume in place before the restore runs, so it must not
        raise on what the restore would refuse."""
        from mpisppy.cylinders.xhatshufflelooper_bounder import (
            XhatShuffleInnerBound)
        for state in (None, ["cursor"], "cursor"):
            with self.subTest(state=state):
                self.assertIsNone(
                    XhatShuffleInnerBound.loop_state_progress(None, state))

    def test_the_fingerprint_is_order_sensitive(self):
        """The same names shuffled differently must not compare equal."""
        forward = _cycler(self.NAMES)
        backward = ScenarioCycler(
            list(enumerate(self.NAMES))[::-1], {}, False, 1)
        self.assertNotEqual(
            forward.checkpoint_state()["order_fingerprint"],
            backward.checkpoint_state()["order_fingerprint"])


def _cycler_after(names, advances):
    """The state of a fresh cycler advanced ``advances`` times."""
    cycler = _cycler(names)
    for _ in range(advances):
        cycler.get_next()
    return cycler.checkpoint_state()


class _CursorSpokeStub:
    """A spoke whose cursor the test moves by hand."""

    def __init__(self, loop_state=None):
        self.strata_rank = 2
        self.best_inner_bound = None
        self.loop_state = loop_state
        self.sent_bounds = []
        self.sent_xhats = 0

    def send_bound(self, value):
        self.sent_bounds.append(value)

    def send_best_xhat(self):
        self.sent_xhats += 1

    def checkpoint_loop_state(self):
        return self.loop_state

    def loop_state_progress(self, state):
        # Delegated to the real spoke rather than reimplemented: a stub that
        # answers this itself cannot notice when the real projection changes,
        # which is how an idle pass came to rewrite the file on every pass.
        from mpisppy.cylinders.xhatshufflelooper_bounder import (
            XhatShuffleInnerBound)
        return XhatShuffleInnerBound.loop_state_progress(self, state)


@unittest.skipIf(not solver_available, "no solver is available")
class TestSpokeWritesWhenTheCursorMoves(unittest.TestCase):
    """The write gate: an unchanged incumbent is no longer enough to skip.

    Before this phase the spoke wrote only when its incumbent improved, which
    is rare. The cursor moves far more often -- but only ever as the result of
    a subproblem solve, so the write is cheap against what caused it. What
    still has to cost nothing is a pass that solves nothing, and that is the
    case pinned here.
    """

    def setUp(self):
        from mpisppy.tests.test_checkpoint import _xhat_eval, _set_and_cache_solution
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        self.opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
        self.spoke = _CursorSpokeStub()
        self.opt.spcomm = self.spoke
        self._set_and_cache_solution = _set_and_cache_solution

    def tearDown(self):
        self._tmp.cleanup()

    def _checkpointer(self):
        from mpisppy.extensions.checkpointer import Checkpointer
        return Checkpointer(self.opt)

    def _spoke_file(self):
        return os.path.join(
            self.ckpt_dir, "spokes",
            "spoke__CursorSpokeStub_ordinal_00_rank_0000.pkl")

    def test_a_pass_that_changes_nothing_writes_nothing(self):
        ext = self._checkpointer()
        self._set_and_cache_solution(self.opt, 1.0)
        self.opt.best_solution_obj_val = 10.0
        ext.maybe_checkpoint()
        first = os.path.getmtime(self._spoke_file())

        # Same incumbent, same cursor: the spinning-loop case.
        ext.maybe_checkpoint()
        self.assertEqual(os.path.getmtime(self._spoke_file()), first)

    def test_an_idle_pass_does_not_write_when_only_xh_iter_ticks(self):
        """The loop increments xh_iter at the bottom of every pass, including
        the ones that only poll the hub and solve nothing, because it feeds
        the periodic debug line about where the delay comes from. Comparing
        the whole loop state therefore made every such pass pickle the
        incumbent, rename the file and fsync the directory -- measured at
        about 8,700 writes a second on farmer over tmpfs, and silent, because
        the write only announces itself when the objective improved."""
        ext = self._checkpointer()
        self._set_and_cache_solution(self.opt, 1.0)
        self.opt.best_solution_obj_val = 10.0
        self.spoke.loop_state = {"xh_iter": 1, "cursor": {"cycle_idx": 3}}
        ext.maybe_checkpoint()
        first = os.path.getmtime(self._spoke_file())

        for n in range(2, 8):
            self.spoke.loop_state = {"xh_iter": n, "cursor": {"cycle_idx": 3}}
            ext.maybe_checkpoint()

        self.assertEqual(
            os.path.getmtime(self._spoke_file()), first,
            msg="a pass that solved nothing rewrote the incumbent file")

    def test_a_cursor_move_alone_triggers_a_write(self):
        import pickle
        ext = self._checkpointer()
        self._set_and_cache_solution(self.opt, 1.0)
        self.opt.best_solution_obj_val = 10.0
        ext.maybe_checkpoint()

        # The incumbent is unchanged; only the cursor moved.
        self.spoke.loop_state = {"xh_iter": 7, "cursor": {"cycle_idx": 3}}
        ext.maybe_checkpoint()
        with open(self._spoke_file(), "rb") as f:
            written = pickle.load(f)
        self.assertEqual(written["loop_state"],
                         {"xh_iter": 7, "cursor": {"cycle_idx": 3}})

    def test_a_spoke_with_no_cursor_carries_none(self):
        """Every xhatter but xhatshuffle, which is most of them."""
        import pickle
        ext = self._checkpointer()
        self._set_and_cache_solution(self.opt, 1.0)
        self.opt.best_solution_obj_val = 10.0
        ext.maybe_checkpoint()
        with open(self._spoke_file(), "rb") as f:
            self.assertIsNone(pickle.load(f)["loop_state"])


class TestRestoredDualsMustMatchTheirFile(unittest.TestCase):
    """A dual cylinder's restored W has to reproduce the E[W] its file recorded.

    Every other check on that file asks whether it describes this model --
    its fingerprint, this rank's scenario names, a weight for every nonant.
    None of them looks at the numbers, and the numbers are what another
    cylinder turns into a Lagrangian bound, which the hub keeps as
    best-so-far. So the file records E[W] as the writing run computed it, and
    the restore recomputes it from the restored models.

    It used to require E[W] = 0 instead, which PH keeps only when rho is the
    same in every scenario: with a scenario-dependent rho the run that wrote
    the file finished normally and its resume was refused, blaming the file.

    No solver and no mpiexec: PH_Prep attaches W, and setting it by hand is
    exactly the state a restore leaves behind.
    """

    CYLINDER = "RelaxedPHSpoke"

    def _prepped_ph(self):
        """A PH whose models carry W, with no solve having happened."""
        opt = PH(
            {"solver_name": solver_name or "unused", "PHIterLimit": 1,
             "defaultPHrho": 1.0, "convthresh": 1e-4, "verbose": False,
             "display_progress": False, "display_timing": False,
             "iter0_solver_options": None, "iterk_solver_options": None,
             "tee-rank0-solves": False, "smoothed": 0},
            ["scen0", "scen1", "scen2"],
            farmer.scenario_creator, farmer.scenario_denouement,
            scenario_creator_kwargs={"use_integer": False,
                                     "crops_multiplier": 1},
        )
        opt.PH_Prep(attach_prox=False)
        return opt

    def _set_W(self, opt, per_scenario):
        for sname, s in opt.local_scenarios.items():
            for ndn_i in s._mpisppy_data.nonant_indices:
                s._mpisppy_model.W[ndn_i]._value = per_scenario[sname]

    def _recorded(self, opt, per_scenario):
        """What the writing run would have recorded for these weights."""
        from mpisppy.phbase import Wbar_by_node
        self._set_W(opt, per_scenario)
        return Wbar_by_node(opt)

    def _check(self, opt, recorded, generation=7):
        from mpisppy.phbase import Wbar_by_node, W_magnitude_by_node
        checkpointing.require_restored_duals_match_their_file(
            opt, self.CYLINDER, generation, recorded,
            Wbar_by_node(opt), W_magnitude_by_node(opt))

    def test_the_weights_the_file_recorded_are_accepted(self):
        opt = self._prepped_ph()
        recorded = self._recorded(opt, {"scen0": 10.0, "scen1": -4.0,
                                        "scen2": -6.0})
        self._check(opt, recorded)

    def test_weights_that_do_not_sum_to_zero_are_accepted_if_recorded(self):
        """What PH produces with a scenario-dependent rho, and what the
        uninterrupted run was publishing too."""
        opt = self._prepped_ph()
        recorded = self._recorded(opt, {"scen0": 10.0, "scen1": -4.0,
                                        "scen2": -5.0})
        self.assertTrue(any(abs(v) > 0.1 for a in recorded.values()
                            for v in a),
                        msg="these weights were meant not to sum to zero")
        self._check(opt, recorded)

    def test_weights_other_than_the_recorded_ones_are_refused(self):
        """The message has to name the cylinder, the iteration the file was
        written at and a variable, because what it is reporting is a file on
        disk rather than anything in the run that reads it."""
        opt = self._prepped_ph()
        recorded = self._recorded(opt, {"scen0": 10.0, "scen1": -4.0,
                                        "scen2": -6.0})
        self._set_W(opt, {"scen0": 10.0, "scen1": -4.0, "scen2": -5.0})
        with self.assertRaises(checkpointing.CheckpointMismatch) as ctx:
            self._check(opt, recorded)
        message = str(ctx.exception)
        self.assertIn("do not reproduce the E[W] that file recorded",
                      message)
        self.assertIn(self.CYLINDER, message)
        self.assertIn("iteration 7", message)
        self.assertIn("DevotedAcreage", message)
        # The tolerance it was judged against, and what went into it: a
        # reader who thinks this refusal is wrong needs both numbers.
        self.assertIn("E1_tolerance", message)
        self.assertIn("size of the weights", message)

    def test_a_file_that_recorded_nothing_is_refused(self):
        opt = self._prepped_ph()
        self._set_W(opt, {"scen0": 0.0, "scen1": 0.0, "scen2": 0.0})
        with self.assertRaises(checkpointing.CheckpointMismatch):
            self._check(opt, None)

    def test_a_difference_within_the_absolute_floor_is_accepted(self):
        opt = self._prepped_ph()
        recorded = self._recorded(opt, {"scen0": 10.0, "scen1": -5.0,
                                        "scen2": -5.0})
        slack = opt.E1_tolerance / 2
        self._set_W(opt, {"scen0": 10.0, "scen1": -5.0,
                          "scen2": -5.0 + 3 * slack})
        self._check(opt, recorded)

    def test_a_difference_that_scales_with_the_weights_is_refused(self):
        """Judged against the size of the weights, so a fraction of large
        weights is refused and float dust beside them is not."""
        opt = self._prepped_ph()
        recorded = self._recorded(opt, {"scen0": 1e9, "scen1": -5e8,
                                        "scen2": -5e8})
        self._set_W(opt, {"scen0": 1e9, "scen1": -5e8, "scen2": -5e8 + 3e-3})
        self._check(opt, recorded)
        self._set_W(opt, {"scen0": 1e9, "scen1": -5e8, "scen2": -5e8 + 3e5})
        with self.assertRaises(checkpointing.CheckpointMismatch):
            self._check(opt, recorded)


if __name__ == "__main__":
    unittest.main()
