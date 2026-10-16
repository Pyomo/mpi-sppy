###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""The A/B checkpoint harness on cylinders (design section 11.1, phase 4).

The serial harness in ``test_checkpoint.py`` proves the mechanism. This proves
the thing people actually run: a hub with spokes, stopped and resumed as
separate MPI jobs.

Each leg is its own ``mpiexec`` job (see ``cylinders_ab_driver.py``), because
that is both what the design's acceptance gate asks for and what a stopped
study really does -- the resume is a new job tomorrow, sharing nothing with
today's but the checkpoint directory.

Three properties, and they are different claims:

* **The hub's primal trajectory is untouched by the stop.** Farmer is a
  deterministic LP and the hub's PH iterate does not depend on the spokes, so
  the resumed run must land bit-identically on the uninterrupted run's state.
* **The best solution survives.** That one does not live on the hub -- the
  xhat spoke holds it -- so it is preserved by the spoke's own incumbent file
  rather than by the hub checkpoint.
* **Bounds stay valid**, in the best-so-far sense the determinism contract
  (section 7) promises, rather than being reproduced exactly.
"""

import json
import math
import os
import pickle
import shutil
import subprocess
import types
import sys
import tempfile
import unittest
from unittest import mock

from mpisppy.tests.utils import get_solver, subprocess_env

solver_available, solver_name, persistent_available, persistent_solver_name = \
    get_solver()

_HERE = os.path.dirname(os.path.abspath(__file__))
_DRIVER = os.path.join(_HERE, "cylinders_ab_driver.py")
#: generic_cylinders resolves --module-name with importlib, so this is the
#: dotted name rather than a path -- and being importable makes the legs
#: independent of the directory mpiexec starts in.
_FARMER = "mpisppy.tests.examples.farmer"

mpiexec_available = shutil.which("mpiexec") is not None


class _ResumeABMixin:
    """The three A/B legs and what they must show, per configuration."""

    #: One rank per cylinder. Multi-rank cylinders are refused at setup until
    #: phase 2, so this is the whole supported space for now.
    NP = 3
    N = 5
    STOP = 2
    #: Set by the concrete cases below.
    MODULE = None
    MODEL_ARGS = ()
    SPOKE_ARGS = ()
    INCUMBENT_SPOKE = None

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _leg(self, name, *extra_args, spoke_writes_may_fail=False):
        """Run one mpiexec job and return the hub's snapshot."""
        out_path = os.path.join(self._tmp.name, f"{name}.json")
        cmd = [
            "mpiexec", "-np", str(self.NP),
            sys.executable, "-m", "mpi4py", _DRIVER,
            "--out", out_path,
            "--module-name", self.MODULE,
            *self.MODEL_ARGS,
            "--solver-name", solver_name,
            *self.SPOKE_ARGS,
            # The comparison needs both legs to run the same iterations, so
            # every early exit the cylinders add has to be off: no
            # inter-cylinder convergence, no gap-based termination.
            "--intra-hub-conv-thresh", "-1",
            "--rel-gap", "0.0", "--abs-gap", "0.0",
            *extra_args,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=1800, check=False,
                                env=subprocess_env())
        self.assertEqual(
            result.returncode, 0,
            msg=f"leg {name!r} failed:\n{result.stdout[-4000:]}\n"
                f"{result.stderr[-4000:]}")
        self.assertTrue(
            os.path.exists(out_path),
            msg=f"leg {name!r} wrote no hub snapshot:\n{result.stdout[-4000:]}")
        # Checkpoint write failures warn and continue by design, so a broken
        # write does not fail a leg -- it just prints. A resume-only leg
        # attempting a write it has nowhere to put got all the way through
        # the harness this way once.
        if not spoke_writes_may_fail:
            self.assertNotIn(
                "could not write its incumbent", result.stdout,
                msg=f"leg {name!r} failed to write a spoke incumbent")
        self.assertNotIn(
            "WARNING: checkpoint write failed", result.stdout,
            msg=f"leg {name!r} failed to write a hub checkpoint")
        with open(out_path) as f:
            snapshot = json.load(f)
        # Whatever the spokes reported about their own restore.
        snapshot["spokes"] = []
        for fname in sorted(os.listdir(self._tmp.name)):
            if fname.startswith(f"{name}.json.spoke"):
                with open(os.path.join(self._tmp.name, fname)) as f:
                    snapshot["spokes"].append(json.load(f))
        return snapshot

    def _run_ab(self):
        reference = self._leg("A", "--max-iterations", str(self.N))
        stopped = self._leg("B1", "--max-iterations", str(self.STOP),
                            "--checkpoint-dir", self.ckpt_dir)
        # --max-iterations bounds this run, so leg B2 asks for the iterations
        # B1 did not do rather than for the study total.
        resumed = self._leg("B2", "--max-iterations", str(self.N - self.STOP),
                            "--resume-from", self.ckpt_dir)
        return reference, stopped, resumed

    def _assert_the_credited_spoke_holds_the_bound(self, snapshot):
        """The cylinder credited with the hub's inner bound writes the
        solution, so it has to hold one with that objective; otherwise the
        run reports one number and writes a solution worth another."""
        credited = [s for s in snapshot["spokes"]
                    if s["strata_rank"] == snapshot["last_ib_idx"]]
        self.assertEqual(
            len(credited), 1,
            msg=f"the inner bound is credited to cylinder "
                f"{snapshot['last_ib_idx']}, which is not an xhat spoke: "
                f"{snapshot['spokes']}")
        self.assertIsNotNone(credited[0]["best_solution_obj_val"],
                             msg="the credited spoke holds no solution")
        self.assertAlmostEqual(
            credited[0]["best_solution_obj_val"], snapshot["BestInnerBound"],
            delta=1e-6 * abs(snapshot["BestInnerBound"]),
            msg="the hub reports an incumbent objective that the spoke it "
                "credits does not hold")

    def test_cylinders_resume_matches_an_uninterrupted_run(self):
        reference, stopped, resumed = self._run_ab()

        # Without this the rest proves only that farmer is deterministic: a
        # leg that ignored the checkpoint and ran all N iterations from
        # scratch would land on the same state and pass everything below.
        self.assertTrue(resumed["resumed"],
                        msg="the third leg started from scratch")
        self.assertEqual(resumed["resume_iteration"], self.STOP)
        self.assertEqual(stopped["iteration"], self.STOP)
        self.assertEqual(resumed["iteration"], self.N)

        want, got = reference["state"], resumed["state"]
        self.assertEqual(set(want), set(got))
        worst = max((abs(want[k] - got[k]) for k in want), default=0.0)
        self.assertEqual(
            worst, 0.0,
            msg=f"the resumed hub differs from the uninterrupted one by "
                f"{worst}; farmer is a deterministic LP and the hub's primal "
                f"trajectory does not depend on the spokes, so this should be "
                f"bit-identical")
        self.assertEqual(reference["trivial_bound"], resumed["trivial_bound"])

    def test_the_spoke_incumbent_survives_the_stop(self):
        """The answer lives on the xhat spoke, not in the hub checkpoint."""
        _, stopped, resumed = self._run_ab()

        spokes_dir = os.path.join(self.ckpt_dir, "spokes")
        written = os.listdir(spokes_dir)
        self.assertTrue(
            any(f.startswith(f"spoke_{self.INCUMBENT_SPOKE}") for f in written),
            msg=f"the xhat spoke checkpointed no incumbent: {written}")

        # The spoke has to say it restored. Comparing bounds alone proves
        # nothing here: farmer is deterministic, so a spoke that read nothing
        # would re-find the same incumbent within an iteration or two and the
        # comparison below would pass regardless.
        restored = [s for s in resumed["spokes"]
                    if s["restored_incumbent_obj"] is not None]
        self.assertTrue(
            restored,
            msg=f"no spoke restored an incumbent: {resumed['spokes']}")
        self.assertEqual(
            restored[0]["restored_incumbent_obj"], stopped["BestInnerBound"],
            msg="the spoke restored something other than the incumbent its "
                "own checkpoint held")

        # The objective can match while the values do not, so compare the
        # solution too: every variable of the file the stopped leg wrote,
        # against the cache the resumed spoke built on models its
        # scenario_creator made fresh in another process. The resumed leg
        # only reads the directory, so the file is still the stopped leg's.
        files = [f for f in written
                 if f.startswith(f"spoke_{self.INCUMBENT_SPOKE}")]
        self.assertEqual(len(files), 1, msg=f"expected one incumbent file: "
                                            f"{written}")
        with open(os.path.join(spokes_dir, files[0]), "rb") as f:
            on_disk = {sname: entry["values"] for sname, entry
                       in pickle.load(f)["solutions"].items()}
        self.assertTrue(all(on_disk.values()),
                        msg="the incumbent file holds no variable values")
        self.assertEqual(
            restored[0]["restored_values"], on_disk,
            msg="the resumed spoke's incumbent is not the solution its "
                "checkpoint file holds")

        # Minimization: a smaller inner bound is a better incumbent, and the
        # resumed run must not report a worse one than the leg it resumed.
        self.assertLessEqual(
            resumed["BestInnerBound"], stopped["BestInnerBound"],
            msg="the resumed run reports a worse incumbent than its "
                "checkpoint; the spoke's best xhat was lost")

        # The hub hears from its spokes only after its first resumed solve,
        # so that solve sees the restored incumbent only if the hub restored
        # it itself.
        self.assertEqual(
            resumed["first_BestInnerBound"],
            restored[0]["restored_incumbent_obj"],
            msg="the resumed hub's first iteration ran without the incumbent "
                "the spoke restored")
        self._assert_the_credited_spoke_holds_the_bound(resumed)

    def test_bounds_stay_valid_after_a_resume(self):
        """Best-so-far, not reproduced: bound timing is spoke-dependent.

        Read what each assertion can and cannot see. Without an outer-bound
        spoke feeding it, the hub's own ``best_bound_obj_val`` is set once
        from the trivial bound and never moves, so the two sandwich
        assertions compare one constant against another across thousands of
        units and only a non-number can redden them. They are here because
        crossed bounds are worth catching cheaply, not because they guard the
        restore. The claim that the restore is what carries the hub's bound
        across is the third one, and it needs the stopped leg to mean
        anything.
        """
        _, stopped, resumed = self._run_ab()
        self.assertLessEqual(
            resumed["BestOuterBound"], resumed["BestInnerBound"],
            msg="the resumed run's outer bound crossed its incumbent")
        self.assertLessEqual(
            resumed["best_bound_obj_val"], resumed["BestInnerBound"],
            msg="the hub's restored best bound crossed the incumbent")
        self.assertEqual(
            resumed["best_bound_obj_val"], stopped["best_bound_obj_val"],
            msg="the resumed hub did not carry the checkpointed best bound "
                "across; it recomputed or discarded it")
        # Against the checkpoint, not against the stopped leg's last word:
        # a spoke's bound can reach the hub after the final checkpoint is
        # written, and nothing carries that one.
        checkpointed = self._checkpointed_outer_bound()
        self.assertIsNotNone(checkpointed,
                             msg="the checkpoint recorded no outer bound")
        self.assertGreaterEqual(
            resumed["BestOuterBound"], checkpointed,
            msg="the resumed run ended with a weaker outer bound than its "
                "checkpoint holds")

    def _checkpointed_outer_bound(self):
        with open(os.path.join(self.ckpt_dir, "manifest.json")) as f:
            generation = json.load(f)["generation"]
        leaf = os.path.join(self.ckpt_dir, "hub", f"gen_{generation:04d}",
                            "hub_rank_0000.pkl")
        with open(leaf, "rb") as f:
            return pickle.load(f)["best_outer_bound"]


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestFarmerCylindersResumeAB(_ResumeABMixin, unittest.TestCase):
    """PH hub + lagrangian + xhatshuffle on the deterministic-LP baseline."""

    MODULE = _FARMER
    #: Late enough that the Lagrangian bounds the resumed spoke re-finds in
    #: its first iterations are weaker than the checkpointed one, so a resume
    #: that drops the hub's outer bound fails the bound test.
    N = 8
    STOP = 6
    MODEL_ARGS = ("--num-scens", "3", "--default-rho", "1")
    SPOKE_ARGS = ("--lagrangian", "--xhatshuffle")
    INCUMBENT_SPOKE = "XhatShuffleInnerBound"

    def test_the_xhatshuffle_cursor_survives_the_stop(self):
        """The spoke picks its exploration up rather than starting over.

        This leaves no trace in the answer -- farmer is deterministic, so the
        resumed spoke reaches the same incumbent whether it walks on from
        where it was or from the start of the order -- so the only way to see
        it is to ask the spoke what it was handed.
        """
        _, stopped, resumed = self._run_ab()

        self.assertTrue(
            any(s["final_loop_state"] for s in stopped["spokes"]),
            msg=f"the xhatshuffle spoke tracked no cursor: "
                f"{stopped['spokes']}")

        # "applied", not "read": the Checkpointer reads the file in pre_iter0
        # whether or not the loop ever adopts what it found, so watching the
        # read would score a discarded cursor as a success.
        applied = [s for s in resumed["spokes"] if s["applied_loop_state"]]
        self.assertTrue(
            applied,
            msg=f"no spoke adopted a restored cursor: {resumed['spokes']}")
        # Against the file rather than the stopped run's final counter: the
        # write happens at the bottom of a pass and the counter advances
        # afterwards, so the two legitimately differ by one pass.
        self.assertEqual(applied[0]["applied_loop_state"],
                         self._checkpointed_loop_state(),
                         msg="the spoke restored something other than the "
                             "cursor its own checkpoint held")

    def _checkpointed_loop_state(self):
        """The loop state sitting in the xhat spoke's checkpoint file."""
        spokes_dir = os.path.join(self.ckpt_dir, "spokes")
        for fname in sorted(os.listdir(spokes_dir)):
            if fname.startswith(f"spoke_{self.INCUMBENT_SPOKE}"):
                with open(os.path.join(spokes_dir, fname), "rb") as f:
                    return pickle.load(f)["loop_state"]
        raise AssertionError(f"no {self.INCUMBENT_SPOKE} file in {spokes_dir}")

    def test_a_bound_the_spoke_never_wrote_is_not_restored(self):
        """The hub's last word can be newer than a spoke's file: a spoke
        publishes an improvement before it writes it, and a failed write is
        not retried. Here every spoke write fails, so the hub heard an
        incumbent that no file holds, and the resumed spoke re-finds a worse
        one within the iterations it gets. A hub that restored its own number
        would credit that spoke with it, and the spoke would then write its
        worse solution under the better objective."""
        # A regular file where the spokes' directory goes makes every spoke
        # write fail while the hub's own checkpoint is written normally.
        os.makedirs(self.ckpt_dir)
        with open(os.path.join(self.ckpt_dir, "spokes"), "w"):
            pass
        stopped = self._leg("B1", "--max-iterations", str(self.STOP),
                            "--checkpoint-dir", self.ckpt_dir,
                            spoke_writes_may_fail=True)
        self.assertLess(stopped["BestInnerBound"], math.inf,
                        msg="the stopped leg found no incumbent to lose")
        resumed = self._leg("B2", "--max-iterations", "1",
                            "--resume-from", self.ckpt_dir)
        # No spoke file holds an incumbent, so the resumed hub starts with
        # none -- whatever the resumed spoke later re-finds.
        self.assertEqual(
            resumed["first_BestInnerBound"], math.inf,
            msg="the resumed hub restored an incumbent no spoke file holds")
        self._assert_the_credited_spoke_holds_the_bound(resumed)


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestStochAdmmCylindersResumeAB(_ResumeABMixin, unittest.TestCase):
    """The same harness on a stoch-ADMM configuration.

    Worth its own case because the ADMM wrapper is where checkpointing has
    the most to get wrong (design section 8.2): scenario names are wrapped,
    so they reach the checkpoint's file names; the wrapper holds its own
    references to the models a resume replaces; and the variable-probability
    mask and fixed-at-0 dummy vars have to survive the dill round-trip. FWPH
    is not in the spoke set because it does not support variable probability.
    """

    MODULE = "mpisppy.tests.examples.stoch_distr.stoch_distr"
    MODEL_ARGS = ("--stoch-admm", "--num-stoch-scens", "4",
                  "--num-admm-subproblems", "2", "--default-rho", "10")
    SPOKE_ARGS = ("--lagrangian", "--xhatxbar")
    INCUMBENT_SPOKE = "XhatXbarInnerBound"

    def test_a_spoke_without_a_cursor_carries_none(self):
        """xhatxbar re-evaluates from scratch whenever new nonants arrive, so
        it has no loop position worth resuming, and neither it nor its
        checkpoint file claims one."""
        _, stopped, _ = self._run_ab()
        (marker,) = [s for s in stopped["spokes"]
                     if s["cylinder"] == self.INCUMBENT_SPOKE]
        self.assertIsNone(marker["final_loop_state"])
        spokes_dir = os.path.join(self.ckpt_dir, "spokes")
        (fname,) = [f for f in os.listdir(spokes_dir)
                    if f.startswith(f"spoke_{self.INCUMBENT_SPOKE}")]
        with open(os.path.join(spokes_dir, fname), "rb") as f:
            self.assertIsNone(pickle.load(f)["loop_state"])


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestADroppedSpokeFileIsReported(unittest.TestCase):
    """Resume without a spoke that held an incumbent, and the run says so.

    The spoke that would have reported the file is the one that is not
    running, so it falls to the hub. Without this the dropped spoke's
    incumbent -- possibly the study's best -- was left behind in silence.
    """

    COMMON = ("--module-name", _FARMER, "--num-scens", "3",
              "--default-rho", "1", "--lagrangian",
              "--intra-hub-conv-thresh", "-1", "--rel-gap", "0.0",
              "--abs-gap", "0.0")

    def _run(self, np, *args):
        cmd = ["mpiexec", "-np", str(np), sys.executable, "-m", "mpi4py",
               "-m", "mpisppy.generic_cylinders", "--solver-name",
               solver_name, *self.COMMON, *args]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=1800, check=False,
                                env=subprocess_env())
        self.assertEqual(result.returncode, 0,
                         msg=result.stdout[-4000:] + result.stderr[-4000:])
        return result.stdout

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt_dir = os.path.join(cls._tmp.name, "ckpt")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_only_the_dropped_spoke_is_named(self):
        self._run(4, "--xhatshuffle", "--xhatxbar", "--max-iterations", "3",
                  "--checkpoint-dir", self.ckpt_dir)
        written = os.listdir(os.path.join(self.ckpt_dir, "spokes"))
        self.assertTrue(any("XhatXbarInnerBound" in f for f in written),
                        msg=f"xhatxbar wrote no incumbent to drop: {written}")
        same = self._run(4, "--xhatshuffle", "--xhatxbar",
                         "--max-iterations", "1",
                         "--resume-from", self.ckpt_dir)
        self.assertNotIn("no such cylinder runs", same)
        dropped = self._run(3, "--xhatshuffle", "--max-iterations", "1",
                            "--resume-from", self.ckpt_dir)
        self.assertIn("holds a file for XhatXbarInnerBound", dropped)
        self.assertNotIn("holds a file for XhatShuffleInnerBound", dropped)


class TestRestoredIncumbentIsRepublished(unittest.TestCase):
    """A spoke that restores an incumbent has written nothing to the directory
    this run writes to, unless that is the very directory it read.

    _last_written_obj means "what the file in self.ckpt_dir already holds", so
    seeding it from a restore out of a *different* directory makes the skip
    test decline to write until the spoke strictly improves on what it
    restored. Stopping today and resuming tomorrow into a fresh
    --checkpoint-dir is the documented flow, and a study whose xhat has
    stopped improving is exactly where that loses the answer -- silently, at
    exit code 0.
    """

    def _checkpointer(self, ckpt_dir, resume_from):
        from mpisppy.extensions.checkpointer import Checkpointer
        ext = Checkpointer.__new__(Checkpointer)
        ext.opt = types.SimpleNamespace(
            options={"resume_from": resume_from},
            spcomm=types.SimpleNamespace(
                best_inner_bound=None,
                # XhatBase's default: the whole loop state is the progress.
                # Spelled out because a SimpleNamespace inherits nothing, and
                # the restore path reads this on branches that carry a cursor.
                loop_state_progress=lambda state: state,
                send_bound=lambda v: None,
                send_best_xhat=lambda: None,
            ),
            cylinder_rank=0,
        )
        ext.ckpt_dir = ckpt_dir
        ext.write_enabled = ckpt_dir is not None
        ext.spoke_mode = True
        ext._last_written_obj = None
        ext._last_failed_obj = None
        ext.restored_incumbent_obj = None
        ext._spoke_identity = lambda: ("XhatShuffleInnerBound", 2)
        return ext

    def _restore(self, ckpt_dir, resume_from):
        from mpisppy.extensions import checkpointer as mod
        ext = self._checkpointer(ckpt_dir, resume_from)
        state = {"best_inner_bound": -108382.22}
        with mock.patch.object(mod.ckpt, "load_spoke_incumbent",
                               return_value=state), \
             mock.patch.object(mod.ckpt, "restore_spoke_incumbent",
                               return_value=-108382.22):
            ext._restore_incumbent()
        return ext

    def test_a_fresh_directory_gets_the_restored_incumbent(self):
        ext = self._restore(ckpt_dir="/tmp/ck2", resume_from="/tmp/ck1")
        self.assertEqual(ext.restored_incumbent_obj, -108382.22)
        self.assertIsNone(
            ext._last_written_obj,
            msg="the spoke believes it has already written its incumbent to a "
                "directory it has never written to, so it will not publish "
                "until it improves")

    def test_the_same_directory_is_not_rewritten(self):
        ext = self._restore(ckpt_dir="/tmp/ck1", resume_from="/tmp/ck1")
        self.assertEqual(ext._last_written_obj, -108382.22,
                         msg="resuming in place should not rewrite the file "
                             "it just read")

    def test_the_same_directory_spelled_two_ways(self):
        ext = self._restore(ckpt_dir="/tmp/ck1", resume_from="/tmp/./ck1/")
        self.assertEqual(ext._last_written_obj, -108382.22)

    def test_a_restore_only_run_writes_nothing_anywhere(self):
        ext = self._restore(ckpt_dir=None, resume_from="/tmp/ck1")
        self.assertEqual(ext.restored_incumbent_obj, -108382.22)
        self.assertIsNone(ext._last_written_obj)


class TestAFailedSpokeWriteIsNotRetriedEveryPass(unittest.TestCase):
    """The hook runs on every pass of a loop that spins while it waits on the
    hub, so a write that fails must not be tried again until there is a new
    incumbent to write."""

    def _checkpointer(self):
        from mpisppy.extensions.checkpointer import Checkpointer
        ext = Checkpointer.__new__(Checkpointer)
        self.cursor = 0
        ext.opt = types.SimpleNamespace(
            options={}, best_solution_obj_val=-108382.22, cylinder_rank=0,
            spcomm=types.SimpleNamespace(
                best_inner_bound=-108382.22,
                checkpoint_loop_state=lambda: {"cursor": self.cursor},
                loop_state_progress=lambda state: state["cursor"]))
        ext.ckpt_dir = "/tmp/ck"
        ext.write_enabled = True
        ext.spoke_mode = True
        ext._last_written_obj = None
        ext._last_written_loop_progress = None
        ext._last_failed_obj = None
        ext._spoke_identity = lambda: ("XhatShuffleInnerBound", 0)
        ext._class_ordinal_and_count = lambda: (0, 1)
        return ext

    def test_a_failure_waits_for_the_next_improvement(self):
        from mpisppy.extensions import checkpointer as mod
        ext = self._checkpointer()
        writer = mock.Mock(side_effect=OSError("disk full"))
        with mock.patch.object(mod.ckpt, "write_spoke_incumbent", writer), \
             mock.patch.object(mod, "global_toc") as toc:
            for _ in range(50):
                ext._spoke_checkpoint()
            self.assertEqual(writer.call_count, 1,
                             msg="an unchanged incumbent was written again "
                                 "after its write failed")
            self.assertEqual(toc.call_count, 1)

            # The cursor moves far more often than the incumbent improves.
            for step in range(1, 50):
                self.cursor = step
                ext._spoke_checkpoint()
            self.assertEqual(writer.call_count, 1,
                             msg="a cursor move alone retried a write that "
                                 "failed for this incumbent")

            ext.opt.best_solution_obj_val = -108500.0
            ext._spoke_checkpoint()
            self.assertEqual(writer.call_count, 2,
                             msg="a new incumbent was not tried after an "
                                 "earlier write failed")

    def test_a_dual_cylinder_failure_is_reported_by_that_rank(self):
        from mpisppy.extensions import checkpointer as mod
        ext = self._checkpointer()
        ext.opt.cylinder_rank = 1
        ext.opt._PHIter = 3
        writer = mock.Mock(side_effect=OSError("disk full"))
        with mock.patch.object(mod.ckpt, "write_dual_spoke_state", writer), \
             mock.patch("mpisppy.phbase.Wbar_by_node", return_value={}), \
             mock.patch.object(mod, "global_toc") as toc:
            ext._dual_spoke_checkpoint()
        (message, prints), _ = toc.call_args
        self.assertIn("rank 1", message)
        self.assertTrue(prints, msg="rank 1's failure was not printed")

    def test_a_dual_cylinder_names_the_extensions_it_does_not_carry(self):
        from mpisppy.extensions import checkpointer as mod
        from mpisppy.extensions.checkpointer import Checkpointer
        ext = self._checkpointer()
        ext.dual_spoke_mode = True
        ext.opt.options = {"resume_from": "/tmp/ck"}
        other = type("GradRho", (), {})()
        ext.opt.extobject = types.SimpleNamespace(
            extdict={"Checkpointer": ext, "GradRho": other})
        ext._spoke_identity = lambda: ("PHDualSpoke", 0)
        assert isinstance(ext, Checkpointer)
        state = {"generation": 3, "class_count": None, "Wbar": {}}

        def restore(found):
            with mock.patch.object(mod.ckpt, "run_agreed",
                                   return_value=found), \
                 mock.patch.object(mod.ckpt, "agree_dual_spoke_restore",
                                   return_value=(found, None)), \
                 mock.patch.object(mod.ckpt,
                                   "require_restored_duals_match_their_file"), \
                 mock.patch("mpisppy.phbase.Wbar_by_node"), \
                 mock.patch("mpisppy.phbase.W_magnitude_by_node"), \
                 mock.patch.object(mod, "global_toc") as toc:
                ext.post_iter0()
            return " ".join(str(c.args[0]) for c in toc.call_args_list)

        said = restore(state)
        self.assertIn("GradRho", said)
        self.assertNotIn("Checkpointer", said)
        # Nothing restored, so nothing to say about what was not carried.
        self.assertNotIn("GradRho", restore(None))

    def test_a_failure_on_any_rank_is_reported_by_that_rank(self):
        """Each rank writes its own file, so a failure on rank 1 is rank 1's
        alone; a rank-0-only warning left it silent."""
        from mpisppy.extensions import checkpointer as mod
        ext = self._checkpointer()
        ext.opt.cylinder_rank = 1
        writer = mock.Mock(side_effect=OSError("disk full"))
        with mock.patch.object(mod.ckpt, "write_spoke_incumbent", writer), \
             mock.patch.object(mod, "global_toc") as toc:
            ext._spoke_checkpoint()
        (message, prints), _ = toc.call_args
        self.assertIn("rank 1", message)
        self.assertTrue(prints, msg="rank 1's failure was not printed")

    def test_a_nan_incumbent_is_not_retried_either(self):
        """NaN != NaN, so an equality test alone makes an unchanged NaN
        incumbent look new on every pass. A fresh NaN object each pass, so
        that nothing here rests on identity."""
        from mpisppy.extensions import checkpointer as mod
        ext = self._checkpointer()
        writer = mock.Mock(side_effect=ValueError("no finite objective"))
        with mock.patch.object(mod.ckpt, "write_spoke_incumbent", writer), \
             mock.patch.object(mod, "global_toc") as toc:
            for _ in range(50):
                ext.opt.best_solution_obj_val = float("nan")
                ext._spoke_checkpoint()
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(toc.call_count, 1)


class TestEverySpokeGivenTheCheckpointerDrivesIt(unittest.TestCase):
    """cfg_vanilla attaches the Checkpointer in one place, and the spokes that
    place builds do not all derive from the same class.

    add_checkpointing is called from _Xhat_Eval_spoke_foundation, which builds
    the four xhat spokes *and* the L-shaped xhatter and the two slammers. The
    latter three derive from InnerBoundNonantSpoke rather than
    XhatInnerBoundBase, which is where the two hooks used to live, so the
    extension was constructed, probed the checkpoint directory, and was then
    inert: it never wrote and never restored, with no warning. The user doc
    says every xhat spoke keeps its own file under ``spokes/``.
    """

    #: Every cylinder class cfg_vanilla builds through
    #: _Xhat_Eval_spoke_foundation, and therefore hands a Checkpointer.
    SPOKES = (
        ("mpisppy.cylinders.xhatlooper_bounder", "XhatLooperInnerBound"),
        ("mpisppy.cylinders.xhatshufflelooper_bounder", "XhatShuffleInnerBound"),
        ("mpisppy.cylinders.xhatspecific_bounder", "XhatSpecificInnerBound"),
        ("mpisppy.cylinders.xhatxbar_bounder", "XhatXbarInnerBound"),
        ("mpisppy.cylinders.lshaped_bounder", "XhatLShapedInnerBound"),
        ("mpisppy.cylinders.slam_heuristic", "SlamMaxHeuristic"),
        ("mpisppy.cylinders.slam_heuristic", "SlamMinHeuristic"),
    )

    def _classes(self):
        import importlib
        for module_name, class_name in self.SPOKES:
            mod = importlib.import_module(module_name)
            yield class_name, getattr(mod, class_name), mod

    def test_the_foundation_still_builds_exactly_these(self):
        """If a spoke is added to _Xhat_Eval_spoke_foundation it gets a
        Checkpointer, so it has to appear above and drive the hooks."""
        import ast
        import inspect
        from mpisppy.utils import cfg_vanilla
        tree = ast.parse(inspect.getsource(cfg_vanilla))
        builders = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_Xhat_Eval_spoke_foundation")
        self.assertEqual(
            builders, len(self.SPOKES),
            msg="the number of spokes built through the foundation that "
                "attaches the Checkpointer changed; add it to SPOKES here "
                "and make its loop drive the hooks")

    def test_each_one_asks_for_its_checkpointed_incumbent(self):
        """Look at the method that runs, not at the module around it.

        The four XhatInnerBoundBase spokes reach the extension through
        ``xhat_prep``, which is defined in cylinders/xhatbase.py -- so a test
        that reads each spoke's own module sees the word ``xhat_prep()``
        where the spoke *calls* it and passes without ever looking at the
        body that does the work. Resolving the attribute walks the MRO to
        wherever it is really defined.
        """
        import inspect
        for name, cls, _ in self._classes():
            with self.subTest(spoke=name):
                entry_name = ("xhat_prep" if hasattr(cls, "xhat_prep")
                              else "restore_checkpointed_incumbent")
                self.assertTrue(
                    hasattr(cls, entry_name),
                    msg=f"{name} has neither pre-loop entry point")
                # Every definition of it along the MRO, most derived first.
                chain = [inspect.getsource(c.__dict__[entry_name])
                         for c in cls.__mro__ if entry_name in c.__dict__]
                reached = False
                for depth, source in enumerate(chain):
                    if "extobject.pre_iter0" in source:
                        reached = True
                        break
                    # An override that does not give the hook itself has to
                    # hand on to the one that does, or the chain stops here.
                    self.assertIn(
                        "super()", source,
                        msg=f"{name}.{entry_name} (override {depth}) neither "
                            f"gives its extensions their hook nor calls "
                            f"super(), so a Checkpointer attached to it can "
                            f"never restore")
                self.assertTrue(
                    reached,
                    msg=f"no definition of {name}.{entry_name} along its "
                        f"MRO calls extobject.pre_iter0(), so the extensions "
                        f"never get their pre-loop hook. Note the marker is "
                        f"the extension call specifically: xhat_prep also "
                        f"calls pre_iter0() on the xhatter, which says "
                        f"nothing about extensions.")

    def test_each_one_offers_a_checkpoint_point(self):
        import inspect
        for name, cls, mod in self._classes():
            with self.subTest(spoke=name):
                self.assertIn(
                    "self.maybe_checkpoint()", inspect.getsource(mod),
                    msg=f"{name}'s loop never offers a checkpoint point, so a "
                        f"Checkpointer attached to it can never write")

    def test_the_hooks_reach_the_extension(self):
        """Inherited from InnerBoundNonantSpoke, so all three families get
        them and not just the XhatInnerBoundBase ones."""
        for name, cls, _ in self._classes():
            with self.subTest(spoke=name):
                spoke = cls.__new__(cls)
                seen = []
                spoke.opt = types.SimpleNamespace(
                    extensions=object(),
                    extobject=types.SimpleNamespace(
                        pre_iter0=lambda: seen.append("pre_iter0"),
                        maybe_checkpoint=lambda: seen.append("maybe_checkpoint"),
                    ),
                )
                spoke.restore_checkpointed_incumbent()
                spoke.maybe_checkpoint()
                self.assertEqual(seen, ["pre_iter0", "maybe_checkpoint"])

    def test_a_spoke_without_extensions_does_nothing(self):
        for name, cls, _ in self._classes():
            with self.subTest(spoke=name):
                spoke = cls.__new__(cls)
                spoke.opt = types.SimpleNamespace(extensions=None)
                spoke.restore_checkpointed_incumbent()   # must not raise
                spoke.maybe_checkpoint()

    def test_each_one_restores_its_extensions_state(self):
        """Every one of these writes its extensions' state, so every one has
        to hand it back on a resume. The L-shaped xhatter and the slammers
        used to write it and drop it: only xhat_prep restored it."""
        import inspect
        import mpisppy.cylinders.xhatbase as xhatbase
        for name, cls, mod in self._classes():
            with self.subTest(spoke=name):
                source = inspect.getsource(
                    xhatbase if issubclass(cls, xhatbase.XhatInnerBoundBase)
                    else mod)
                self.assertIn(
                    "self._restore_extension_state_if_resuming()", source,
                    msg=f"{name}'s prep never hands its extensions the state "
                        f"its Checkpointer read, so a resume drops it")

    def test_the_restore_reaches_the_extension(self):
        """Run the shared restore on each spoke, single rank: the state the
        Checkpointer read is handed to the extension it belongs to."""
        from mpisppy.extensions.extension import Extension

        class _Counter(Extension):
            def checkpoint_state(self):
                return {"n": 0}

            def restore_state(self, state):
                self.restored = state

        for name, cls, _ in self._classes():
            with self.subTest(spoke=name):
                counter = _Counter.__new__(_Counter)
                holder = types.SimpleNamespace(restored_extension_state={
                    "extensions": {"_Counter": {"n": 7}}, "converger": None})
                spoke = cls.__new__(cls)
                spoke.opt = types.SimpleNamespace(
                    n_proc=1, cylinder_rank=0, convobject=None,
                    extobject=types.SimpleNamespace(
                        extdict={"Holder": holder, "_Counter": counter}))
                spoke._restore_extension_state_if_resuming()
                self.assertEqual(counter.restored, {"n": 7})


class TestEveryWriterProbesItsOwnFile(unittest.TestCase):
    """Several cylinders share one --checkpoint-dir and all of them probe it.

    The hub and every xhat spoke carrying a Checkpointer create and delete a
    probe file in the same directory during setup. A name they share makes
    whichever removes it second fail on a directory that is perfectly
    writable, and the run dies at startup rather than checkpointing. It is a
    race, so it needs two writers to show at all: one xhat spoke never fails,
    two fail most of the time.
    """

    def _probe_names(self, global_ranks):
        """The probe file each of these ranks leaves behind, in order."""
        from mpisppy.extensions.checkpointer import Checkpointer
        import mpisppy.utils.checkpointing as ckpt

        # The probe moved out to checkpointing.probe_directory_is_writable
        # once several ranks per cylinder had to agree on it. Drive whichever
        # one this branch actually calls, so the property is tested where it
        # lives rather than where it used to.
        helper = getattr(ckpt, "probe_directory_is_writable", None)
        seen = []
        with tempfile.TemporaryDirectory() as ckpt_dir:
            for gr in global_ranks:
                opt = types.SimpleNamespace(global_rank=gr, cylinder_rank=0)
                # Stub the cleanup so the file each probe would remove stays
                # visible to the assertion.
                if helper is not None:
                    with mock.patch("mpisppy.utils.checkpointing.os.remove"):
                        helper(opt, ckpt_dir)
                else:
                    ext = Checkpointer.__new__(Checkpointer)
                    ext.ckpt_dir = ckpt_dir
                    with mock.patch(
                            "mpisppy.extensions.checkpointer.os.remove"):
                        Checkpointer._probe_directory(ext, opt)
                seen = sorted(os.listdir(ckpt_dir))
        return seen

    def test_two_cylinders_do_not_share_a_probe_file(self):
        """Cylinder rank is 0 for both; only the global rank separates them."""
        names = self._probe_names([0, 3])
        self.assertEqual(
            len(names), 2,
            msg=f"two cylinders wrote {len(names)} probe file(s) {names}; "
                f"sharing one means whichever removes it second fails on a "
                f"writable directory")

    def test_the_probe_name_carries_the_global_rank(self):
        names = self._probe_names([7])
        self.assertEqual(names, [".mpisppy_write_probe_0007"])


if __name__ == "__main__":
    unittest.main()
