###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""The A/B checkpoint harness on multi-rank cylinders (design phase 2).

Phase 1a checkpointed a single-rank hub and refused anything else; phase 4 put
that hub in a wheel with spokes, still one rank each. This is the phase that
lets a cylinder span ranks, which is how mpi-sppy is actually run on a cluster,
and it is where a checkpoint stops being one rank's business:

* **A generation spans every rank.** Each rank owns a different slice of the
  scenarios, so the checkpoint is the *set* of per-rank files and is resumable
  only if all of them are there. The manifest must therefore be written after
  the last rank's write, not after rank 0's -- and every rank must resume from
  its own slice. So these tests compare **every** hub rank's state across
  legs, not just rank 0's, which is where a single-rank harness stops seeing
  anything.
* **Failure is collective or it is a hang.** The design has a failed write warn
  and let the run continue; on several ranks that only works if the ranks agree
  about whether the write failed, or the one that gave up leaves the others
  waiting at a barrier forever.
* **The distribution has to be exactly reproduced.** Resuming with a different
  rank count, or with the scenarios landing on different ranks, is refused
  rather than half-restored (section 5.7).

Each leg is its own ``mpiexec`` job (``cylinders_ab_driver.py``), for the
reason the single-rank cylinders harness gives: the design's acceptance gate
asks for the resume to happen in a fresh process, and a stopped study really
does resume as tomorrow's job.

The instances are the ones section 11.1 assigns to this phase: farmer with the
scenarios spread evenly and unevenly, proper bundles (section 8.1), the
``sizes`` MIP, a full multi-rank wheel, and stoch-ADMM with and without
bundles (section 8.2), whose wrapper is where a resume has the most to get
wrong.
"""

import ast
import importlib
import inspect
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

import mpisppy.tests.multirank_agreement_driver as agreement_driver
import mpisppy.utils.checkpointing as checkpointing
from mpisppy.cylinders.xhatbase import XhatInnerBoundBase
from mpisppy.cylinders.xhatshufflelooper_bounder import (ScenarioCycler,
                                                         XhatShuffleInnerBound)
from mpisppy.extensions.checkpointer import Checkpointer
from mpisppy.phbase import PHBase
from mpisppy.tests.utils import get_solver, subprocess_env

solver_available, solver_name, persistent_available, persistent_solver_name = \
    get_solver()

_HERE = os.path.dirname(os.path.abspath(__file__))
_DRIVER = os.path.join(_HERE, "cylinders_ab_driver.py")
_FAILURE_DRIVER = os.path.join(_HERE, "multirank_failure_driver.py")
_DEADLINE_DRIVER = os.path.join(_HERE, "multirank_deadline_driver.py")
_AGREEMENT_DRIVER = os.path.join(_HERE, "multirank_agreement_driver.py")
#: generic_cylinders resolves --module-name with importlib, so these are dotted
#: names rather than paths -- which also makes the legs independent of the
#: directory mpiexec starts in.
_FARMER = "mpisppy.tests.examples.farmer"
_SIZES = "mpisppy.tests.examples.sizes.sizes"
_STOCH_DISTR = "mpisppy.tests.examples.stoch_distr.stoch_distr"

mpiexec_available = shutil.which("mpiexec") is not None


def _run_leg(tmpdir, name, np, module, model_args, spoke_args, extra_args,
             check=True):
    """Run one mpiexec job. Returns (CompletedProcess, out_path)."""
    out_path = os.path.join(tmpdir, f"{name}.json")
    cmd = [
        "mpiexec", "-np", str(np),
        sys.executable, "-m", "mpi4py", _DRIVER,
        "--out", out_path,
        "--module-name", module,
        *model_args,
        "--solver-name", solver_name,
        *spoke_args,
        # The comparison needs both legs to run the same iterations, so every
        # early exit has to be off: no inter-cylinder convergence, no
        # gap-based termination.
        "--intra-hub-conv-thresh", "-1",
        "--rel-gap", "0.0", "--abs-gap", "0.0",
        *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=3600, check=False, env=subprocess_env())
    if check and result.returncode != 0:
        raise AssertionError(
            f"leg {name!r} failed:\n{result.stdout[-4000:]}\n"
            f"{result.stderr[-4000:]}")
    return result, out_path


def _hub_ranks(out_path):
    """Every hub rank's snapshot for a leg, ordered by rank."""
    directory = os.path.dirname(out_path)
    prefix = f"{os.path.basename(out_path)}.hubrank"
    snapshots = []
    for fname in sorted(os.listdir(directory)):
        if fname.startswith(prefix):
            with open(os.path.join(directory, fname)) as f:
                snapshots.append(json.load(f))
    return sorted(snapshots, key=lambda s: s["cylinder_rank"])


def _spoke_ranks(out_path, cylinder):
    """Every rank's marker for one spoke cylinder, ordered by rank."""
    directory = os.path.dirname(out_path)
    prefix = f"{os.path.basename(out_path)}.cyl"
    markers = []
    for fname in sorted(os.listdir(directory)):
        if fname.startswith(prefix):
            with open(os.path.join(directory, fname)) as f:
                marker = json.load(f)
            if marker["cylinder"] == cylinder:
                markers.append(marker)
    return markers


def _checkpointed_outer_bounds(ckpt_dir):
    """Each hub rank's best_outer_bound in the published generation."""
    generation = _published_generation(ckpt_dir)["generation"]
    gen_dir = os.path.join(ckpt_dir, "hub", f"gen_{generation:04d}")
    bounds = {}
    for fname in os.listdir(gen_dir):
        if fname.startswith("hub_rank_") and fname.endswith(".pkl") \
                and "_scen_" not in fname:
            with open(os.path.join(gen_dir, fname), "rb") as f:
                bounds[int(fname[len("hub_rank_"):-len(".pkl")])] = \
                    pickle.load(f)["best_outer_bound"]
    return bounds


def _published_generation(ckpt_dir):
    with open(os.path.join(ckpt_dir, "manifest.json")) as f:
        return json.load(f)


class _MultiRankABMixin:
    """Three legs -- reference, stopped, resumed -- run once per class.

    Once per class rather than once per test: each leg is an mpiexec job, and
    running the whole A/B again for every assertion would multiply the CI cost
    of this file by the number of things it checks without checking anything
    new.
    """

    #: Total ranks. With no spokes the whole job is the hub, so this is the
    #: number of ranks *within* one cylinder -- which is what phase 2 is about.
    NP = 2
    #: Ranks the hub ends up with, once the wheel has split them by cylinder.
    HUB_RANKS = 2
    N = 4
    STOP = 2
    MODULE = None
    MODEL_ARGS = ()
    SPOKE_ARGS = ()
    #: False for MIP instances, where the section 7 contract promises a valid
    #: continuation rather than a reproducible trajectory under default solver
    #: settings.
    BIT_IDENTICAL = True
    #: Relative agreement required of the expected objective when
    #: BIT_IDENTICAL is False. A MIP with alternate optima can be resumed onto
    #: a different one of them, so the per-variable iterate is allowed to
    #: differ; the objective is not allowed to walk away.
    OBJECTIVE_RTOL = 1e-3

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt_dir = os.path.join(cls._tmp.name, "ckpt")

        def leg(name, *extra):
            _, out_path = _run_leg(cls._tmp.name, name, cls.NP, cls.MODULE,
                                   cls.MODEL_ARGS, cls.SPOKE_ARGS, extra)
            return _hub_ranks(out_path)

        cls.reference = leg("A", "--max-iterations", str(cls.N))
        cls.stopped = leg("B1", "--max-iterations", str(cls.STOP),
                          "--checkpoint-dir", cls.ckpt_dir)
        # --max-iterations bounds this run, so leg B2 asks for the iterations
        # B1 did not do rather than for the study total.
        cls.resumed = leg("B2", "--max-iterations", str(cls.N - cls.STOP),
                          "--resume-from", cls.ckpt_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_rank_took_part(self):
        """A hub really did span ranks, and each owns a distinct slice.

        Without this the rest of the file could pass on a run that quietly
        put every scenario on rank 0 -- which is the single-rank case phase 1a
        already covered, wearing a multi-rank command line.
        """
        self.assertEqual(len(self.reference), self.HUB_RANKS)
        for leg in (self.reference, self.stopped, self.resumed):
            owned = [tuple(snap["scenario_names"]) for snap in leg]
            self.assertEqual(len(set(owned)), len(owned),
                             msg=f"ranks share scenarios: {owned}")
            for snap in leg:
                self.assertEqual(snap["n_proc"], self.HUB_RANKS)
                self.assertTrue(snap["scenario_names"],
                                msg="a hub rank owns no scenarios")

    def test_every_rank_resumed(self):
        for snap in self.resumed:
            self.assertTrue(
                snap["resumed"],
                msg=f"rank {snap['cylinder_rank']} started from scratch")
            self.assertEqual(snap["resume_iteration"], self.STOP)
            self.assertEqual(snap["iteration"], self.N)
        for snap in self.stopped:
            self.assertEqual(snap["iteration"], self.STOP)

    def test_the_generation_holds_every_rank_and_is_published_once(self):
        """The manifest names one generation, complete on every rank.

        This is the phase-2 write protocol stated as a file-system fact: a
        manifest published before the last rank finished would name a
        generation missing that rank's leaf file, and the resume above would
        have refused it.
        """
        manifest = _published_generation(self.ckpt_dir)
        self.assertEqual(manifest["generation"], self.STOP)
        self.assertEqual(manifest["n_proc"], self.HUB_RANKS)

        hub_dir = os.path.join(self.ckpt_dir, "hub")
        generations = [d for d in os.listdir(hub_dir) if d.startswith("gen_")]
        self.assertEqual(generations, [f"gen_{self.STOP:04d}"],
                         msg=f"expected exactly one live generation: "
                             f"{sorted(os.listdir(hub_dir))}")

        written = os.listdir(os.path.join(hub_dir, generations[0]))
        for rank in range(self.HUB_RANKS):
            self.assertIn(f"hub_rank_{rank:04d}.pkl", written)
            self.assertTrue(
                any(f.startswith(f"hub_rank_{rank:04d}_scen_")
                    for f in written),
                msg=f"no model files for rank {rank}: {sorted(written)}")

    def test_resume_matches_the_uninterrupted_run_on_every_rank(self):
        """The iterate itself, rank by rank.

        The key-set comparison runs for every instance and is not a formality:
        it is what would catch a resume that re-attached the W or proximal
        terms and so carries duplicated components. The *values* are compared
        only where the determinism contract promises they can be -- see
        ``test_the_objective_agrees_within_tolerance`` for what a MIP under
        default solver settings gets instead.
        """
        for want_snap, got_snap in zip(self.reference, self.resumed):
            rank = got_snap["cylinder_rank"]
            self.assertEqual(want_snap["scenario_names"],
                             got_snap["scenario_names"],
                             msg=f"rank {rank} owns different scenarios")
            want, got = want_snap["state"], got_snap["state"]
            self.assertEqual(set(want), set(got))
            if not self.BIT_IDENTICAL:
                continue
            worst = max((abs(want[k] - got[k]) for k in want), default=0.0)
            self.assertEqual(
                worst, 0.0,
                msg=f"rank {rank} differs from the uninterrupted run by "
                    f"{worst}; this instance is deterministic, so a "
                    f"resume must land bit-identically")

    def test_the_objective_agrees_within_tolerance(self):
        """For a MIP this is the comparison; for an LP it is a corollary.

        A MIP with alternate optima can be resumed onto a different optimal
        solution than the uninterrupted run found, which moves the iterate
        without meaning anything went wrong. What would mean something went
        wrong is the objective walking away, so that is what is pinned.
        """
        for want_snap, got_snap in zip(self.reference, self.resumed):
            want, got = want_snap["objective"], got_snap["objective"]
            self.assertIsNotNone(want)
            scale = max(1.0, abs(want))
            self.assertLessEqual(
                abs(want - got), self.OBJECTIVE_RTOL * scale,
                msg=f"rank {got_snap['cylinder_rank']}: the resumed run's "
                    f"expected objective is {got}, the uninterrupted run's "
                    f"is {want}")

    def test_bounds_stay_valid_after_a_resume(self):
        """Read from the wheel, where the bounds live.

        The hub's own best_solution_obj_val is never set on a PH hub -- the
        incumbent arrives from a spoke as a bare number -- so comparing it
        checked nothing on any case here.
        """
        checkpointed = _checkpointed_outer_bounds(self.ckpt_dir)
        self.assertEqual(sorted(checkpointed), list(range(self.HUB_RANKS)))
        for snap in self.resumed:
            rank = snap["cylinder_rank"]
            self.assertLessEqual(
                snap["BestOuterBound"], snap["BestInnerBound"],
                msg=f"rank {rank}: the outer bound crossed the incumbent")
            self.assertIsNotNone(checkpointed[rank])
            self.assertGreaterEqual(
                snap["BestOuterBound"], checkpointed[rank],
                msg=f"rank {rank}: the resumed run ended with a weaker "
                    f"outer bound than its checkpoint holds")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestFarmerMultiRankHub(_MultiRankABMixin, unittest.TestCase):
    """The baseline: a two-rank hub on a deterministic LP, evenly split."""

    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "6", "--default-rho", "1")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestFarmerUnevenMultiRankHub(_MultiRankABMixin, unittest.TestCase):
    """Five scenarios over two ranks: 3 and 2.

    Worth its own case because an even split hides anything that assumes the
    ranks are interchangeable -- a per-rank file whose name or contents were
    derived from a scenario *count* rather than from the rank's own scenario
    list would still work when every rank holds the same number.
    """

    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "5", "--default-rho", "1")

    def test_the_split_really_is_uneven(self):
        sizes = sorted(len(snap["scenario_names"]) for snap in self.reference)
        self.assertEqual(sizes, [2, 3])


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestBundlesMultiRankHub(_MultiRankABMixin, unittest.TestCase):
    """Proper bundles across ranks (design section 8.1).

    A proper bundle is a first-class subproblem -- its own entry in
    ``local_scenarios``, its own ``nonant_indices``, its own Pyomo model -- so
    the claim under test is that checkpointing needs no bundle-specific code
    at all. Validating it is also what retires the 2019 "will not work on
    bundles" warning on ``_restore_nonants``.
    """

    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "8", "--scenarios-per-bundle", "2",
                  "--default-rho", "1")

    def test_the_subproblems_really_are_bundles(self):
        names = [n for snap in self.reference for n in snap["scenario_names"]]
        self.assertTrue(all(n.startswith("Bundle") for n in names),
                        msg=f"expected bundles, got {names}")

    def test_bundles_are_checkpointed_by_name(self):
        """Each bundle gets its own model file, named after the bundle.

        Bundle names are not scenario names and carry no usable scenario
        index, which is the case ``sputils.extract_num`` would have got wrong
        (section 10).
        """
        gen_dir = os.path.join(self.ckpt_dir, "hub", f"gen_{self.STOP:04d}")
        written = os.listdir(gen_dir)
        for snap in self.reference:
            rank = snap["cylinder_rank"]
            for bundle in snap["scenario_names"]:
                self.assertIn(f"hub_rank_{rank:04d}_scen_{bundle}.dill",
                              written)


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestCvarMultiRankHub(_MultiRankABMixin, unittest.TestCase):
    """farmer + ``--cvar``: a model mutated after its creator returned.

    Everything else in this file checkpoints a model that its
    ``scenario_creator`` built and nobody touched afterwards. CVaR rewrites
    one: it deactivates the risk-neutral objective, adds an active
    ``WITH_CVAR`` alongside it, and appends the value-at-risk variable eta to
    the root nonants. All three have to come back through the dill, and the
    resume branch has to rebuild ``saved_objectives`` from the *active*
    objective -- picking the deactivated original instead would leave the run
    reporting a risk-neutral number for a risk-averse problem, with nothing
    raising anywhere.

    Originally a phase-1b instance; phase 1b was retired and this is the
    phase that absorbed it.
    """

    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "6", "--default-rho", "1",
                  "--cvar", "--cvar-weight", "0.5", "--cvar-alpha", "0.8")

    def test_the_run_really_is_risk_averse(self):
        """Otherwise everything below is the plain farmer case again."""
        for snap in self.reference:
            for sname, objname in snap["active_objective_names"].items():
                self.assertIn("WITH_CVAR", objname,
                              msg=f"{sname} is not solving the CVaR objective")
        eta_nonants = [k for snap in self.reference for k in snap["state"]
                       if "|x|" in k and "eta" in k]
        self.assertTrue(eta_nonants,
                        msg="eta was not appended to the root nonants")

    def test_the_resume_resolves_to_the_active_objective(self):
        for want_snap, got_snap in zip(self.reference, self.resumed):
            self.assertEqual(want_snap["active_objective_names"],
                             got_snap["active_objective_names"],
                             msg="the resumed run reads a different objective "
                                 "than the uninterrupted one")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestSizesMultiRankHub(_MultiRankABMixin, unittest.TestCase):
    """The MIP target, across ranks.

    Under default solver settings section 7 promises a valid *continuation*,
    not a reproduced trajectory, so the inherited comparison runs to a
    tolerance here. What this case is really for is what a MIP resume can
    lose outright: the warm start that rode back in the dill, and the trivial
    bound, which a resume must carry rather than recompute. It has no
    incumbent to protect -- nothing runs an xhat spoke -- which is
    TestFarmerMultiRankCylinders' job.
    """

    MODULE = _SIZES
    MODEL_ARGS = ("--num-scens", "3", "--default-rho", "1")
    BIT_IDENTICAL = False
    N = 3
    STOP = 1

    def test_the_trivial_bound_is_carried_not_recomputed(self):
        """Resume skips iteration 0, so the bound has to come from the file.

        A resumed run that recomputed it would be computing it from the
        checkpointed (W-laden) iterate, which is not the same quantity -- and
        for a MIP it would also pay a full round of subproblem solves.
        """
        for stopped, resumed in zip(self.stopped, self.resumed):
            self.assertEqual(stopped["trivial_bound"],
                             resumed["trivial_bound"])


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestFarmerMultiRankCylinders(_MultiRankABMixin, unittest.TestCase):
    """The real shape: three cylinders, two ranks each.

    Phase 4 proved a wheel resumes; this proves it still does when each
    cylinder is itself parallel, which is the configuration a cluster run
    actually has. The hub's barriers are now over a strict subset of
    COMM_WORLD, and getting that comm wrong -- COMM_WORLD instead of the
    cylinder's -- deadlocks the job against spokes that never call it.
    """

    NP = 6
    HUB_RANKS = 2
    #: Late enough that the Lagrangian bounds a restarted spoke re-finds are
    #: weaker than the checkpointed one, so losing the hub's outer bound
    #: across the resume fails the bound test.
    N = 8
    STOP = 6
    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "6", "--default-rho", "1")
    SPOKE_ARGS = ("--lagrangian", "--xhatshuffle")

    SPOKE_RANKS = 2

    def test_the_spoke_incumbent_survives_the_stop(self):
        """The best solution lives on the spoke, one file per spoke rank."""
        written = sorted(
            f for f in os.listdir(os.path.join(self.ckpt_dir, "spokes"))
            if f.startswith("spoke_XhatShuffleInnerBound"))
        self.assertEqual(
            [f[-len("rank_0000.pkl"):] for f in written],
            [f"rank_{r:04d}.pkl" for r in range(self.SPOKE_RANKS)],
            msg=f"expected one incumbent file per spoke rank: {written}")

    def test_every_spoke_rank_restores_the_same_incumbent(self):
        markers = _spoke_ranks(os.path.join(self._tmp.name, "B2.json"),
                               "XhatShuffleInnerBound")
        self.assertEqual([m["cylinder_rank"] for m in markers],
                         list(range(self.SPOKE_RANKS)))
        restored = {m["restored_incumbent_obj"] for m in markers}
        self.assertEqual(len(restored), 1,
                         msg=f"the spoke's ranks restored different "
                             f"incumbents: {restored}")
        self.assertIsNotNone(restored.pop(),
                             msg="the spoke restored no incumbent")

    def test_the_incumbent_does_not_regress_across_the_stop(self):
        """Against the hub's inner bound, which is where the spoke's
        incumbent arrives -- the hub's best_solution_obj_val is never set."""
        restored = _spoke_ranks(
            os.path.join(self._tmp.name, "B2.json"),
            "XhatShuffleInnerBound")[0]["restored_incumbent_obj"]
        for snap in self.resumed:
            self.assertLessEqual(
                snap["BestInnerBound"], restored,
                msg=f"rank {snap['cylinder_rank']}: the resumed run reports "
                    f"a worse incumbent than the one it restored")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestTornSpokeIncumbentIsDropped(unittest.TestCase):
    """A two-rank xhat spoke whose ranks' files hold different incumbents.

    What one rank's failed write leaves behind. Restoring it put a different
    best-so-far on each rank: they then published different numbers of
    times, after which the hub rejected everything the spoke sent, or walked
    different scenario orders and hung in the spoke's broadcast.
    """

    NP = 4  # a two-rank hub and a two-rank xhatshuffle spoke
    MODEL_ARGS = ("--num-scens", "6", "--default-rho", "1")
    SPOKE_ARGS = ("--xhatshuffle",)

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt_dir = os.path.join(cls._tmp.name, "ckpt")
        _run_leg(cls._tmp.name, "B1", cls.NP, _FARMER, cls.MODEL_ARGS,
                 cls.SPOKE_ARGS, ("--max-iterations", "2",
                                  "--checkpoint-dir", cls.ckpt_dir))
        spokes = os.path.join(cls.ckpt_dir, "spokes")
        rank1 = [f for f in os.listdir(spokes) if f.endswith("rank_0001.pkl")]
        assert len(rank1) == 1, os.listdir(spokes)
        path = os.path.join(spokes, rank1[0])
        with open(path, "rb") as f:
            state = pickle.load(f)
        # An older, worse incumbent: what rank 1 still holds when its latest
        # write failed and rank 0's succeeded.
        state["best_solution_obj_val"] += 100.0
        state["best_inner_bound"] += 100.0
        with open(path, "wb") as f:
            pickle.dump(state, f)
        cls.result, cls.out_path = _run_leg(
            cls._tmp.name, "B2", cls.NP, _FARMER, cls.MODEL_ARGS,
            cls.SPOKE_ARGS, ("--max-iterations", "3",
                             "--resume-from", cls.ckpt_dir),
            check=False)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_resume_finishes(self):
        self.assertEqual(self.result.returncode, 0,
                         msg=self.result.stdout[-4000:] +
                             self.result.stderr[-4000:])

    def test_it_says_why(self):
        self.assertIn("checkpointed different incumbents",
                      self.result.stdout)

    def test_no_spoke_rank_restores_either_one(self):
        markers = _spoke_ranks(self.out_path, "XhatShuffleInnerBound")
        self.assertEqual(len(markers), 2)
        for m in markers:
            self.assertIsNone(
                m["restored_incumbent_obj"],
                msg=f"spoke rank {m['cylinder_rank']} restored an incumbent "
                    f"the other rank does not hold")

    def test_the_hub_credits_nothing_the_spoke_dropped(self):
        """The resumed hub takes its inner bound from the spokes' files, so
        it must drop what the spoke's ranks drop. Crediting rank 0's file
        made the run report that incumbent and write the worse solution the
        spoke went on to find."""
        for snap in _hub_ranks(self.out_path):
            self.assertEqual(
                snap["first_BestInnerBound"], float("inf"),
                msg=f"hub rank {snap['cylinder_rank']} restored an incumbent "
                    f"the spoke's ranks did not agree on")
        spoke = {m["strata_rank"]: m
                 for m in _spoke_ranks(self.out_path, "XhatShuffleInnerBound")
                 if m["cylinder_rank"] == 0}
        for snap in _hub_ranks(self.out_path):
            credited = spoke.get(snap["last_ib_idx"])
            self.assertIsNotNone(credited, msg="no xhat spoke is credited")
            self.assertAlmostEqual(
                credited["best_solution_obj_val"], snap["BestInnerBound"],
                delta=1e-6 * abs(snap["BestInnerBound"]),
                msg="the hub reports an incumbent the spoke it credits "
                    "does not hold")

    def test_the_hub_still_hears_the_spoke(self):
        """The failure this guards left the hub's inner bound at inf."""
        for snap in _hub_ranks(self.out_path):
            self.assertLess(snap["BestInnerBound"], float("inf"),
                            msg=f"hub rank {snap['cylinder_rank']} never "
                                f"received an incumbent from the spoke")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestStochAdmmMultiRank(_MultiRankABMixin, unittest.TestCase):
    """stoch-ADMM across ranks (design section 8.2).

    The wrapper is where checkpointing has the most to get wrong, and most of
    it fails quietly rather than loudly: wrapped scenario names reach the
    checkpoint's file names, the probability mask and the fixed-at-0 dummy
    vars are built by identity at construction and only their *result* rides
    in the dill, and the wrapper keeps its own references to the models a
    resume replaces. A run with any of those broken still solves and still
    prints numbers.
    """

    NP = 4
    HUB_RANKS = 2
    MODULE = _STOCH_DISTR
    #: Three ADMM subproblems, not two: with two regions every consensus
    #: variable happens to appear in both, so no nonant gets probability zero
    #: and no dummy var is added -- the run would pass every check below
    #: without exercising either. The meta-assertions in those tests are there
    #: to keep that from going unnoticed again.
    MODEL_ARGS = ("--stoch-admm", "--num-stoch-scens", "4",
                  "--num-admm-subproblems", "3", "--default-rho", "10")
    SPOKE_ARGS = ("--xhatxbar",)

    def test_wrapped_names_reach_the_files_intact(self):
        """File discovery enumerates local_scenarios, never a name creator.

        ADMM's names come from the wrapper and collide on their trailing
        digits across subproblems, so anything derived from
        ``sputils.extract_num`` would map two distinct subproblems onto one
        file (section 8.2, item 1).
        """
        gen_dir = os.path.join(self.ckpt_dir, "hub", f"gen_{self.STOP:04d}")
        written = os.listdir(gen_dir)
        seen = set()
        for snap in self.reference:
            rank = snap["cylinder_rank"]
            for sname in snap["scenario_names"]:
                self.assertIn("ADMM", sname)
                fname = f"hub_rank_{rank:04d}_scen_{sname}.dill"
                self.assertIn(fname, written)
                self.assertNotIn(fname, seen)
                seen.add(fname)

    def test_the_probability_mask_survives_the_restore(self):
        """Variable probabilities are consumed once, at construction.

        After that the reloaded model's own ``_mpisppy_data`` masks are
        authoritative (section 8.2, item 3). If the dill lost them, W would be
        applied to nonants that this subproblem does not own and the run would
        converge to the wrong answer without complaining.
        """
        for want_snap, got_snap in zip(self.reference, self.resumed):
            self.assertEqual(want_snap["probability_mask"],
                             got_snap["probability_mask"],
                             msg=f"rank {got_snap['cylinder_rank']}: the "
                                 f"variable-probability mask changed")
            self.assertTrue(
                any(0.0 in values
                    for key, values in want_snap["probability_mask"].items()
                    if "prob0_mask" in key),
                msg="no nonant has zero probability, so this instance does "
                    "not actually exercise the mask")

    def test_the_dummy_variables_are_still_fixed_at_zero(self):
        """The wrapper's inline dummy vars are added after construction.

        They are not nonants, so nothing else in this file looks at them, and
        a resume that relaxed them would quietly drop the consensus structure.
        """
        for want_snap, got_snap in zip(self.reference, self.resumed):
            self.assertEqual(want_snap["fixed_variables"],
                             got_snap["fixed_variables"],
                             msg=f"rank {got_snap['cylinder_rank']}: fixed "
                                 f"variables changed across the resume")
            self.assertTrue(want_snap["fixed_variables"],
                            msg="this instance fixes nothing, so the check "
                                "above proves nothing")

    def test_the_wrapper_does_not_keep_the_replaced_models(self):
        """Otherwise a resumed ADMM run holds two copies of every scenario.

        The wrapper builds every scenario at startup whether or not the run is
        resuming -- it needs them to assemble consensus lists and node names --
        so the freshly built models stay reachable through it after the swap.
        Nothing in the run reads them again; they just occupy memory, which on
        a large MIP is the memory checkpointing exists to save (section 8.2,
        item 2).
        """
        for snap in self.resumed:
            self.assertTrue(
                snap["model_holder_is_current"],
                msg=f"rank {snap['cylinder_rank']}: the ADMM wrapper still "
                    f"points at the models the resume replaced")
        # The same probe on the uninterrupted run, so a True above cannot be
        # the probe failing to find the wrapper at all.
        for snap in self.reference:
            self.assertTrue(snap["model_holder_is_current"],
                            msg="the probe found no ADMM wrapper to check")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestBundledStochAdmmMultiRank(_MultiRankABMixin, unittest.TestCase):
    """stoch-ADMM with proper bundles across ranks (design section 8.2).

    ``--scenarios-per-bundle`` hands scenario creation to ``AdmmBundler``
    instead of ``Stoch_AdmmWrapper``, and the bundler keeps the bundles it
    built for a different reason and under a different attribute, so the
    wrapper test above says nothing about it.
    """

    NP = 4
    HUB_RANKS = 2
    MODULE = _STOCH_DISTR
    #: AdmmBundler requires one bundle per ADMM subproblem, holding every
    #: stochastic scenario, so --scenarios-per-bundle equals --num-stoch-scens.
    MODEL_ARGS = ("--stoch-admm", "--num-stoch-scens", "2",
                  "--num-admm-subproblems", "3", "--scenarios-per-bundle", "2",
                  "--default-rho", "10")
    SPOKE_ARGS = ("--xhatxbar",)

    def test_the_subproblems_really_are_bundles(self):
        names = [n for snap in self.reference for n in snap["scenario_names"]]
        self.assertTrue(all(n.startswith("Bundle") for n in names),
                        msg=f"expected bundles, got {names}")

    def test_the_bundler_does_not_keep_the_replaced_models(self):
        """Otherwise a resumed bundled run holds two copies of every bundle
        (section 8.2, item 2)."""
        for snap in self.resumed:
            self.assertTrue(
                snap["model_holder_is_current"],
                msg=f"rank {snap['cylinder_rank']}: the ADMM bundler still "
                    f"points at the bundles the resume replaced")
        for snap in self.reference:
            self.assertTrue(snap["model_holder_is_current"],
                            msg="the probe found no ADMM bundler to check")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestMultiRankGeometryRefusal(unittest.TestCase):
    """Resuming into a different rank layout is refused, not half-restored.

    Cross-geometry resume is a stated non-goal (section 12), and the failure
    mode without a refusal is the bad kind: each rank would look for the file
    of a rank that no longer exists, or find one holding somebody else's
    scenarios, and either restore nothing or restore the wrong slice.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        _run_leg(self._tmp.name, "write", 2, _FARMER,
                 ("--num-scens", "6", "--default-rho", "1"), (),
                 ("--max-iterations", "2", "--checkpoint-dir", self.ckpt_dir))

    def tearDown(self):
        self._tmp.cleanup()

    def test_resuming_on_fewer_ranks_is_refused(self):
        result, _ = _run_leg(
            self._tmp.name, "resume1", 1, _FARMER,
            ("--num-scens", "6", "--default-rho", "1"), (),
            ("--max-iterations", "4", "--resume-from", self.ckpt_dir),
            check=False)
        self.assertNotEqual(result.returncode, 0,
                            msg="a rank-count change was accepted")
        self.assertIn("rank count", result.stdout + result.stderr)


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestOneRankFailingDoesNotHangTheOthers(unittest.TestCase):
    """A write that fails on one rank must not deadlock the cylinder.

    This is the failure mode the multi-rank protocol exists to prevent, and it
    is invisible to every other test in this file: with the ranks disagreeing
    about whether the write failed, the run does not crash, it *stops* -- one
    rank has returned to the PH loop while the others wait at a barrier it
    will never reach -- and the job burns its wall-clock allocation with no
    error in the log. So the test asserts the run finished at all, which is
    most of the point, and then that the failure was handled the way section 8
    promises: every rank warns, no generation is published for the failed
    write, and the previous checkpoint is still there to resume from.

    The last iteration is the one sabotaged, so the manifest is left naming
    the generation before it. A failure in the middle would be repaired by the
    next successful write and prove less.
    """

    N = 3
    FAIL_AT = 3
    FAIL_RANK = 1

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        cmd = [
            "mpiexec", "-np", "2",
            sys.executable, "-m", "mpi4py", _FAILURE_DRIVER,
            "--fail-on-rank", str(self.FAIL_RANK),
            "--fail-at-generation", str(self.FAIL_AT),
            "--module-name", _FARMER, "--num-scens", "6",
            "--default-rho", "1", "--solver-name", solver_name,
            "--max-iterations", str(self.N),
            "--intra-hub-conv-thresh", "-1",
            "--rel-gap", "0.0", "--abs-gap", "0.0",
            "--checkpoint-dir", self.ckpt_dir,
        ]
        # Generous, but it is a timeout on a farmer LP that finishes in under a
        # second when it finishes at all: what this is really measuring is
        # whether the job returns.
        self.result = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=600, check=False,
                                     env=subprocess_env())

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_run_finishes(self):
        self.assertEqual(
            self.result.returncode, 0,
            msg=f"the run did not survive a one-rank write failure:\n"
                f"{self.result.stdout[-4000:]}\n{self.result.stderr[-4000:]}")
        self.assertIn("Reached user-specified limit", self.result.stdout,
                      msg="the run did not reach its iteration limit")

    def test_both_ranks_report_the_failure(self):
        """Every rank raises, so every rank warns -- and says where.

        The rank whose own write succeeded has no exception to describe, so it
        names the rank that does. Without that, a multi-rank failure prints
        one real diagnosis and n-1 misleading ones.
        """
        warnings = [line for line in self.result.stdout.splitlines()
                    if "WARNING: checkpoint write failed" in line]
        self.assertEqual(len(warnings), 2,
                         msg=f"expected one warning per rank: {warnings}")
        for rank in (0, self.FAIL_RANK):
            self.assertTrue(
                any(f"on rank {rank}" in line for line in warnings),
                msg=f"rank {rank} did not report the failure: {warnings}")
        # Rank 0's write succeeded, so its message must point at the rank that
        # holds the cause rather than implying its own write went wrong.
        self.assertIn("could not write its part of the checkpoint",
                      self.result.stdout)
        # And the failing rank's own message must carry the real cause, which
        # is the thing a rank-0-only warning would have thrown away.
        self.assertIn("No space left on device", self.result.stdout)

    def test_the_failed_generation_is_not_published(self):
        """All-or-nothing: rank 0's half of it is discarded too."""
        manifest = _published_generation(self.ckpt_dir)
        self.assertEqual(manifest["generation"], self.FAIL_AT - 1)

        hub_dir = os.path.join(self.ckpt_dir, "hub")
        self.assertEqual(
            sorted(d for d in os.listdir(hub_dir) if d.startswith("gen_")),
            [f"gen_{self.FAIL_AT - 1:04d}"],
            msg=f"the abandoned generation was left behind: "
                f"{sorted(os.listdir(hub_dir))}")

    def test_the_previous_checkpoint_is_still_resumable(self):
        _, out_path = _run_leg(
            self._tmp.name, "resume", 2, _FARMER,
            ("--num-scens", "6", "--default-rho", "1"), (),
            ("--max-iterations", str(self.N + 1),
             "--resume-from", self.ckpt_dir))
        for snap in _hub_ranks(out_path):
            self.assertTrue(snap["resumed"])
            self.assertEqual(snap["resume_iteration"], self.FAIL_AT - 1)


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestDeadlineOnOneRankDoesNotHangTheOthers(unittest.TestCase):
    """``--checkpoint-before-seconds`` is the one trigger that is rank-local.

    Every other checkpoint point is a pure function of the iteration number, so
    the ranks arrive at the write together without being asked. Elapsed wall
    clock is not, and the ranks of a cylinder do not share a clock: a rank that
    believed its own and started writing would wait in the write's barrier for
    ranks that went on with the iteration, and the job would burn its
    allocation with nothing in the log.

    So the driver puts one rank a year past the deadline and leaves the other
    where it was. The run either agrees -- through ``allreduce_or`` -- and
    writes on both ranks, or it hangs; there is no third outcome, which is what
    makes the timeout below an assertion rather than a guess.

    The cadence is set past the iteration count so that nothing but the
    deadline (and the final iteration, which is always written) can produce a
    write, and the deadline is set to an hour that a farmer LP has no other way
    of reaching.
    """

    N = 4
    SKEW_AT = 2
    SKEW_RANK = 1
    DEADLINE = 3600.0

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        cmd = [
            "mpiexec", "-np", "2",
            sys.executable, "-m", "mpi4py", _DEADLINE_DRIVER,
            "--skew-on-rank", str(self.SKEW_RANK),
            "--skew-at-generation", str(self.SKEW_AT),
            "--module-name", _FARMER, "--num-scens", "6",
            "--default-rho", "1", "--solver-name", solver_name,
            "--max-iterations", str(self.N),
            "--intra-hub-conv-thresh", "-1",
            "--rel-gap", "0.0", "--abs-gap", "0.0",
            "--checkpoint-dir", self.ckpt_dir,
            "--checkpoint-every-iterations", "100",
            "--checkpoint-before-seconds", str(self.DEADLINE),
        ]
        # Generous, but it is a timeout on a farmer LP that finishes in under a
        # second when it finishes at all: what this is really measuring is
        # whether the job returns.
        self.result = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=600, check=False,
                                     env=subprocess_env())

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_run_finishes(self):
        self.assertEqual(
            self.result.returncode, 0,
            msg=f"the run did not survive a one-rank deadline:\n"
                f"{self.result.stdout[-4000:]}\n{self.result.stderr[-4000:]}")
        self.assertIn("Reached user-specified limit", self.result.stdout,
                      msg="the run did not reach its iteration limit")

    def test_the_deadline_wrote_the_skewed_generation(self):
        """Off cadence and not the last iteration, so the deadline is the only
        thing that could have written it -- and one rank's clock was enough."""
        self.assertIn(f"Checkpoint written at iteration {self.SKEW_AT}",
                      self.result.stdout)
        self.assertIn("--checkpoint-before-seconds", self.result.stdout)

    def test_every_rank_wrote_its_slice(self):
        """A generation spans the ranks, so a write that only the skewed rank
        joined would leave a generation that cannot be resumed."""
        gen_dir = os.path.join(self.ckpt_dir, "hub",
                               f"gen_{self.SKEW_AT:04d}")
        # gen_0002 is retired by the final iteration's write, so the run's own
        # log is what says both ranks were in it; what remains checkable here
        # is that the generation that replaced it is whole.
        final_dir = os.path.join(self.ckpt_dir, "hub", f"gen_{self.N:04d}")
        self.assertFalse(os.path.isdir(gen_dir))
        leaves = sorted(f for f in os.listdir(final_dir)
                        if f.startswith("hub_rank_") and f.endswith(".pkl"))
        self.assertEqual(leaves,
                         ["hub_rank_0000.pkl", "hub_rank_0001.pkl"])

    def test_it_fires_once_and_not_on_every_later_iteration(self):
        """The skewed rank stays past the deadline for the rest of the run.
        Without the latch, every iteration after it would write."""
        written = [line for line in self.result.stdout.splitlines()
                   if "Checkpoint written at iteration" in line]
        self.assertEqual(len(written), 2, msg=f"expected the deadline write "
                                              f"and the final one: {written}")

    def test_the_deadline_checkpoint_is_resumable(self):
        """Resume from the deadline's own generation, not the one that
        replaced it: rerun with the iteration limit at the skew point so the
        final-iteration rule cannot write anything later."""
        cmd = [
            "mpiexec", "-np", "2",
            sys.executable, "-m", "mpi4py", _DEADLINE_DRIVER,
            "--skew-on-rank", str(self.SKEW_RANK),
            "--skew-at-generation", str(self.SKEW_AT),
            "--module-name", _FARMER, "--num-scens", "6",
            "--default-rho", "1", "--solver-name", solver_name,
            "--max-iterations", str(self.SKEW_AT),
            "--intra-hub-conv-thresh", "-1",
            "--rel-gap", "0.0", "--abs-gap", "0.0",
            "--checkpoint-dir", self.ckpt_dir,
            "--checkpoint-every-iterations", "100",
            "--checkpoint-before-seconds", str(self.DEADLINE),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       check=True, env=subprocess_env())
        self.assertEqual(_published_generation(self.ckpt_dir)["generation"],
                         self.SKEW_AT)

        _, out_path = _run_leg(
            self._tmp.name, "resume", 2, _FARMER,
            ("--num-scens", "6", "--default-rho", "1"), (),
            ("--max-iterations", str(self.N),
             "--resume-from", self.ckpt_dir))
        for snap in _hub_ranks(out_path):
            self.assertTrue(snap["resumed"])
            self.assertEqual(snap["resume_iteration"], self.SKEW_AT)


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestMultiRankSpokeCursorAgreement(unittest.TestCase):
    """A multi-rank xhat spoke resumes onto one cursor, not one per rank.

    The ranks of an xhatshuffle spoke explore together: they pick the same
    scenario and ``_try_one`` broadcasts its nonants from the rank that owns
    it. Each rank writes its own checkpoint file at the bottom of its own
    pass, though, so a stop can land between two of those writes and leave
    files whose cursors disagree -- and a rank that never found an incumbent
    writes no file at all. Resuming each rank onto whatever its own file says
    makes the ranks pick different scenarios and broadcast from different
    roots, and the objective that reaches the hub is then a blend of several
    scenarios' solutions, reported as an ordinary feasible inner bound with
    no error and no warning.

    Both tests manufacture the disagreement by editing what the stopped leg
    wrote. Racing the two ranks' writes would produce it only sometimes,
    which is no way to guard against it.
    """

    NP = 6
    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "6", "--default-rho", "1")
    SPOKE_ARGS = ("--lagrangian", "--xhatshuffle")
    STOP = 2
    RESUME_FOR = 2
    SPOKE = "XhatShuffleInnerBound"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        _run_leg(self._tmp.name, "B1", self.NP, self.MODULE, self.MODEL_ARGS,
                 self.SPOKE_ARGS,
                 ("--max-iterations", str(self.STOP),
                  "--checkpoint-dir", self.ckpt_dir))

    def tearDown(self):
        self._tmp.cleanup()

    def _spoke_files(self):
        """This spoke's checkpoint files, one per rank, ordered by rank."""
        spokes_dir = os.path.join(self.ckpt_dir, "spokes")
        names = sorted(f for f in os.listdir(spokes_dir)
                       if f.startswith(f"spoke_{self.SPOKE}"))
        self.assertGreater(
            len(names), 1,
            msg=f"expected one file per spoke rank, got {names}")
        return [os.path.join(spokes_dir, n) for n in names]

    def _resume(self):
        result, out_path = _run_leg(
            self._tmp.name, "B2", self.NP, self.MODULE, self.MODEL_ARGS,
            self.SPOKE_ARGS,
            ("--max-iterations", str(self.RESUME_FOR),
             "--resume-from", self.ckpt_dir))
        return result, _spoke_ranks(out_path, self.SPOKE)

    def test_the_ranks_adopt_one_cursor(self):
        paths = self._spoke_files()
        with open(paths[-1], "rb") as f:
            state = pickle.load(f)
        self.assertIsNotNone(
            state["loop_state"],
            msg="the stopped leg checkpointed no cursor, so this test would "
                "pass without exercising anything")
        # Wind the last rank's file back a pass: what a stop that landed
        # between the two ranks' writes leaves behind.
        state["loop_state"]["xh_iter"] = int(state["loop_state"]["xh_iter"]) - 1
        cursor = state["loop_state"]["cursor"]
        cursor["cycle_idx"] = max(0, int(cursor["cycle_idx"]) - 1)
        with open(paths[-1], "wb") as f:
            pickle.dump(state, f)

        _, markers = self._resume()
        self.assertGreater(len(markers), 1,
                           msg=f"expected a marker per spoke rank: {markers}")
        adopted = [m["applied_loop_state"] for m in markers]
        for marker in markers:
            self.assertIsNotNone(
                marker["applied_loop_state"],
                msg="a spoke rank adopted no cursor at all")
        distinct = {json.dumps(a, sort_keys=True) for a in adopted}
        self.assertEqual(
            len(distinct), 1,
            msg=f"the spoke's ranks resumed onto {len(distinct)} different "
                f"cursors: {adopted}")

    def test_ranks_holding_different_incumbents_stop_the_whole_restore(self):
        """Half of one xhat beside half of another is not a solution.

        The cached values cannot be agreed by broadcast -- each rank owns
        different scenarios -- so what is agreed is whether they all came
        from the same pass. The objective of the cached solution says so:
        every rank reads it out of the same reduction.
        """
        paths = self._spoke_files()
        with open(paths[-1], "rb") as f:
            state = pickle.load(f)
        self.assertIsNotNone(
            state["best_solution_obj_val"],
            msg="the stopped leg checkpointed no incumbent, so this test "
                "would pass without exercising anything")
        # An incumbent from a different pass: what a stop landing between the
        # two ranks' writes leaves when one of them has just improved.
        state["best_solution_obj_val"] = \
            float(state["best_solution_obj_val"]) - 1.0
        with open(paths[-1], "wb") as f:
            pickle.dump(state, f)

        result, markers = self._resume()
        for marker in markers:
            self.assertIsNone(
                marker["restored_incumbent_obj"],
                msg="a spoke rank restored values from a pass that another "
                    "rank of the same spoke did not checkpoint")
        self.assertIn("checkpointed different incumbents", result.stdout,
                      msg="the run declined to restore without saying so")

    def test_a_file_that_does_not_match_stops_every_rank(self):
        """A load that refuses on some ranks and not others.

        The refusal is per rank -- the file is named after the rank and
        checked against the scenarios that rank owns -- and the agreement
        that follows is collective, so a rank-local raise leaves the others
        waiting for a rank that has gone. The plainest way in is a resume
        with a different rank count; this manufactures the same split
        directly, which does not depend on how ranks divide scenarios.
        """
        paths = self._spoke_files()
        os.remove(paths[0])
        with open(paths[-1], "rb") as f:
            state = pickle.load(f)
        state["format_version"] = 0
        with open(paths[-1], "wb") as f:
            pickle.dump(state, f)

        result, _ = _run_leg(
            self._tmp.name, "B2", self.NP, self.MODULE, self.MODEL_ARGS,
            self.SPOKE_ARGS,
            ("--max-iterations", str(self.RESUME_FOR),
             "--resume-from", self.ckpt_dir),
            check=False)
        self.assertNotEqual(result.returncode, 0,
                            msg="a checkpoint that does not match this run "
                                "was accepted")
        self.assertIn(
            "could not read their checkpoint", result.stdout + result.stderr,
            msg="the run died on one rank's traceback without the agreement "
                "saying how many ranks could not read theirs")

    def test_a_rank_without_a_file_stops_the_whole_restore(self):
        """Half an incumbent is not a solution the study ever found."""
        paths = self._spoke_files()
        os.remove(paths[-1])

        result, markers = self._resume()
        for marker in markers:
            self.assertIsNone(
                marker["restored_incumbent_obj"],
                msg="a spoke rank restored an incumbent although another "
                    "rank of the same spoke had no file to restore from")
        self.assertIn("ranks of this spoke have a checkpointed incumbent",
                      result.stdout,
                      msg="the run declined to restore without saying so")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestScenarioDependentRhoDualResume(unittest.TestCase):
    """A dual cylinder resumes when rho differs by scenario.

    PH keeps sum_s p_s W_s = 0 only with the same rho in every scenario. The
    restore used to require exactly that, so this run wrote its checkpoint
    at exit 0 and its resume died at startup, blaming the file.
    """

    NP = 4
    MODULE = "mpisppy.tests.examples.farmer_scenario_rho"
    MODEL_ARGS = ("--num-scens", "6", "--default-rho", "1")
    SPOKE_ARGS = ("--lagrangian", "--xhatshuffle", "--relaxed-ph",
                  "--ph-primal-hub")
    CYLINDER = "RelaxedPHSpoke"

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt_dir = os.path.join(cls._tmp.name, "ckpt")
        _run_leg(cls._tmp.name, "B1", cls.NP, cls.MODULE, cls.MODEL_ARGS,
                 cls.SPOKE_ARGS, ("--max-iterations", "4",
                                  "--checkpoint-dir", cls.ckpt_dir))
        cls.result, cls.out_path = _run_leg(
            cls._tmp.name, "B2", cls.NP, cls.MODULE, cls.MODEL_ARGS,
            cls.SPOKE_ARGS, ("--max-iterations", "2",
                             "--resume-from", cls.ckpt_dir),
            check=False)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_weights_really_do_not_sum_to_zero(self):
        """Otherwise this is an ordinary resume and tests nothing new."""
        spokes = os.path.join(self.ckpt_dir, "spokes")
        (name,) = [f for f in os.listdir(spokes)
                   if f.startswith(f"spoke_{self.CYLINDER}")]
        with open(os.path.join(spokes, name), "rb") as f:
            wbar = pickle.load(f)["Wbar"]
        self.assertGreater(max(abs(v) for vals in wbar.values()
                               for v in vals), 1e-3)

    def test_the_resume_finishes(self):
        self.assertEqual(self.result.returncode, 0,
                         msg=self.result.stdout[-4000:] +
                             self.result.stderr[-4000:])

    def test_the_dual_cylinder_restored_its_weights(self):
        (marker,) = _spoke_ranks(self.out_path, self.CYLINDER)
        self.assertIsNotNone(marker["restored_dual_generation"])


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestMultiRankDualWeightAgreement(unittest.TestCase):
    """A multi-rank dual cylinder restores one iteration's W, not one each.

    W is per scenario and so per rank, and there is nothing to broadcast --
    but the iteration it belongs to is the cylinder's. Ranks that restore W
    from different iterations hand ``Compute_Xbar`` an allreduce over values
    from two points of the run, and under ``--ph-primal-hub`` that blended W
    is what the hub's W is built from. Like the xhat case, it costs no error
    and no warning.

    Each rank writes at the bottom of its own iteration and a failed write
    warns and carries on, so the files really can disagree; the tests
    manufacture that by editing what the stopped leg wrote.
    """

    NP = 4
    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "6", "--default-rho", "1")
    SPOKE_ARGS = ("--relaxed-ph",)
    STOP = 2
    RESUME_FOR = 2
    CYLINDER = "RelaxedPHSpoke"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # Removed here rather than in tearDown, which unittest does not call
        # when setUp raises -- and the leg below raises on a failed run.
        self.addCleanup(self._tmp.cleanup)
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        _run_leg(self._tmp.name, "B1", self.NP, self.MODULE, self.MODEL_ARGS,
                 self.SPOKE_ARGS,
                 ("--max-iterations", str(self.STOP),
                  "--checkpoint-dir", self.ckpt_dir))

    def _dual_files(self):
        """This cylinder's checkpoint files, one per rank, ordered by rank."""
        spokes_dir = os.path.join(self.ckpt_dir, "spokes")
        names = sorted(f for f in os.listdir(spokes_dir)
                       if f.startswith(f"spoke_{self.CYLINDER}"))
        self.assertGreater(
            len(names), 1,
            msg=f"expected one file per cylinder rank, got {names}")
        return [os.path.join(spokes_dir, n) for n in names]

    def _resume(self):
        result, out_path = _run_leg(
            self._tmp.name, "B2", self.NP, self.MODULE, self.MODEL_ARGS,
            self.SPOKE_ARGS,
            ("--max-iterations", str(self.RESUME_FOR),
             "--resume-from", self.ckpt_dir))
        markers = _spoke_ranks(out_path, self.CYLINDER)
        self.assertGreater(
            len(markers), 1,
            msg=f"expected a marker per cylinder rank: {markers}")
        return result, markers

    def test_the_ranks_restore_the_same_iteration(self):
        """The undisturbed case, so the tests below cannot pass vacuously."""
        _, markers = self._resume()
        generations = {m["restored_dual_generation"] for m in markers}
        self.assertEqual(
            len(generations), 1,
            msg=f"the cylinder's ranks restored W from {len(generations)} "
                f"different iterations: {generations}")
        self.assertNotIn(None, generations,
                         msg="no rank restored any dual weights at all")

    def test_ranks_holding_different_iterations_stop_the_whole_restore(self):
        paths = self._dual_files()
        with open(paths[-1], "rb") as f:
            state = pickle.load(f)
        self.assertIsNotNone(state["generation"])
        # W from the iteration before: what a stop landing between the two
        # ranks' writes leaves behind.
        state["generation"] = int(state["generation"]) - 1
        with open(paths[-1], "wb") as f:
            pickle.dump(state, f)

        result, markers = self._resume()
        for marker in markers:
            self.assertIsNone(
                marker["restored_dual_generation"],
                msg="a rank restored W from an iteration another rank of the "
                    "same cylinder did not checkpoint")
        self.assertIn("dual weights from different iterations",
                      result.stdout,
                      msg="the run declined to restore without saying so")

    def test_a_rank_without_a_file_stops_the_whole_restore(self):
        paths = self._dual_files()
        os.remove(paths[-1])

        result, markers = self._resume()
        for marker in markers:
            self.assertIsNone(
                marker["restored_dual_generation"],
                msg="a rank restored W although another rank of the same "
                    "cylinder had no file to restore from")
        self.assertIn("ranks of this cylinder have checkpointed dual weights",
                      result.stdout,
                      msg="the run declined to restore without saying so")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestNoRankLocalRaiseOnTheCheckpointPaths(unittest.TestCase):
    """A failure in setting a checkpoint up or restoring one ends the job.

    Every step of both paths is per rank by nature: it reads the file named
    after this rank, checks it against the scenarios this rank owns, or writes
    from the node this rank is on. So each of them can fail on one rank and
    pass on the others -- and what follows is collective. A rank that raises
    on its own leaves the rest of its cylinder waiting in that collective for
    a rank that has already gone. How that ends is up to the launcher and
    neither ending is the one wanted: under ``python -m mpi4py`` the whole
    job is aborted holding one rank's traceback, taking down the cylinders
    that were mid-write with it, and under a plain ``python`` launch the job
    sits in the collective until its wall-clock limit with nothing in the log.

    Written as a sweep rather than as one case per bug on purpose. Each of
    these paths has had this defect, and fixing the instance a review happened
    to reach moved it rather than removed it, three rounds running; the
    property is that *no* step on them raises alone, so the test asks it of
    every agreement on them. ``multirank_agreement_driver.STEPS`` names one
    step inside each, and
    ``TestEveryCheckpointStepOnThosePathsIsAgreed`` is what keeps a step
    added later from quietly falling outside the sweep.

    Two things are asserted of each run, and the first is most of the point:
    the job **ends** (a hang shows up here as the subprocess timeout), and it
    ends saying how many ranks could not do the step -- the diagnosis a
    rank-local raise loses, since it kills the job holding one rank's
    traceback while the agreement's own explanation never runs.
    """

    #: Three cylinders of two ranks each, which is what it takes for one run
    #: to reach every path: the hub's, an xhat spoke's and a dual cylinder's.
    #: No lagrangian beside them -- it checkpoints nothing, and a bound spoke
    #: cannot choose between two cylinders publishing the nonants it reads.
    NP = 6
    RANKS_PER_CYLINDER = 2
    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "6", "--default-rho", "1")
    SPOKE_ARGS = ("--xhatshuffle", "--relaxed-ph")
    #: The rank of its own cylinder that the injected step fails on. Any rank
    #: but 0: rank 0 is the one that publishes and prints, so a failure there
    #: is the case least likely to go unnoticed.
    FAIL_ON = 1
    #: A farmer LP leg is seconds; this is a timeout on whether the job comes
    #: back at all, which is what a rank-local raise takes away.
    TIMEOUT = 300

    @classmethod
    def setUpClass(cls):
        """One stopped leg, so the restore steps have a checkpoint to read."""
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt_dir = os.path.join(cls._tmp.name, "ckpt")
        _run_leg(cls._tmp.name, "B1", cls.NP, cls.MODULE, cls.MODEL_ARGS,
                 cls.SPOKE_ARGS,
                 ("--max-iterations", "2", "--checkpoint-dir", cls.ckpt_dir))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run_with(self, step):
        """Resume with ``step`` failing on one rank of every cylinder that
        runs it. Writing is on as well as reading, so the setup path is live
        too -- into its own directory, so the checkpoint the other steps read
        is left as the stopped leg wrote it."""
        cmd = [
            "mpiexec", "-np", str(self.NP),
            sys.executable, "-m", "mpi4py", _AGREEMENT_DRIVER,
            "--break-step", step,
            "--on-cylinder-rank", str(self.FAIL_ON),
            "--module-name", self.MODULE, *self.MODEL_ARGS,
            "--solver-name", solver_name, *self.SPOKE_ARGS,
            "--intra-hub-conv-thresh", "-1",
            "--rel-gap", "0.0", "--abs-gap", "0.0",
            "--max-iterations", "3",
            "--resume-from", self.ckpt_dir,
            "--checkpoint-dir", os.path.join(self._tmp.name, f"w_{step}"),
        ]
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self.TIMEOUT, check=False,
                                  env=subprocess_env())
        except subprocess.TimeoutExpired:
            self.fail(
                f"the job never came back after {step} failed on one rank: "
                f"the ranks that did not fail are still waiting in a "
                f"collective for the rank that raised, which is the hang "
                f"this agreement exists to prevent")

    def test_every_agreement_on_these_paths_ends_the_job(self):
        for step in agreement_driver.STEPS:
            with self.subTest(step=step):
                result = self._run_with(step)
                output = result.stdout + result.stderr
                self.assertIn(
                    agreement_driver.INJECTED, output,
                    msg=f"{step} was never reached, so this ran the "
                        f"unsabotaged job and proved nothing")
                self.assertNotEqual(
                    result.returncode, 0,
                    msg=f"a rank that could not {step} was ignored and the "
                        f"run reported success:\n{output[-4000:]}")
                self.assertIn(
                    f"1 of {self.RANKS_PER_CYLINDER} ranks of this cylinder "
                    f"could not",
                    output,
                    msg=f"the job died on one rank's traceback rather than on "
                        f"the agreement, which is what leaves the other ranks "
                        f"in a collective:\n{output[-4000:]}")


class TestEveryCheckpointStepOnThosePathsIsAgreed(unittest.TestCase):
    """Nothing local happens on the setup or restore paths outside an
    agreement.

    The companion to the sweep above, and the half that survives a refactor.
    That one proves the agreements that exist do their job; this one proves
    there is no step beside them -- a new call, a raise, or a few lines of
    file handling written inline, as the directory write probe once was --
    that a rank can fail on its own.

    Reads the source rather than running anything, because what it is asking
    about is the shape of these functions: which of the steps they take are
    inside ``run_agreed`` and which are not. It reads the source of what they
    *call*, too. A helper called from one of these paths outside an agreement
    is doing this rank's own work exactly as inline code is -- the write
    probe could be moved into a method of its own and this would have gone
    quiet -- so the closure of bare calls is what is checked, not the four
    functions alone.

    Three things it deliberately does not do by name-matching, because each
    was demonstrated evading a version that did:

    * calls are resolved to the *object* they name, so the same call written
      ``from ... import probe_directory_is_writable as _p; _p(...)``, or
      fully dotted, or through a renamed module alias, is the same call here;
    * a call is inside an agreement when that call *node* is inside one, not
      when some other call of the same name is;
    * doing a rank's own file handling is decided by the module the callable
      comes from, so ``os.mkdir`` cannot pass where ``os.makedirs`` fails.
    """

    #: The entry points: the functions on these paths that run *outside* an
    #: agreement. A function called through ``run_agreed`` is not here -- it
    #: is already inside one -- but a function called beside one is reached
    #: through the closure below.
    PATHS = (
        ("Checkpointer.__init__", Checkpointer.__init__),
        ("Checkpointer.pre_iter0", Checkpointer.pre_iter0),
        ("Checkpointer._restore_incumbent", Checkpointer._restore_incumbent),
        ("PHBase._restore_from_checkpoint_if_resuming",
         PHBase._restore_from_checkpoint_if_resuming),
        ("PHBase._restore_extension_state_if_resuming",
         PHBase._restore_extension_state_if_resuming),
        ("Checkpointer.post_iter0", Checkpointer.post_iter0),
        ("XhatInnerBoundBase._restore_extension_state_if_resuming",
         XhatInnerBoundBase._restore_extension_state_if_resuming),
        ("XhatShuffleInnerBound._restore_loop_state_if_resuming",
         XhatShuffleInnerBound._restore_loop_state_if_resuming),
        # Listed as well as reached: the call to it goes through
        # self.scenario_cycler, an attribute whose class the closure cannot
        # resolve from the source, and following such calls by name instead
        # drags in every mpisppy method sharing the name.
        ("ScenarioCycler.restore_state", ScenarioCycler.restore_state),
    )

    #: The accessors that hand a spoke what a resume read for it. Together
    #: with every ``restored_*`` attribute the Checkpointer assigns, these are
    #: the names whose readers are on a restore path and so belong in PATHS;
    #: test_every_reader_of_the_restored_state_is_a_path checks that.
    RESTORED_STATE_ACCESSOR_NAMES = frozenset({"_checkpointed_loop_state"})

    #: Readers that only return what they read to their caller. They are not
    #: paths themselves; their callers are, by the rule above.
    RESTORED_STATE_ACCESSORS = frozenset({
        "XhatInnerBoundBase._checkpointed_loop_state",
    })

    #: Every agreement reached from those paths, by the function it is
    #: written in and in the order it appears there, named by the step
    #: ``multirank_agreement_driver`` breaks to exercise it. Declared rather
    #: than inferred: two cylinders take the same step through the same
    #: function name, so which agreement a step exercises is not something
    #: the name can say. An agreement added without a step here fails this,
    #: and so does a step the driver stops naming.
    AGREEMENTS = {
        "Checkpointer.__init__": ("probe_directory_is_writable",),
        "Checkpointer._restore_incumbent": ("load_spoke_incumbent",
                                            "restore_spoke_incumbent"),
        "PHBase._restore_from_checkpoint_if_resuming": ("load_checkpoint",),
        "PHBase._restore_extension_state_if_resuming":
            ("restore_extension_state",),
        "Checkpointer.post_iter0": ("load_dual_spoke_state",
                                    "restore_dual_spoke_state",
                                    "require_restored_duals_match_their_file"),
        "Checkpointer._report_unclaimed_spoke_files":
            ("unclaimed_spoke_files",),
        "XhatInnerBoundBase._restore_extension_state_if_resuming":
            ("restore_extension_state_on_a_spoke",),
    }

    #: The raises reached from these paths that every rank of a cylinder
    #: makes or none does, with how many there are and what makes them the
    #: same on every rank. A raise that reads a file, a model, or anything
    #: else only this rank can see does not belong here -- it belongs inside
    #: an agreement, which is the whole subject of this class.
    RANK_INDEPENDENT_RAISES = {
        "Checkpointer.__init__": (
            4, "the options and the cylinder's class: an --checkpoint-every "
               "below 1, a --checkpoint-before-seconds that is not positive, "
               "neither writing nor resuming, and spoke mode on something "
               "that is not an Xhat_Eval. Every rank of the wheel is given "
               "the same command line. The backend refusal is a fifth of "
               "these and now lives in require_implemented_backend, below."),
        "Checkpointer._spoke_identity": (
            1, "the cylinder has no spcomm, which is a property of how the "
               "wheel was built rather than of anything this rank read."),
        "PHBase._restore_from_checkpoint_if_resuming": (
            1, "this hub is not a PH, which is the same object on every rank "
               "of the cylinder."),
        "require_implemented_backend": (
            1, "--checkpoint-backend names a backend that is designed but "
               "not built. The value comes from the command line the wheel "
               "hands every rank, and nothing else is consulted."),
        "ScenarioCycler._fill_nodescen_dict": (
            1, "no scenario fits some nonleaf node. What it walks is the "
               "cursor every rank adopted (agree_spoke_restore broadcasts "
               "rank 0's) and the scenario order, which is drawn from one "
               "fixed seed, so every rank walks the same list."),
    }

    #: Checkpointing calls that agree across the cylinder themselves, so they
    #: may be called from a path directly. Each one is collective inside.
    AGREE_THEMSELVES = frozenset({
        "run_agreed",
        "probe_model_is_dillable",
        "agree_spoke_restore",
        "agree_dual_spoke_restore",
    })

    #: And calls that need no agreement because there is nothing in them for
    #: one rank to fail at: they read what is already in memory -- the
    #: command line every rank was given, say -- and return an answer, so
    #: they arrive at the same one on every rank or raise on all of them.
    #: Nothing that touches a file or a model belongs here.
    CANNOT_FAIL_ON_ONE_RANK = frozenset({
        "converger_state_is_carried",
        "require_implemented_backend",
        # Walks the extension object's attributes; reads no file or model.
        "_extension_objects",
    })

    #: Functions, by qualified name, that catch every failure of the work
    #: they do and carry on, and that talk to no other rank. A rank whose
    #: step fails there continues exactly as one whose step succeeded, so it
    #: cannot leave the others waiting, and the closure below does not
    #: follow into them. The claim is checked, not trusted:
    #: test_catch_and_continue_members_do reads each one's source.
    CATCH_AND_CONTINUE = frozenset({
        "Checkpointer._spoke_checkpoint",
    })

    #: Packages imported before the collective check below, so that an
    #: attribute call the source reader cannot resolve -- a spoke's
    #: ``checkpoint_loop_state()``, an extension's ``checkpoint_state()``,
    #: xhatshuffle's ``cycler.checkpoint_state()`` -- can be followed into
    #: every function or method of that name in these packages and in any
    #: mpisppy module already loaded. Naming the classes one at a time missed
    #: the next object down each time it was tried. These are the packages
    #: that define spokes, extensions and convergers; mpisppy.utils.w_utils
    #: is listed because its extensions are imported only lazily. The rest
    #: of mpisppy.utils is not walked: a demo in it raises at import.
    NAME_FALLBACK_PACKAGES = (
        "mpisppy.cylinders", "mpisppy.extensions", "mpisppy.convergers",
        "mpisppy.utils.w_utils",
    )

    #: Method names that are collectives on an MPI communicator, for the
    #: check that a CATCH_AND_CONTINUE member reaches none.
    COLLECTIVE_METHODS = frozenset({
        "Barrier", "barrier", "bcast", "Bcast", "allreduce", "Allreduce",
        "reduce", "Reduce", "allgather", "Allgather", "Allgatherv", "gather",
        "Gather", "Gatherv", "scatter", "Scatter", "alltoall", "Alltoall",
        "allreduce_or",
    })

    #: Modules whose callables do this rank's own work: they touch the file
    #: system or turn bytes into objects, and either can fail on one rank
    #: alone. Named as modules rather than as functions so that the next
    #: spelling of the same idea is caught too -- the write probe was four
    #: lines of ``os`` and ``open`` in a constructor, and it hung two real
    #: jobs before it was found.
    LOCAL_WORK_MODULES = frozenset({
        "os", "posix", "nt", "shutil", "pickle", "dill", "json", "io",
        "_io", "pathlib", "tempfile", "glob"
    })

    #: And the same idea reached through an object, where there is no module
    #: to resolve: ``some_path.read_text()`` names nothing this can look up.
    LOCAL_WORK_METHODS = frozenset({
        "open", "read_text", "write_text", "read_bytes", "write_bytes",
        "mkdir", "unlink", "rmdir", "touch", "iterdir", "samefile",
    })

    # ---- reading the source -------------------------------------------

    @classmethod
    def _owner_classes(cls):
        """The classes the paths are methods of, for resolving ``self.x``."""
        classes = []
        for _, func in cls.PATHS:
            owner = func.__globals__.get(func.__qualname__.split(".")[0])
            if owner is not None and owner not in classes:
                classes.append(owner)
        return classes

    @staticmethod
    def _namespace(func, tree):
        """What names mean inside ``func``: its module's, plus its own.

        A function that imports what it needs where it needs it --
        ``from mpisppy.utils.checkpointing import probe_directory_is_writable
        as _p`` -- binds a name that is in no module namespace, and reading
        only the module's left that call invisible.
        """
        namespace = dict(func.__globals__)
        for node in ast.walk(tree):
            try:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        namespace[alias.asname or top] = importlib.import_module(
                            alias.name if alias.asname else top)
                elif isinstance(node, ast.ImportFrom) and node.module \
                        and not node.level:
                    module = importlib.import_module(node.module)
                    for alias in node.names:
                        namespace[alias.asname or alias.name] = getattr(
                            module, alias.name, None)
            except ImportError:
                continue
        return namespace

    @classmethod
    def _resolve(cls, node, namespace, owner):
        """The object an expression names, or None if it names nothing here.

        Handles a bare name, a dotted chain from a module, and a method
        reached through ``self``/``opt``/``cls`` -- looked for on the class
        the call is written in first, since two of these classes have a
        method of the same name.
        """
        if isinstance(node, ast.Name):
            return namespace.get(node.id)
        if not isinstance(node, ast.Attribute):
            return None
        base = node.value
        if isinstance(base, ast.Name) and base.id in ("self", "opt", "cls"):
            ordered = ([owner] if owner is not None else []) + [
                c for c in cls._owner_classes() if c is not owner]
            for candidate in ordered:
                got = getattr(candidate, node.attr, None)
                if got is not None:
                    return got
            return None
        parent = cls._resolve(base, namespace, owner)
        return getattr(parent, node.attr, None) if parent is not None else None

    @staticmethod
    def _agreed_call_ids(tree):
        """The ids of the call nodes that sit inside a ``run_agreed``.

        By node rather than by name: a second, bare call of something an
        agreement elsewhere in the same function happens to name is not
        inside an agreement, and used to read as though it were.
        """
        agreed = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "run_agreed"):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call) and inner is not node:
                        agreed.add(id(inner))
        return agreed

    @classmethod
    def _closure(cls, func):
        """``func`` and everything it calls outside an agreement.

        Returns [(qualname, function, tree, agreed call ids, namespace)], in
        the order reached. Only functions mpi-sppy defines are followed: the point is
        this library's own steps, and the standard library is where the
        local work being looked for comes *from*.
        """
        found, seen = [], set()

        def walk(f):
            qualname = getattr(f, "__qualname__", None)
            if qualname is None or qualname in seen:
                return
            seen.add(qualname)
            try:
                tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
            except (OSError, TypeError):       # a builtin, or no source
                return
            owner = f.__globals__.get(qualname.split(".")[0])
            namespace = cls._namespace(f, tree)
            agreed = cls._agreed_call_ids(tree)
            found.append((qualname, f, tree, agreed, namespace))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or id(node) in agreed:
                    continue
                if cls._called_name(node) in cls.AGREE_THEMSELVES:
                    continue
                target = cls._resolve(node.func, namespace, owner)
                if getattr(target, "__qualname__", None) \
                        in cls.CATCH_AND_CONTINUE:
                    continue
                module = getattr(target, "__module__", None) or ""
                if (target is not None and module.startswith("mpisppy")
                        and not inspect.isclass(target)):
                    walk(target)

        walk(func)
        return found

    @staticmethod
    def _called_name(node):
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
        return ""

    @classmethod
    def _checkpointing_calls(cls, func, tree, namespace):
        """(node, real name) for each checkpointing function called here.

        The name is the function's own, so an import renamed on the way in
        is reported as what it actually is. Classes are left out: raising
        ``ckpt.CheckpointMismatch`` is a raise, and the raise test is what
        has something to say about it.
        """
        owner = func.__globals__.get(
            getattr(func, "__qualname__", "").split(".")[0])
        out = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = cls._resolve(node.func, namespace, owner)
            if target is None or inspect.isclass(target):
                continue
            if getattr(target, "__module__", None) != checkpointing.__name__:
                continue
            out.append((node, getattr(target, "__name__",
                                      cls._called_name(node))))
        return out

    # ---- the properties ------------------------------------------------

    def test_every_reader_of_the_restored_state_is_a_path(self):
        """A restore step reached only from a spoke's main() is invisible to
        the closure walk unless its function is listed in PATHS, which is how
        the xhatshuffle cursor restore was once missed. So find every reader
        rather than trusting the list.

        Reads the source files rather than importing them, so a module that
        cannot be imported here is still read, and takes the held names from
        the Checkpointer's own assignments, so a new one is covered without
        being added here.
        """
        import ast
        import mpisppy
        checkpointer_tree = ast.parse(inspect.getsource(
            inspect.getmodule(Checkpointer)))
        held = {node.attr for node in ast.walk(checkpointer_tree)
                if isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Store)
                and node.attr.startswith("restored_")}
        self.assertIn("restored_loop_state", held,
                      msg="no held state found, so the scan proves nothing")
        names = held | self.RESTORED_STATE_ACCESSOR_NAMES

        def reads(func):
            for node in ast.walk(func):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.ctx, ast.Load)
                        and node.attr in names):
                    return True
                if isinstance(node, ast.Constant) and node.value in names:
                    return True
            return False

        root = os.path.dirname(os.path.abspath(mpisppy.__file__))
        tests = os.path.join(root, "tests")
        readers = set()
        for directory, _, files in os.walk(root):
            if directory == tests or directory.startswith(tests + os.sep):
                continue
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                with open(os.path.join(directory, fname)) as f:
                    tree = ast.parse(f.read())
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef,
                                         ast.AsyncFunctionDef)):
                        if reads(node):
                            readers.add(node.name)
                    elif isinstance(node, ast.ClassDef):
                        for func in node.body:
                            if (isinstance(func, (ast.FunctionDef,
                                                  ast.AsyncFunctionDef))
                                    and reads(func)):
                                readers.add(f"{node.name}.{func.name}")
        self.assertIn("XhatShuffleInnerBound._restore_loop_state_if_resuming",
                      readers, msg="the scan found nothing, so it proves "
                                   "nothing")
        paths = {name for name, _ in self.PATHS}
        self.assertEqual(
            readers - paths - self.RESTORED_STATE_ACCESSORS, set(),
            msg="these functions read what a resume restored but are not in "
                "PATHS, so nothing checks that their steps are agreed")

    def test_no_checkpointing_step_is_called_outside_an_agreement(self):
        for name, entry in self.PATHS:
            for qualname, func, tree, agreed, ns in self._closure(entry):
                for node, called in self._checkpointing_calls(func, tree, ns):
                    if (called in self.AGREE_THEMSELVES
                            or called in self.CANNOT_FAIL_ON_ONE_RANK
                            or id(node) in agreed):
                        continue
                    with self.subTest(path=name, function=qualname,
                                      step=called):
                        self.fail(
                            f"{qualname}, reached from {name}, calls "
                            f"checkpointing.{called} outside run_agreed. If "
                            f"it agrees across the ranks itself, or reads "
                            f"nothing a single rank can fail at, say so by "
                            f"naming it in AGREE_THEMSELVES or "
                            f"CANNOT_FAIL_ON_ONE_RANK; otherwise the rank it "
                            f"fails on leaves the rest of the cylinder "
                            f"waiting in the next collective.")

    @staticmethod
    def _catch_all_bodies(tree):
        """The ids of the nodes inside the body of a ``try`` that has an
        ``except Exception`` (or bare ``except``) handler which does not
        re-raise."""
        inside = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            catches_all = any(
                (handler.type is None
                 or (isinstance(handler.type, ast.Name)
                     and handler.type.id in ("Exception", "BaseException")))
                and not any(isinstance(n, ast.Raise)
                            for n in ast.walk(handler))
                for handler in node.handlers)
            if catches_all:
                for stmt in node.body:
                    inside.update(id(n) for n in ast.walk(stmt))
        return inside

    @classmethod
    def _functions_by_name(cls):
        """Every function and method defined in a loaded mpisppy module, by
        name, after importing NAME_FALLBACK_PACKAGES."""
        import pkgutil
        for package in cls.NAME_FALLBACK_PACKAGES:
            pkg = importlib.import_module(package)
            for info in pkgutil.walk_packages(pkg.__path__, f"{package}."):
                try:
                    importlib.import_module(info.name)
                except ImportError:
                    # An optional dependency is missing, so nothing in that
                    # module can be attached to a run here either.
                    continue
        index = {}
        for name, module in list(sys.modules.items()):
            if not name.startswith("mpisppy") or module is None:
                continue
            for obj in list(vars(module).values()):
                if inspect.isclass(obj) and obj.__module__ == name:
                    for attr, value in vars(obj).items():
                        value = getattr(value, "__func__", value)
                        if inspect.isfunction(value):
                            index.setdefault(attr, set()).add(value)
                elif inspect.isfunction(obj) and obj.__module__ == name:
                    index.setdefault(obj.__name__, set()).add(obj)
        return index

    @classmethod
    def _closure_by_name(cls, func):
        """``func`` and everything it can reach, as [(qualname, tree)].

        Conservative where _closure is exact: a call it can resolve is
        followed to what it names, and an attribute call it cannot is
        followed into every mpisppy function or method of that name.
        Agreements are followed too, since the question here is whether a
        collective is reached at all.
        """
        index = cls._functions_by_name()
        found, seen = [], set()

        def walk(f):
            key = f"{f.__module__}.{f.__qualname__}"
            if key in seen:
                return
            seen.add(key)
            try:
                tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
            except (OSError, TypeError):
                return
            found.append((f.__qualname__, tree))
            owner = f.__globals__.get(f.__qualname__.split(".")[0])
            namespace = cls._namespace(f, tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = cls._resolve(node.func, namespace, owner)
                target = getattr(target, "__func__", target)
                if target is not None:
                    module = getattr(target, "__module__", None) or ""
                    if module.startswith("mpisppy") \
                            and inspect.isfunction(target):
                        walk(target)
                elif isinstance(node.func, ast.Attribute):
                    for candidate in index.get(node.func.attr, ()):
                        walk(candidate)

        walk(func)
        return found

    def test_catch_and_continue_members_do(self):
        """Each CATCH_AND_CONTINUE member raises nothing, does every step
        the checks above would flag inside a catch-all ``try``, and reaches
        no collective, directly or through anything it calls."""
        members = {}
        for owner in self._owner_classes():
            for attr in vars(owner).values():
                qualname = getattr(attr, "__qualname__", None)
                if qualname in self.CATCH_AND_CONTINUE:
                    members[qualname] = attr
        self.assertEqual(sorted(members), sorted(self.CATCH_AND_CONTINUE),
                         msg="CATCH_AND_CONTINUE names a function that no "
                             "longer exists")
        for qualname, func in members.items():
            tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
            ns = self._namespace(func, tree)
            owner = func.__globals__.get(qualname.split(".")[0])
            caught = self._catch_all_bodies(tree)
            with self.subTest(function=qualname, check="raises"):
                self.assertFalse(
                    [n for n in ast.walk(tree) if isinstance(n, ast.Raise)],
                    msg=f"{qualname} raises, so a failure on one rank does "
                        f"not continue like a success on the others")
            flagged = [(n, called) for n, called
                       in self._checkpointing_calls(func, tree, ns)]
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                called = self._called_name(node)
                target = self._resolve(node.func, ns, owner)
                if (getattr(target, "__module__", None)
                        in self.LOCAL_WORK_MODULES
                        or called in self.LOCAL_WORK_METHODS):
                    flagged.append((node, called))
            for node, called in flagged:
                with self.subTest(function=qualname, step=called):
                    self.assertIn(
                        id(node), caught,
                        msg=f"{qualname} calls {called} outside a try whose "
                            f"handler catches every exception, so it can "
                            f"fail on one rank alone")
            for reached, sub_tree in self._closure_by_name(func):
                for node in ast.walk(sub_tree):
                    if not isinstance(node, ast.Call):
                        continue
                    called = self._called_name(node)
                    with self.subTest(function=qualname, reached=reached,
                                      call=called):
                        self.assertFalse(
                            called in self.COLLECTIVE_METHODS
                            or called in self.AGREE_THEMSELVES,
                            msg=f"{qualname} reaches the collective "
                                f"{called} (in {reached}); a rank whose "
                                f"step failed would skip it or reach it "
                                f"at a different point")

    def test_no_path_does_a_rank_s_own_file_handling_inline(self):
        for name, entry in self.PATHS:
            for qualname, func, tree, agreed, ns in self._closure(entry):
                owner = func.__globals__.get(qualname.split(".")[0])
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call) or id(node) in agreed:
                        continue
                    called = self._called_name(node)
                    target = self._resolve(node.func, ns, owner)
                    module = getattr(target, "__module__", None)
                    if (module not in self.LOCAL_WORK_MODULES
                            and called not in self.LOCAL_WORK_METHODS):
                        continue
                    with self.subTest(path=name, function=qualname,
                                      call=called):
                        self.fail(
                            f"{qualname}, reached from {name}, calls "
                            f"{called}() outside run_agreed. Reading or "
                            f"writing a file is this rank's own work and can "
                            f"fail on this rank alone; move it inside the "
                            f"agreement.")

    def test_no_path_raises_on_one_rank_alone(self):
        """Every raise reached from these paths is one every rank makes.

        The refusals these paths make are pure functions of the options and
        the cylinder's class, so they arrive on every rank or on none. That
        is a property of the ones written today, not of raising, and nothing
        made it hold: this asks that each one be named, with what makes it
        rank-independent, so that a raise which reads a file or a model has
        to be argued for rather than added.
        """
        counted = {}
        for name, entry in self.PATHS:
            for qualname, _, tree, _agreed, _ns in self._closure(entry):
                raises = [node for node in ast.walk(tree)
                          if isinstance(node, ast.Raise)]
                if raises:
                    counted[qualname] = (name, len(raises))

        for qualname, (name, count) in sorted(counted.items()):
            declared = self.RANK_INDEPENDENT_RAISES.get(qualname)
            with self.subTest(path=name, function=qualname):
                self.assertIsNotNone(
                    declared,
                    msg=f"{qualname}, reached from {name} outside any "
                        f"agreement, raises. If what it raises on is a file, "
                        f"a model, or anything else only this rank can see, "
                        f"the rank it fires on leaves the others waiting in "
                        f"the next collective -- put it inside run_agreed. "
                        f"If every rank really does raise together, name it "
                        f"in RANK_INDEPENDENT_RAISES with what makes that "
                        f"true.")
                self.assertEqual(
                    count, declared[0],
                    msg=f"{qualname} now raises {count} times where "
                        f"RANK_INDEPENDENT_RAISES accounts for "
                        f"{declared[0]}, said to be rank-independent because "
                        f"{declared[1]} A new raise here needs the same "
                        f"argument made for it.")

        self.assertEqual(
            sorted(self.RANK_INDEPENDENT_RAISES), sorted(counted),
            msg="RANK_INDEPENDENT_RAISES names a function these paths no "
                "longer reach, or does not name one they do")

    def test_every_agreement_is_exercised_by_the_driver(self):
        """Each ``run_agreed`` on these paths has a step that breaks it.

        Per agreement, not per path: a path with two agreements and one step
        used to pass, and three of the eight steps could be dropped from the
        driver with nothing going red.
        """
        for name, entry in self.PATHS:
            for qualname, func, tree, _agreed, ns in self._closure(entry):
                agreements = [node for node in ast.walk(tree)
                              if isinstance(node, ast.Call)
                              and isinstance(node.func, ast.Attribute)
                              and node.func.attr == "run_agreed"]
                # In source order, which is the order they are declared in.
                agreements.sort(key=lambda node: (node.lineno, node.col_offset))
                declared = self.AGREEMENTS.get(qualname, ())
                with self.subTest(path=name, function=qualname):
                    self.assertEqual(
                        len(agreements), len(declared),
                        msg=f"{qualname} has {len(agreements)} agreement(s) "
                            f"and AGREEMENTS names {len(declared)} step(s) "
                            f"for it. An agreement the driver cannot break "
                            f"is one whose first failure is a hung job.")
                for node, step in zip(agreements, declared):
                    with self.subTest(path=name, function=qualname,
                                      step=step):
                        self.assertIn(
                            step, agreement_driver.STEPS,
                            msg=f"{qualname} says its agreement is exercised "
                                f"by the step '{step}', which the driver "
                                f"does not have")
                        self.assertTrue(
                            self._agreement_runs(node, func, ns, step),
                            msg=f"the step '{step}' breaks "
                                f"{agreement_driver.STEPS[step][1]}, which "
                                f"this agreement in {qualname} does not run, "
                                f"so breaking it exercises something else")

    def test_every_step_the_driver_breaks_belongs_to_one_agreement(self):
        """And the other direction: a step named twice, or named for an
        agreement that is gone, is a step exercising something other than
        what it is written down as exercising."""
        claimed = [step for steps in self.AGREEMENTS.values()
                   for step in steps]
        self.assertEqual(sorted(claimed), sorted(set(claimed)),
                         msg=f"a step is claimed by two agreements: {claimed}")
        self.assertEqual(
            sorted(claimed), sorted(agreement_driver.STEPS),
            msg="the driver's steps and the agreements they are written down "
                "against have parted company")

    @classmethod
    def _agreement_runs(cls, node, func, namespace, step):
        """Whether the work this agreement hands over runs the driver's step.

        The step is broken by replacing a checkpointing attribute, so the
        question is whether that attribute is called inside the agreement --
        directly, or in the method the agreement hands over by name.
        """
        attribute = agreement_driver.STEPS[step][1]
        owner = func.__globals__.get(
            getattr(func, "__qualname__", "").split(".")[0])
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                target = cls._resolve(inner.func, namespace, owner)
                if getattr(target, "__name__", None) == attribute:
                    return True
            elif isinstance(inner, ast.Attribute):
                handed = cls._resolve(inner, namespace, owner)
                module = getattr(handed, "__module__", None) or ""
                if (handed is None or not module.startswith("mpisppy")
                        or inspect.isclass(handed)):
                    continue
                for qualname, _, tree, _agreed, inner_ns in cls._closure(handed):
                    inner_owner = inner_ns.get(qualname.split(".")[0])
                    for call in ast.walk(tree):
                        if not isinstance(call, ast.Call):
                            continue
                        target = cls._resolve(call.func, inner_ns, inner_owner)
                        if getattr(target, "__name__", None) == attribute:
                            return True
        return False


if __name__ == "__main__":
    unittest.main()
