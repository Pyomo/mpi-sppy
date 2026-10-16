###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""A/B checkpoint harness on a production option profile.

The other checkpoint harnesses each isolate one mechanism. This one runs the
option set a user actually runs -- a relaxed-PH cylinder feeding a primal hub,
with the relaxed-PH fixer, automatic mipgaps, sep-rho, integer relaxation,
presolve, linearized prox terms and warm starts all on at once -- and asks the
question that matters to that user: stop the run in the middle, resume it, and
does the answer still hold?

Nothing else in the checkpointing tests covers any of these:

* ``PHPrimalHub``. Every other checkpoint test runs ``PHHub``, whose W comes
  from its own subproblem duals. This hub's W arrives from a cylinder, so its
  iterate depends on a cylinder's state as well as its own.
* ``RelaxedPHSpoke``, which no other resumed run has, and which is the only
  cylinder that checkpoints PH state of its own rather than an incumbent.
* ``RelaxedPHFixer``. Phase 3 changed it -- its "the modeler fixed this" set
  now goes through ``was_initially_fixed`` -- with no A/B test behind the
  change. It is a different extension from the ``Fixer`` in
  ``test_checkpoint_extensions.py``.
* ``IntegerRelaxThenEnforce`` and ``Gapper``, neither of which implements
  ``checkpoint_state``.

``IntegerRelaxThenEnforce`` is the one with a hazard visible in the source:
``pre_iter0`` applies ``core.relax_integer_vars`` unconditionally and sets
``_integers_relaxed``, while a resumed run reloads models that already carry
whatever relaxation state they were checkpointed in and builds a fresh
extension object that believes nothing has happened yet. Both instances here
stop while the integers are relaxed, which is the state a real run is
overwhelmingly likely to be stopped in.

No leg of this profile reproduces another exactly, stop or no stop, and the
file is built around that rather than against it. The hub's W arrives from a
cylinder and its incumbent from an asynchronous one, so how far each gets
depends on wall-clock timing: two identical uninterrupted legs measured
bit-identical on a quiet machine and 1.0 apart in a nonant under load. A
resumed leg differs by more, and not because a restore was lost: the dual
cylinder now carries its W, but it counts its own iterations and spins far
ahead of the hub -- 62 of them by the hub's third -- so the point it resumes
from is its own, and the fixer downstream of it acts on what that produces.
Measured on sizes with 10 scenarios: 480 of 3250 recorded state entries
differ and the expected objective by 4.8e-4 relative.

So this file pins what the determinism contract (design section 7) promises
for a configuration whose hub reads a cylinder: a valid continuation, an
objective that does not walk away, bounds that stay ordered, an incumbent
that does not regress, and each extension picking up in the state the study
left it in -- not a reproduced trajectory. A test asserting reproduction here
would be asserting that the cylinders got scheduled the same way twice.

Two departures from the profile as the user writes it, both deliberate:

* The user's Gurobi ``solver_options`` are left out. These tests run whatever
  solver ``get_solver`` finds.
* The profile's own early exits (``intra_hub_conv_thresh``, ``rel_gap``) are
  overridden by ``_run_leg``, as in the other harnesses: an A/B comparison
  needs both legs to run the same iterations.

One correction: the profile sets ``integer_relax_then_enforce_ratio`` without
``integer_relax_then_enforce``, and the ratio alone does nothing --
``mpisppy/generic/extensions.py`` builds the extension only when the boolean
is set. The boolean is set here, or the extension under test would not exist.
"""

import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import pyomo.environ as pyo

from mpisppy.tests.utils import (get_solver, solver_takes_model_size,
                                 subprocess_env)

solver_available, solver_name, persistent_available, persistent_solver_name = \
    get_solver()

_HERE = os.path.dirname(os.path.abspath(__file__))
_DRIVER = os.path.join(_HERE, "cylinders_ab_driver.py")
#: Dotted names: generic_cylinders resolves --module-name with importlib, so
#: this keeps the legs independent of the directory mpiexec starts in.
_SIZES = "mpisppy.tests.examples.sizes.sizes"
_UC = "mpisppy.tests.examples.uc.uc_funcs"

mpiexec_available = shutil.which("mpiexec") is not None
egret_available = importlib.util.find_spec("egret") is not None
#: The uc case below is minutes rather than seconds. It runs by default
#: wherever the solver can take it, because a case that never runs covers
#: nothing; set this to skip it while working on the fast ones.
skip_slow = os.environ.get("MPISPPY_SKIP_SLOW_TESTS", "") not in ("", "0")


def _skip_unless_the_solver_takes_a_uc_scenario():
    """Skip where the available solver is a size-limited build.

    A uc scenario is tens of thousands of rows and columns, which is past
    what the community editions of cplex, gurobi and xpress accept -- and
    those are what CI installs, so this case runs on a developer's or a
    cluster's solver rather than on GitHub's. The size is read off the
    scenario rather than written down here, and the probe that asks the
    solver about it is a trivial LP of that size: about 6.6 seconds together
    on a fast machine with a full licence (2.0 s to build the scenario, 2.1 s
    for the probe, the rest importing egret), against the minutes a leg takes.

    The skip says what the solver actually did. Being refused for size and
    being unable to solve at all are both reasons this case cannot run, and
    an expired licence reported as "the model is too big" would send whoever
    reads it looking in the wrong place.
    """
    from mpisppy.tests.examples.uc import uc_funcs
    scenario = uc_funcs.scenario_creator("Scenario1")
    num_vars = sum(1 for _ in scenario.component_data_objects(pyo.Var,
                                                             active=True))
    num_cons = sum(1 for _ in scenario.component_data_objects(pyo.Constraint,
                                                             active=True))
    solved, why = solver_takes_model_size(solver_name, num_vars, num_cons)
    if not solved:
        raise unittest.SkipTest(
            f"a uc scenario is {num_vars} variables and {num_cons} "
            f"constraints, and {why}")


#: The user's profile, minus the solver-specific and early-exit options named
#: in the module docstring. Four cylinders: the primal hub plus lagrangian,
#: relaxed-PH and xhatshuffle.
_PROFILE_ARGS = (
    "--default-rho", "1e-4",
    "--max-solver-threads", "2",
    "--max-stalled-iters", "30000",
    "--bundles-per-rank", "0",
    "--linearize-proximal-terms",
    "--proximal-linearization-tolerance", "1e-12",
    "--sep-rho", "--sep-rho-multiplier", "0.1",
    "--presolve",
    "--integer-relax-then-enforce",
    "--rounding-bias", "0.3",
    "--starting-mipgap", "1e-2", "--mipgap-ratio", "1",
    "--lagrangian-starting-mipgap", "0.01",
    "--lagrangian-mipgap-ratio", "0.1",
    "--warmstart-subproblems",
    "--relaxed-ph-rescale-rho-factor", "10",
    "--ph-primal-hub",
    "--lagrangian",
    "--relaxed-ph",
    "--relaxed-ph-fixer",
    "--xhatshuffle",
)


def _run_leg(tmpdir, name, module, model_args, extra_args):
    """Run one mpiexec job. Returns (CompletedProcess, out_path)."""
    out_path = os.path.join(tmpdir, f"{name}.json")
    cmd = [
        "mpiexec", "-np", "4",
        sys.executable, "-m", "mpi4py", _DRIVER,
        "--out", out_path,
        "--module-name", module,
        *model_args,
        "--solver-name", solver_name,
        *_PROFILE_ARGS,
        # The comparison needs both legs to run the same iterations, so every
        # early exit has to be off: no inter-cylinder convergence, no
        # gap-based termination.
        "--intra-hub-conv-thresh", "-1",
        "--rel-gap", "0.0", "--abs-gap", "0.0",
        *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=3600, check=False, env=subprocess_env())
    if result.returncode != 0:
        raise AssertionError(
            f"leg {name!r} failed:\n{result.stdout[-4000:]}\n"
            f"{result.stderr[-4000:]}")
    return result, out_path


def _snapshot(out_path, spokes_from):
    """The hub's snapshot for a leg, with whatever its spokes reported."""
    with open(out_path) as f:
        snapshot = json.load(f)
    snapshot["spokes"] = []
    prefix = f"{os.path.basename(out_path)}.spoke"
    for fname in sorted(os.listdir(spokes_from)):
        if fname.startswith(prefix):
            with open(os.path.join(spokes_from, fname)) as f:
                snapshot["spokes"].append(json.load(f))
    return snapshot


class _ProfileABMixin:
    """Three legs -- reference, stopped, resumed -- run once per class.

    Once per class rather than once per test: each leg is an mpiexec job, and
    the uc instance takes about a minute of them.
    """

    N = 6
    STOP = 3
    MODULE = None
    MODEL_ARGS = ()
    #: Relative agreement required of the expected objective, or None where
    #: the instance cannot support the comparison at all (see the uc case).
    #: The resumed leg takes a different path (see the module docstring), so
    #: this is the quantity that is pinned rather than the iterate.
    OBJECTIVE_RTOL = 1e-2
    #: How much of the run to spend with the integers relaxed. The profile
    #: says 1.1, which puts both the iteration and the time condition out of
    #: reach, so a run stops while still relaxed; the uc case below lowers it
    #: to reach the other state.
    RATIO = "1.1"

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt_dir = os.path.join(cls._tmp.name, "ckpt")
        cls.logs = {}

        def leg(name, *extra):
            extra = ("--integer-relax-then-enforce-ratio", cls.RATIO) + extra
            result, out_path = _run_leg(cls._tmp.name, name, cls.MODULE,
                                        cls.MODEL_ARGS, extra)
            cls.logs[name] = result.stdout
            return _snapshot(out_path, cls._tmp.name)

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

    def test_the_profile_was_in_force(self):
        """Every option under test left a mark on the reference leg.

        Without this the rest of the file could pass on a plain PH run: a
        profile option that silently stopped being wired up would take its
        coverage with it and nothing below would notice.
        """
        log = self.logs["A"]
        for evidence, what in (
                ("PHPrimalHub", "the primal hub"),
                ("RelaxedPHSpoke", "the relaxed-PH cylinder"),
                ("fixed by heuristic", "the relaxed-PH fixer"),
                ("relaxing integrality constraints", "integer relaxation"),
                ("Gapper: Changing mipgap", "the automatic mipgap"),
                ("Using sep-rho rho setter", "sep-rho"),
                ("SPFBBT", "the presolver"),
        ):
            self.assertIn(evidence, log,
                          msg=f"{what} left no trace in the reference leg")

    def test_the_resumed_leg_really_resumed(self):
        """Without this the rest proves only that the instance is solvable.

        A leg that ignored the checkpoint and ran the iterations from scratch
        would satisfy every comparison below.
        """
        self.assertEqual(self.stopped["iteration"], self.STOP)
        self.assertTrue(self.resumed["resumed"],
                        msg="the third leg started from scratch")
        self.assertEqual(self.resumed["resume_iteration"], self.STOP)
        self.assertEqual(self.resumed["iteration"], self.N)

    def test_the_objective_agrees_within_tolerance(self):
        if self.OBJECTIVE_RTOL is None:
            self.skipTest("this instance's objective is not leg-comparable")
        want, got = self.reference["objective"], self.resumed["objective"]
        self.assertIsNotNone(want)
        scale = max(1.0, abs(want))
        self.assertLessEqual(
            abs(want - got), self.OBJECTIVE_RTOL * scale,
            msg=f"the resumed run's expected objective is {got}, the "
                f"uninterrupted run's is {want}")

    def test_bounds_stay_valid_after_a_resume(self):
        self.assertLessEqual(
            self.resumed["BestOuterBound"], self.resumed["BestInnerBound"],
            msg="the resumed run's outer bound crossed its incumbent")
        bound = self.resumed["best_bound_obj_val"]
        incumbent = self.resumed["best_solution_obj_val"]
        if bound is not None and incumbent is not None:
            self.assertLessEqual(
                bound, incumbent,
                msg="the hub's restored best bound crossed the incumbent")

    def test_the_incumbent_does_not_regress_across_the_stop(self):
        """Minimization: a smaller inner bound is a better incumbent."""
        self.assertLessEqual(
            self.resumed["BestInnerBound"], self.stopped["BestInnerBound"],
            msg="the resumed run reports a worse incumbent than the "
                "checkpoint it resumed from")

    def test_the_xhat_spoke_restored_what_it_wrote(self):
        """The answer lives on the xhat spoke, not in the hub checkpoint."""
        written = os.listdir(os.path.join(self.ckpt_dir, "spokes"))
        self.assertTrue(
            any(f.startswith("spoke_XhatShuffleInnerBound") for f in written),
            msg=f"the xhat spoke checkpointed no incumbent: {written}")
        restored = [s for s in self.resumed["spokes"]
                    if s["restored_incumbent_obj"] is not None]
        self.assertTrue(restored,
                        msg=f"no spoke restored an incumbent: "
                            f"{self.resumed['spokes']}")
        self.assertEqual(
            restored[0]["restored_incumbent_obj"],
            self.stopped["BestInnerBound"],
            msg="the spoke restored something other than the incumbent its "
                "own checkpoint held")

    def test_the_dual_cylinder_carried_its_dual_weights(self):
        """The W the primal hub iterates on is the study's, not a fresh zero.

        This cylinder holds no incumbent, so nothing in the hub checkpoint or
        in an xhat spoke's file describes it; before it wrote its own, a
        resumed wheel restored the hub perfectly and then fed it duals from a
        cylinder starting over.
        """
        spokes_dir = os.path.join(self.ckpt_dir, "spokes")
        names = [f for f in os.listdir(spokes_dir)
                 if f.startswith("spoke_RelaxedPHSpoke")]
        self.assertTrue(
            names,
            msg=f"the dual cylinder checkpointed nothing: "
                f"{os.listdir(spokes_dir)}")
        with open(os.path.join(spokes_dir, names[0]), "rb") as f:
            state = pickle.load(f)
        weights = [w for entry in state["duals"].values()
                   for w in entry["W"].values()]
        self.assertTrue(
            any(w for w in weights),
            msg="the dual cylinder wrote a W of all zeros, which a resume "
                "cannot be distinguished from starting over")
        restored = [spoke["restored_dual_generation"]
                    for spoke in self.resumed["spokes"]
                    if spoke["cylinder"] == "RelaxedPHSpoke"]
        self.assertEqual(
            restored, [state["generation"]],
            msg="the resumed cylinder did not pick up the W on disk")

    def test_the_fixer_ran_on_both_sides_of_the_stop(self):
        """The extension phase 3 changed, exercised across a resume."""
        for name in ("A", "B1", "B2"):
            self.assertIn("fixed by heuristic", self.logs[name],
                          msg=f"the relaxed-PH fixer never fixed in leg "
                              f"{name}")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestSizesProfileResumeAB(_ProfileABMixin, unittest.TestCase):
    """The fast instance: a MIP that runs the whole profile in seconds."""

    MODULE = _SIZES
    MODEL_ARGS = ("--num-scens", "10")

    def test_the_checkpoint_was_taken_with_the_integers_relaxed(self):
        """The state the hazard lives in, and the one a real run stops in.

        At ratio 1.1 neither the iteration nor the time condition can fire
        inside a run, so the stopped leg is still relaxed when it writes. The
        resumed leg then reloads models that carry the relaxation and builds a
        fresh extension that believes nothing has happened -- so it applies
        the transformation a second time. It must survive that, and it must
        not decide the run is finished with integrality.
        """
        self.assertIn("relaxing integrality constraints", self.logs["B1"])
        self.assertNotIn("nforcing integrality constraints", self.logs["B1"],
                         msg="the stopped leg enforced integrality, so the "
                             "checkpoint was not taken in the relaxed state")
        # The resumed leg does not relax anything: the models it reloaded are
        # already relaxed, and pre_iter0 leaves a resumed run's models alone
        # rather than applying the transformation to them twice.
        self.assertNotIn("relaxing integrality constraints", self.logs["B2"],
                         msg="the resumed leg relaxed models that came back "
                             "from the checkpoint already relaxed")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestSizesProfileEnforcedResumeAB(_ProfileABMixin, unittest.TestCase):
    """The same profile, stopped after integrality has been enforced.

    The profile's own ratio of 1.1 cannot reach this state -- above 1 neither
    the iteration nor the time condition fires inside a run -- so the ratio is
    lowered here. It is lowered to 0.1 rather than to something in the middle
    because the ratio is a fraction of *this run's* iteration budget: at 0.1
    both the uninterrupted leg and the stopped leg enforce at iteration 1, and
    a ratio where they enforced at different iterations would be comparing two
    different studies.

    ``test_checkpoint_extensions.py`` pins the mechanism in one process. This
    is the same claim through the wheel, where the hub's W arrives from a
    cylinder and the fixer acts on what that cylinder sends.
    """

    MODULE = _SIZES
    MODEL_ARGS = ("--num-scens", "10")
    RATIO = "0.1"

    def test_the_checkpoint_was_taken_with_the_integers_enforced(self):
        self.assertIn("nforcing integrality constraints", self.logs["B1"],
                      msg="the stopped leg never enforced, so the checkpoint "
                          "was not taken in the enforced state")

    def test_the_resumed_leg_does_not_re_relax_what_was_enforced(self):
        """The property the whole file exists for, in its sharpest form.

        A resumed leg that relaxed here would solve relaxed subproblems where
        the uninterrupted leg solved integral ones, and would enforce a second
        time from a different iterate.
        """
        self.assertNotIn(
            "relaxing integrality constraints", self.logs["B2"],
            msg="the resumed leg relaxed integrality that the study had "
                "already enforced")
        self.assertNotIn(
            "nforcing integrality constraints", self.logs["B2"],
            msg="the resumed leg enforced integrality a second time")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
@unittest.skipIf(not egret_available, "uc needs egret")
@unittest.skipIf(skip_slow, "MPISPPY_SKIP_SLOW_TESTS is set")
class TestUCProfileResumeAB(_ProfileABMixin, unittest.TestCase):
    """The slow instance: unit commitment, minutes rather than seconds.

    Worth its minutes because it is a real MIP at a size where the parts of a
    resume that are cheap on ``sizes`` are not: one checkpoint write here is
    47 MB of dilled models and takes about a minute, which is the cost a user
    of this profile actually pays.

    What it cannot do is compare objectives. The profile's automatic mipgap
    reads the wheel's bound gap, the incumbent it reads arrives from an
    asynchronous xhat cylinder, and the slower a leg runs the further that
    cylinder gets -- so a leg that writes checkpoints finds a better incumbent
    early, tightens its subproblem mipgap, and solves to a much better
    expected objective than an uninterrupted leg of the same length. Measured:
    leg A ended at 5,342,186 and the resumed leg at 67,588, a relative
    difference of 0.99, while the two legs' recorded nonants never differed by
    more than 3.9. That is the adaptive mipgap reacting to timing, not a
    resume losing anything, and no tolerance can tell the two apart here.
    """

    MODULE = _UC
    MODEL_ARGS = ("--num-scens", "3")
    N = 4
    STOP = 2
    OBJECTIVE_RTOL = None

    @classmethod
    def setUpClass(cls):
        _skip_unless_the_solver_takes_a_uc_scenario()
        super().setUpClass()


class TestSolverSizeProbe(unittest.TestCase):
    """The probe that decides whether the uc case can run at all.

    Its whole job is to tell "this solver will not take a model this big"
    from "this solver worked". Getting that wrong in the permissive
    direction puts the uc legs back in front of a solver that cannot run
    them, which is the CI failure the guard exists to remove; getting it
    wrong in the other direction silently turns off the only uc coverage
    there is.
    """

    SIZE = (10, 10)

    class _Results:
        def __init__(self, condition):
            self.solver = types.SimpleNamespace(
                termination_condition=condition)

    class _Solver:
        """A non-persistent solver interface that returns without solving."""
        def __init__(self, results=None, raises=None):
            self._results = results
            self._raises = raises
            self.options = {}

        def solve(self, model, **kwargs):
            if self._raises is not None:
                raise self._raises
            return self._results

    def _probe_with(self, solver):
        with mock.patch("pyomo.environ.SolverFactory",
                        return_value=solver):
            return solver_takes_model_size("stand-in", *self.SIZE)

    def test_a_refusal_reported_through_the_results_is_not_a_solve(self):
        # Some solvers raise on a refusal and some report it in the results
        # object -- xpress is recorded doing the latter. A probe that asked
        # only whether solve() raised would call this a success.
        solved, why = self._probe_with(
            self._Solver(results=self._Results("maxTimeLimit")))
        self.assertFalse(solved)
        self.assertIn("maxTimeLimit", why)

    def test_an_optimal_solve_is_a_solve(self):
        solved, why = self._probe_with(
            self._Solver(results=self._Results("optimal")))
        self.assertTrue(solved)
        self.assertIsNone(why)

    def test_a_raised_error_is_reported_in_the_solvers_own_words(self):
        solved, why = self._probe_with(
            self._Solver(raises=RuntimeError("size limits exceeded")))
        self.assertFalse(solved)
        self.assertIn("size limits exceeded", why)
        self.assertIn("RuntimeError", why)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_the_available_solver_solves_a_small_model(self):
        """Whatever the interface, a real solver answers yes to a tiny LP.

        Without this the three above could all pass while the probe was
        broken for every real solver -- the persistent interfaces need
        set_instance, and their results object is shaped differently.
        """
        solved, why = solver_takes_model_size(solver_name, *self.SIZE)
        self.assertTrue(solved, msg=why)


if __name__ == "__main__":
    unittest.main()
