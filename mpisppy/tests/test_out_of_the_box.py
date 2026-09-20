###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""CI tests for the out-of-the-box (OOTB) interpreter wiring.

Solver-free: the OOTB decision/apply path and the base-tier probe only need to
*build* a scenario (no solve), so the whole gather_facts -> recommend ->
apply_decision -> configure flow runs in CI on the farmer example. The
environment rank count is supplied via --inspect-only N so the EF and
decomposition branches are both exercised deterministically without launching
mpiexec. (The solver-dependent run/measurement tiers are exercised on demand /
locally; see test_ootb_validate / test_ootb_calibrate.)
"""

import copy
import os
import shlex
import sys
import unittest

import pyomo.environ as pyo

import mpisppy.utils.config as config
from mpisppy.generic import parsing
from mpisppy.generic import out_of_the_box as ootb


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


_FARMER_DIR = os.path.join(_repo_root(), "examples", "farmer")


def _farmer_module():
    if _FARMER_DIR not in sys.path:
        sys.path.insert(0, _FARMER_DIR)
    import farmer
    return farmer


def _farmer_cfg(num_scens=6, **overrides):
    module = _farmer_module()
    cfg = config.Config()
    parsing.add_driver_args(cfg, module)
    cfg.num_scens = num_scens
    cfg.module_name = "farmer"
    for k, v in overrides.items():
        cfg[k] = v
    return cfg, module


class TestConfigureNoSolver(unittest.TestCase):
    """Drive configure() (probe + recommend + apply) without solving."""

    def setUp(self):
        self._argv = sys.argv
        sys.argv = ["prog", "--module-name", "farmer", "--num-scens", "6"]

    def tearDown(self):
        sys.argv = self._argv

    def test_base_few_ranks_picks_ef(self):
        cfg, module = _farmer_cfg(out_of_the_box="", inspect_only="1")
        state = ootb.configure(module, cfg)
        self.assertTrue(state.decision.run_ef)
        self.assertEqual(state.decision.ef_reason, "min_ranks")
        self.assertTrue(cfg.EF)                       # apply_decision set it
        # the probe populated a size profile (farmer is continuous)
        self.assertIsNotNone(state.facts.vars_cont)
        self.assertEqual(state.facts.vars_int, 0)

    def test_minus_tier_decomposes_by_count(self):
        cfg, module = _farmer_cfg(out_of_the_box_minus="", inspect_only="6")
        state = ootb.configure(module, cfg)
        self.assertFalse(state.decision.run_ef)
        self.assertTrue(cfg.lagrangian and cfg.xhatshuffle)
        self.assertIsNone(state.facts.vars_cont)      # minus: no probe
        # minus cannot bundle
        self.assertIsNone(cfg.get("scenarios_per_bundle"))

    def test_base_forced_decomposition_bundles(self):
        # user --lagrangian forces decomposition even though the problem is small
        sys.argv = sys.argv + ["--lagrangian"]
        cfg, module = _farmer_cfg(num_scens=60, out_of_the_box="",
                                  inspect_only="3", lagrangian=True)
        state = ootb.configure(module, cfg)
        self.assertFalse(state.decision.run_ef)
        spb = cfg.get("scenarios_per_bundle")
        self.assertIsNotNone(spb)                     # 60 scens -> bundles
        self.assertEqual(60 % int(spb), 0)

    def test_report_suggestions_runs(self):
        cfg, module = _farmer_cfg(out_of_the_box="", inspect_only="1")
        state = ootb.configure(module, cfg)
        ootb.report_suggestions(state)                # config-time suggestions
        self.assertIsInstance(state.decision.suggestions, list)


class TestInspectStandalone(unittest.TestCase):
    def test_verify_instantiation_and_standalone(self):
        cfg, module = _farmer_cfg()
        profile = ootb.verify_instantiation(module, cfg)
        self.assertEqual(set(profile),
                         {"vars_int", "vars_cont", "nonants_total", "nonants_int",
                          "model_degree"})
        self.assertGreater(profile["vars_cont"], 0)
        self.assertEqual(profile["model_degree"], "linear")   # farmer is an LP
        ootb.inspect_only_standalone(module, cfg)     # prints, returns None


class TestApplyDecision(unittest.TestCase):
    def test_apply_sets_cfg_values(self):
        cfg, _ = _farmer_cfg()
        d = ootb.Decision()
        d.args = [ootb.ChosenArg("--lagrangian", None, "x"),
                  ootb.ChosenArg("--solver-name", "gurobi", "x"),
                  ootb.ChosenArg("--scenarios-per-bundle", "10", "x"),
                  ootb.ChosenArg("--rel-gap", "0.01", "x")]
        ootb.apply_decision(d, cfg)
        self.assertTrue(cfg.lagrangian)
        self.assertEqual(cfg.solver_name, "gurobi")
        self.assertEqual(cfg.scenarios_per_bundle, 10)
        self.assertAlmostEqual(cfg.rel_gap, 0.01)

    def test_ef_solver_name_carried_over(self):
        cfg, _ = _farmer_cfg()
        d = ootb.Decision(run_ef=True)
        d.args = [ootb.ChosenArg("--solver-name", "cplex", "x")]
        ootb.apply_decision(d, cfg)
        self.assertTrue(cfg.EF)
        self.assertEqual(cfg.EF_solver_name, "cplex")   # belt-and-suspenders


class TestCommandLineAndFlags(unittest.TestCase):
    def test_command_line_two_stage(self):
        facts = ootb.Facts("farmer", 3, set(), 6, scen_anchor="--num-scens 6")
        d = ootb.Decision()
        d.args = [ootb.ChosenArg("--lagrangian", None, "x"),
                  ootb.ChosenArg("--solver-name", "gurobi", "x")]
        cl = d.command_line(facts)
        self.assertIn("--module-name farmer", cl)
        self.assertIn("--num-scens 6", cl)
        self.assertIn("--lagrangian", cl)
        self.assertIn("-np 3", cl)

    def test_command_line_multistage(self):
        # gather_facts quotes the list value; a ListOf option is ONE argparse
        # token, so an unquoted "3 2" gives "unrecognized arguments: 2".
        facts = ootb.Facts("aircond", 3, set(), 6, multistage=True,
                           branching_factors=[3, 2],
                           scen_anchor="--branching-factors '3 2'")
        cl = ootb.Decision().command_line(facts)
        self.assertIn("--branching-factors '3 2'", cl)
        self.assertNotIn("--num-scens", cl)

    def test_empty_branching_factors_on_the_driver_path(self):
        # The guard has to hold where OOTB hands off too: np.prod([]) is 1.0,
        # a FLOAT, which fails as a range bound inside the model. OOTB plans
        # for --num-scens, so the run must agree.
        from mpisppy.generic import parsing as _parsing
        cfg, module = _farmer_cfg()
        cfg.num_scens = 100
        cfg.branching_factors = []
        names, nodenames = _parsing.name_lists(module, cfg)
        self.assertEqual(len(names), 100)
        self.assertIsNone(nodenames)

    def test_empty_branching_factors_is_not_a_scenario_count(self):
        # math.prod([]) is 1, so an empty list used to report a one-scenario
        # problem and quietly take the EF instead of saying it cannot tell.
        cfg, module = _farmer_cfg()
        cfg.branching_factors = []
        cfg.num_scens = None
        with self.assertRaises(RuntimeError):
            ootb._detect_num_scens(module, cfg)

    def test_empty_branching_factors_is_not_multistage(self):
        cfg, module = _farmer_cfg()
        cfg.branching_factors = []
        facts = ootb.gather_facts(module, cfg, "minus", ootb.load_policy())
        self.assertFalse(facts.multistage)
        self.assertIsNone(facts.branching_factors)

    def test_empty_branching_factors_is_not_an_anchor(self):
        # --branching-factors "" parses to [], which is not None; a bare
        # --branching-factors token makes the printed line fail with
        # "expected one argument".
        cfg, module = _farmer_cfg()
        cfg.branching_factors = []
        facts = ootb.gather_facts(module, cfg, "minus", ootb.load_policy())
        cl = ootb.Decision().command_line(facts)
        self.assertNotIn("--branching-factors", cl)

    def test_kw_creator_failure_is_reported_not_swallowed(self):
        # The generic "define --num-scens" advice is wrong for a model that
        # declares no such flag; the real cause is the model's own missing
        # option, and it is the only message that says which.
        class _Mod:
            @staticmethod
            def kw_creator(cfg):
                raise AttributeError("'NoneType' object has no attribute 'split'")

            @staticmethod
            def scenario_names_creator(n, start=None):
                return [f"s{i}" for i in range(start or 0, (start or 0) + n)]

        cfg, _ = _farmer_cfg()
        cfg.num_scens = None
        with self.assertRaises(RuntimeError) as cm:
            ootb._detect_num_scens(_Mod, cfg)
        msg = str(cm.exception)
        self.assertIn("AttributeError", msg)
        self.assertIn("split", msg)
        self.assertNotIn("--num-scens", msg)

    def test_ef_command_line_is_serial(self):
        # An EF is one monolithic solve; echoing mpiexec would tell the reader
        # to do the thing ExtensiveForm itself warns about ("Creating an
        # ExtensiveForm object in parallel. Why?").
        facts = ootb.Facts("farmer", 6, set(), 6, scen_anchor="--num-scens 6")
        d = ootb.Decision(run_ef=True)
        cl = d.command_line(facts)
        self.assertTrue(cl.startswith("python -m mpisppy.generic_cylinders"),
                        msg=cl)
        self.assertNotIn("mpiexec", cl)
        self.assertNotIn("mpi4py", cl)
        # a decomposition still gets the parallel launcher
        cl2 = ootb.Decision(run_ef=False).command_line(facts)
        self.assertIn("mpiexec -np 6", cl2)

    def _ef_msgs(self, facts, ef_reason):
        d = ootb.Decision(run_ef=True, ef_reason=ef_reason)
        return ootb.make_suggestions(d, facts, ootb.load_policy())

    def test_ef_under_mpiexec_is_called_out(self):
        facts = ootb.Facts("farmer", 6, set(), 6)
        self.assertTrue(any("idled" in m for m in self._ef_msgs(facts,
                                                                "small_effort")),
                        msg="no suggestion about wasting ranks on an EF")

    def test_ef_suggestion_only_when_decomposing_was_available(self):
        facts = ootb.Facts("farmer", 6, set(), 6)
        # the EF is the only sensible choice at one rank
        solo = ootb.Facts("farmer", 1, set(), 6)
        self.assertFalse(any("idled" in m
                             for m in self._ef_msgs(solo, "small_effort")))
        # the policy REFUSED to decompose at this rank count, so telling the
        # reader to decompose would contradict the suggestion beside it
        for reason in ("min_ranks", "request_too_big"):
            with self.subTest(reason=reason):
                self.assertFalse(any("idled" in m
                                     for m in self._ef_msgs(facts, reason)))
        # --inspect-only reports and exits, so nothing was solved and nothing
        # idled -- in EITHER form, with a planned rank count or without
        planned = ootb.Facts("farmer", 512, set(), 6, inspect_only=True)
        self.assertFalse(any("idled" in m
                             for m in self._ef_msgs(planned, "small_effort")))
        bare = ootb.Facts("farmer", 6, set(), 6, inspect_only=True)
        self.assertFalse(any("idled" in m
                             for m in self._ef_msgs(bare, "small_effort")))
        # an explicit --EF below the rank floor: OOTB would not have
        # decomposed there either, so do not tell the reader to
        below_floor = ootb.Facts("farmer", 2, set(), 6)
        self.assertFalse(any("idled" in m
                             for m in self._ef_msgs(below_floor, "user")))

    def test_command_line_has_no_anchor_when_the_model_takes_none(self):
        # netdes declares no --num-scens; printing a derived one gave a line
        # its own parser rejects.
        facts = ootb.Facts("netdes", 1, set(), 20)
        cl = ootb.Decision().command_line(facts)
        self.assertNotIn("--num-scens", cl)
        self.assertNotIn("--branching-factors", cl)

    def test_gather_facts_quotes_the_branching_factor_anchor(self):
        import shlex
        cfg, module = _farmer_cfg()
        cfg.branching_factors = [3, 2]   # already declared by the driver
        facts = ootb.gather_facts(module, cfg, "minus", ootb.load_policy())
        self.assertIsNotNone(facts.scen_anchor)
        toks = shlex.split(facts.scen_anchor)
        self.assertEqual(toks, ["--branching-factors", "3 2"])

    def test_requested_and_effort_and_policy(self):
        cfg, _ = _farmer_cfg(out_of_the_box="")
        self.assertTrue(ootb.requested(cfg))
        effort, path = ootb.effort_and_policy(cfg)
        self.assertEqual(effort, "base")
        self.assertIsNone(path)
        cfg2, _ = _farmer_cfg()
        self.assertFalse(ootb.requested(cfg2))
        self.assertEqual(ootb.effort_and_policy(cfg2), (None, None))

    def test_two_tiers_is_an_error(self):
        cfg, _ = _farmer_cfg(out_of_the_box="", out_of_the_box_minus="")
        with self.assertRaises(RuntimeError):
            ootb.effort_and_policy(cfg)

    def test_user_flags_from_argv(self):
        saved = sys.argv
        try:
            sys.argv = ["p", "--lagrangian", "--max-iterations=50", "-q"]
            flags = ootb._user_flags()
            self.assertIn("--lagrangian", flags)
            self.assertIn("--max-iterations", flags)   # =value stripped
            self.assertNotIn("-q", flags)              # single dash ignored
        finally:
            sys.argv = saved


class TestSuggestionGenerators(unittest.TestCase):
    """Exercise each computed suggestion generator."""

    def setUp(self):
        self.policy = ootb.load_policy()

    def _msgs(self, d, facts, outcome=None):
        return ootb.make_suggestions(d, facts, self.policy, outcome)

    def test_few_ranks_and_minus(self):
        facts = ootb.Facts("m", 1, set(), 3, effort="minus")
        d = ootb.Decision(run_ef=True, ef_reason="min_ranks")
        self.assertTrue(any("only 1 MPI" in m for m in self._msgs(d, facts)))

    def test_no_persistent_and_more_ranks(self):
        facts = ootb.Facts("m", 6, set(), 10, effort="base")
        d = ootb.Decision(run_ef=False, chosen_solver="gurobi", num_cylinders=3)
        msgs = self._msgs(d, facts)
        self.assertTrue(any("persistent" in m for m in msgs))
        self.assertTrue(any("cylinders configured" in m for m in msgs))

    def test_linearized_prox(self):
        facts = ootb.Facts("m", 6, set(), 10, effort="base")
        d = ootb.Decision(run_ef=False, chosen_solver="cbc", num_cylinders=3)
        self.assertTrue(any("linearized" in m for m in self._msgs(d, facts)))

    def test_minus_no_bundling(self):
        facts = ootb.Facts("m", 6, set(), 100, effort="minus")
        d = ootb.Decision(run_ef=False, chosen_solver="gurobi", num_cylinders=3)
        self.assertTrue(any("minus" in m for m in self._msgs(d, facts)))

    def test_no_generator_depends_on_an_outcome(self):
        # No caller supplies `outcome` -- both report_suggestions call sites
        # pass none -- so a generator reading it could never fire. The one
        # that did was removed rather than left as dead code claiming a
        # capability; this keeps a new one from being added inert.
        facts = ootb.Facts("m", 6, set(), 10, effort="base")
        d = ootb.Decision(run_ef=False, chosen_solver="gurobi", num_cylinders=3)
        outcome = {"converged": False, "iterations": 100, "rel_gap": 0.2}
        self.assertEqual(self._msgs(d, facts, outcome),
                         self._msgs(d, facts, None),
                         msg="a suggestion generator reads `outcome`, which "
                             "nothing ever supplies")

    def test_disabled_generator_skipped(self):
        facts = ootb.Facts("m", 1, set(), 3, effort="base")
        d = ootb.Decision(run_ef=True, ef_reason="min_ranks")
        pol = ootb.load_policy()
        pol["suggestions"]["disabled"] = ["_sg_ran_ef_few_ranks"]
        msgs = ootb.make_suggestions(d, facts, pol)
        self.assertFalse(any("only 1 MPI" in m for m in msgs))

    def test_no_class_solver_suggestion(self):
        # MINLP with nothing installed -> "install one or pass --solver-name"
        facts = ootb.Facts("m", 6, {"gurobi"}, 10, effort="base",
                           vars_int=3, vars_cont=5, model_degree="nonlinear")
        d = ootb.Decision(run_ef=False, chosen_solver=None, problem_class="MINLP")
        self.assertTrue(any("MINLP" in m for m in self._msgs(d, facts)))


def _tiny_model(kind):
    """A one-scenario-shaped model whose objective/constraint degree we control."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.y = pyo.Var(bounds=(0, 10))
    m.c = pyo.Constraint(expr=m.x + m.y <= 5)          # linear unless overridden
    if kind == "linear":
        m.o = pyo.Objective(expr=m.x + 2 * m.y)
    elif kind == "quadratic":
        m.o = pyo.Objective(expr=m.x ** 2 + m.y)
    elif kind == "cubic":
        m.o = pyo.Objective(expr=m.x ** 3 + m.y)
    elif kind == "nonpoly":
        m.o = pyo.Objective(expr=pyo.log(m.x + 1) + m.y)
    elif kind == "quad_constraint":
        m.o = pyo.Objective(expr=m.x + m.y)
        m.c2 = pyo.Constraint(expr=m.x * m.y <= 4)     # degree comes from a con
    return m


class TestModelDegreeAndClass(unittest.TestCase):
    def test_model_degree(self):
        self.assertEqual(ootb._model_degree(_tiny_model("linear")), "linear")
        self.assertEqual(ootb._model_degree(_tiny_model("quadratic")), "quadratic")
        self.assertEqual(ootb._model_degree(_tiny_model("cubic")), "nonlinear")
        self.assertEqual(ootb._model_degree(_tiny_model("nonpoly")), "nonlinear")
        # a quadratic CONSTRAINT (linear objective) still makes the model QP
        self.assertEqual(ootb._model_degree(_tiny_model("quad_constraint")),
                         "quadratic")

    def test_problem_class_mapping(self):
        def pc(degree, vint):
            return ootb._problem_class(
                ootb.Facts("m", 3, set(), 10, vars_int=vint, model_degree=degree))
        self.assertEqual(pc("linear", 0), "LP")
        self.assertEqual(pc("quadratic", 0), "QP")
        self.assertEqual(pc("nonlinear", 0), "NLP")
        self.assertEqual(pc("linear", 5), "MIP")
        self.assertEqual(pc("quadratic", 5), "MIQP")
        self.assertEqual(pc("nonlinear", 5), "MINLP")
        # minus tier: not instantiated -> class unknown
        self.assertIsNone(ootb._problem_class(ootb.Facts("m", 3, set(), 10)))


class TestSolverRoutingByClass(unittest.TestCase):
    def setUp(self):
        self.policy = ootb.load_policy()

    def _rec(self, degree, vars_int, available):
        facts = ootb.Facts("m", 6, set(available), 10, effort="base",
                           vars_int=vars_int, vars_cont=5, model_degree=degree)
        return ootb.recommend(facts, self.policy)

    def test_nonlinear_continuous_routes_to_ipopt(self):
        d = self._rec("nonlinear", 0, {"gurobi", "ipopt"})
        self.assertEqual(d.problem_class, "NLP")
        self.assertEqual(d.chosen_solver, "ipopt")     # not gurobi

    def test_nonlinear_without_nlp_solver_picks_nothing(self):
        # a MIP solver must NOT be chosen for a nonlinear model
        d = self._rec("nonlinear", 0, {"gurobi", "cbc"})
        self.assertEqual(d.problem_class, "NLP")
        self.assertIsNone(d.chosen_solver)

    def test_integer_model_never_routes_to_ipopt(self):
        d = self._rec("linear", 5, {"ipopt", "cbc"})
        self.assertEqual(d.problem_class, "MIP")
        self.assertEqual(d.chosen_solver, "cbc")       # ipopt can't do integers

    def test_integer_nonlinear_is_minlp(self):
        d = self._rec("nonlinear", 3, {"gurobi", "ipopt"})
        self.assertEqual(d.problem_class, "MINLP")
        self.assertIsNone(d.chosen_solver)             # no MINLP solver installed

    def test_user_solver_wins_over_class_routing(self):
        facts = ootb.Facts("m", 6, {"gurobi", "ipopt"}, 10, effort="base",
                           vars_int=0, vars_cont=5, model_degree="nonlinear",
                           user_solver_name="cbc",
                           user_flags={"--solver-name"})
        d = ootb.recommend(facts, self.policy)
        self.assertEqual(d.chosen_solver, "cbc")       # OOTB defers to the user


class TestMultistageBundleSizing(unittest.TestCase):
    """A multistage bundle must consume whole second-stage nodes.

    ProperBundler.set_bunBFs raises "Bundles must consume the same number of
    entire second stage nodes" unless scenarios_per_bundle is a multiple of
    prod(branching_factors[1:]), so a size that only divides num_scens aborts
    the run at startup.
    """

    def setUp(self):
        self.policy = ootb.load_policy()

    @staticmethod
    def _facts(bf, num_scens, **kw):
        return ootb.Facts("m", 6, {"gurobi"}, num_scens, effort="base",
                          vars_int=0, vars_cont=2, nonants_int=0,
                          model_degree="linear", multistage=bf is not None,
                          branching_factors=bf, **kw)

    def test_multiple_of_beyond2size(self):
        self.assertEqual(ootb._spb_multiple_of(self._facts([4, 3, 3, 2], 72)), 18)
        # two-stage is unconstrained
        self.assertEqual(ootb._spb_multiple_of(self._facts(None, 72)), 1)

    def test_sizer_only_offers_whole_node_multiples(self):
        """A budget under which the ONLY affordable sizes are illegal ones.

        With a generous budget the sizer returns 72, which is a multiple of 18
        anyway -- that assertion passes with the constraint deleted and proves
        nothing. At 10x, the legal sizes (18, 36, 72) are all too expensive, so
        the unconstrained sizer would return an illegal 9 and the constrained
        one must decline.
        """
        facts = self._facts([4, 3, 3, 2], 72)
        scaling = self.policy["effort_scaling"]
        self.assertIsNone(
            ootb._pick_spb_by_effort(72, 1, facts, scaling, 10.0,
                                     multiple_of=18))
        # unconstrained, the same call picks a size set_bunBFs would reject
        illegal = ootb._pick_spb_by_effort(72, 1, facts, scaling, 10.0,
                                           multiple_of=1)
        self.assertIsNotNone(illegal)
        self.assertNotEqual(illegal % 18, 0)

    def test_degenerate_branching_factor_does_not_divide_by_zero(self):
        facts = self._facts([10, 0], 120)
        self.assertEqual(ootb._spb_multiple_of(facts), 1)
        # recommend() must not raise either (num_scens beats the bad bf)
        ootb.recommend(facts, self.policy)

    def test_recommend_never_emits_an_illegal_bundle_size(self):
        """Guards the shipped behavior, not just the helper.

        The sizing tests below call _pick_spb_by_effort directly and pass
        multiple_of themselves, so they stay green even if recommend() stops
        passing it -- deleting `multiple_of=mult` at the call site left the
        whole suite passing. This exercises recommend() with the shipped
        policy: wired it emits no bundle size here, unwired it emits 2, which
        set_bunBFs rejects (2 % 18 != 0).
        """
        from mpisppy.utils.proper_bundler import ProperBundler
        bf = [4, 3, 3, 2]
        facts = ootb.Facts(
            "m", 6, {"gurobi"}, 72, effort="base", vars_int=200, vars_cont=500,
            nonants_int=50, nonants_total=100, model_degree="linear",
            multistage=True, branching_factors=bf,
            user_flags={"--lagrangian"},        # force decomposition
        )
        d = ootb.recommend(facts, self.policy)
        self.assertFalse(d.run_ef)
        emitted = [a.value for a in d.args
                   if a.flag == "--scenarios-per-bundle"]
        for spb in emitted:                     # whatever it chose must be usable
            cfg = config.Config()
            cfg.add_branching_factors()
            cfg.proper_bundle_config()
            cfg.branching_factors = bf
            cfg.scenarios_per_bundle = int(spb)
            ProperBundler.set_bunBFs(ProperBundler.__new__(ProperBundler), cfg)

    def test_chosen_size_survives_set_bunBFs(self):
        """The end-to-end property: whatever OOTB picks, the run can use it."""
        from mpisppy.utils.proper_bundler import ProperBundler
        bf = [4, 3, 3, 2]
        facts = self._facts(bf, 72)
        scaling = self.policy["effort_scaling"]
        for max_hardness in (1e9, 100.0, 10.0):
            spb = ootb._pick_spb_by_effort(
                72, 1, facts, scaling, max_hardness,
                multiple_of=ootb._spb_multiple_of(facts))
            if spb is None:
                continue                       # unbundled is always safe
            cfg = config.Config()
            cfg.add_branching_factors()
            cfg.proper_bundle_config()
            cfg.branching_factors = bf
            cfg.scenarios_per_bundle = spb
            # raises RuntimeError if spb is not a whole number of stage-2 nodes
            ProperBundler.set_bunBFs(ProperBundler.__new__(ProperBundler), cfg)


class TestUserOptionsWin(unittest.TestCase):
    """What the user set, and how the equivalent command line reports it."""

    def setUp(self):
        self.policy = ootb.load_policy()
        self._argv = sys.argv

    def tearDown(self):
        sys.argv = self._argv

    def test_abbreviated_flag_counts_as_user_set(self):
        """argparse accepts unambiguous prefixes; an argv scan does not.

        --solver-nam sets solver_name, but scanning argv yields the token
        "--solver-nam", which matches no flag OOTB knows, so OOTB decided the
        user had chosen no solver and overwrote it.
        """
        cfg, _ = _farmer_cfg(solver_name="cbc")
        sys.argv = ["prog", "--module-name", "farmer", "--solver-nam", "cbc"]
        self.assertIn("--solver-name", ootb._user_flags(cfg))
        # the argv fallback is what got this wrong
        self.assertNotIn("--solver-name", ootb._user_flags(None))

    def test_command_line_echoes_user_flags(self):
        """The printed line has to RUN; the docs tell people to paste it."""
        cfg, _ = _farmer_cfg(solver_name="cbc", lagrangian=True)
        args = ootb._user_args(cfg)
        flags = dict(args)
        self.assertEqual(flags.get("--solver-name"), "cbc")
        self.assertIn("--lagrangian", flags)
        # OOTB's own flags and the anchors are not echoed twice
        for absent in ("--out-of-the-box", "--module-name", "--num-scens"):
            self.assertNotIn(absent, flags)
        facts = ootb.Facts("farmer", 3, {"cbc"}, 6, user_args=args)
        line = ootb.Decision().command_line(facts)
        self.assertIn("--solver-name cbc", line)
        self.assertIn("--lagrangian", line)

    def test_inspect_only_rank_count_is_validated(self):
        for bad in ("foo", "0", "-4"):
            cfg, _ = _farmer_cfg(inspect_only=bad)
            with self.assertRaises(RuntimeError):
                ootb._inspect_ranks(cfg)


class TestRankFloorAndRoster(unittest.TestCase):
    """An explicit decomposition request vs. the policy's rank floor."""

    def setUp(self):
        self.policy = ootb.load_policy()

    @staticmethod
    def _facts(num_ranks, user_flags):
        return ootb.Facts("m", num_ranks, {"gurobi"}, 6, effort="base",
                          vars_int=0, vars_cont=5, nonants_int=0,
                          model_degree="linear", user_flags=set(user_flags))

    def test_explicit_request_beats_the_rank_floor(self):
        """hub + one spoke on 2 ranks is a configuration mpi-sppy runs."""
        d = ootb.recommend(self._facts(2, {"--lagrangian"}), self.policy)
        self.assertFalse(d.run_ef)
        self.assertEqual(d.num_cylinders, 2)

    def test_request_that_does_not_fit_says_so(self):
        d = ootb.recommend(self._facts(1, {"--lagrangian"}), self.policy)
        self.assertTrue(d.run_ef)
        self.assertTrue(any("only 1 are available" in n for n in d.notes))

    def test_roster_never_exceeds_available_ranks(self):
        """Every cylinder needs a rank; apportion_ranks raises otherwise."""
        for n in (2, 3, 4):
            d = ootb.recommend(self._facts(n, {"--lagrangian"}), self.policy)
            if not d.run_ef:
                self.assertLessEqual(d.num_cylinders, n)

    def test_incumbent_options_dropped_without_an_inner_spoke(self):
        """--grad-rho and dynamic rho read BEST_XHAT; with no inner-bound
        spoke to publish one the run dies in do_decomp."""
        d = ootb.recommend(self._facts(2, {"--lagrangian"}), self.policy)
        emitted = {a.flag for a in d.args}
        self.assertFalse(emitted & ootb.NEEDS_INNER_BOUND)


def _ladder_flags():
    return {r["flag"] for r in ootb.load_policy()["spoke_ladder"]["rungs"]}


class TestCopilotReviewFindings(unittest.TestCase):
    """Cases from the Copilot review of PR #779."""

    @staticmethod
    def _facts(**kw):
        base = dict(vars_int=0, vars_cont=50, nonants_total=20, nonants_int=0,
                    model_degree="linear", effort="base")
        base.update(kw)
        ranks = base.pop("ranks", 6)
        solvers = base.pop("solvers", {"gurobi"})
        return ootb.Facts("farmer", ranks, solvers, 1000, **base)

    def test_highs_is_not_offered_for_an_miqp(self):
        # phbase.py: "HiGHS, which cannot solve an MIQP".
        miqp = ootb.load_policy()["solver"]["preference_order_by_class"]["MIQP"]
        self.assertNotIn("highs", miqp)
        self.assertNotIn("appsi_highs", miqp)

    def test_highs_on_an_integer_model_linearizes_the_prox(self):
        # On the decomposition path the prox makes every MIP subproblem an
        # MIQP, so highs would fail on the first proximal solve.
        d = ootb.recommend(self._facts(solvers={"highs"}, vars_int=50,
                                       nonants_int=10),
                           ootb.load_policy())
        self.assertFalse(d.run_ef)
        self.assertEqual(d.chosen_solver, "highs")
        self.assertIn("--linearize-proximal-terms", [a.flag for a in d.args])

    def test_ef_solver_override_does_not_drive_the_decomposition(self):
        # --EF-solver-name governs the EF; --solver-name the decomposition.
        # --lagrangian forces the decomposition so the branch under test runs
        facts = self._facts(ranks=6, solvers={"gurobi", "cplex"},
                            user_flags={"--EF-solver-name", "--lagrangian"},
                            user_ef_solver_name="cplex")
        d = ootb.recommend(facts, ootb.load_policy())
        self.assertFalse(d.run_ef)
        self.assertNotEqual(d.chosen_solver, "cplex",
                            msg="an EF-only override chose the PH solver")
        self.assertIn("--solver-name", [a.flag for a in d.args])

    def test_ef_solver_override_is_honored_when_the_ef_is_certain(self):
        facts = self._facts(ranks=1, solvers={"gurobi", "cplex"},
                            user_flags={"--EF-solver-name"},
                            user_ef_solver_name="cplex")
        d = ootb.recommend(facts, ootb.load_policy())
        self.assertTrue(d.run_ef)
        self.assertEqual(d.chosen_solver, "cplex")

    def test_ef_override_never_drives_a_decomposition_below_the_floor(self):
        # The EF gate skips the EF for an explicit decomposition request
        # BEFORE it consults the rank floor, so a floor test alone said "EF
        # for sure" for a run that decomposes -- reinstating the original bug.
        facts = self._facts(ranks=2, solvers={"gurobi", "cplex"},
                            user_flags={"--EF-solver-name", "--lagrangian"},
                            user_ef_solver_name="cplex")
        d = ootb.recommend(facts, ootb.load_policy())
        self.assertFalse(d.run_ef)
        self.assertNotEqual(d.chosen_solver, "cplex")
        self.assertNotIn(("--solver-name", "cplex"),
                         [(a.flag, a.value) for a in d.args])

    def test_miqp_model_warns_instead_of_linearizing(self):
        # The model's OWN objective is quadratic with integers, so linearizing
        # the prox changes nothing and the solve still fails.
        facts = self._facts(ranks=6, solvers={"highs"}, vars_int=50,
                            nonants_int=10, model_degree="quadratic",
                            user_flags={"--solver-name", "--lagrangian"},
                            user_solver_name="highs")
        d = ootb.recommend(facts, ootb.load_policy())
        self.assertEqual(d.problem_class, "MIQP")
        self.assertNotIn("--linearize-proximal-terms", [a.flag for a in d.args])
        self.assertTrue(any(n.startswith("WARNING") and "MIQP" in n
                            for n in d.notes))

    def test_minus_tier_linearizes_for_a_no_miqp_solver(self):
        # Integrality is unknown with no size profile; linearizing is safe,
        # and not doing it kills an integer model at PH iteration 1.
        facts = ootb.Facts("m", 6, {"highs"}, 1000, effort="minus",
                           user_flags={"--solver-name", "--lagrangian"},
                           user_solver_name="highs")
        d = ootb.recommend(facts, ootb.load_policy())
        self.assertIsNone(d.problem_class)
        self.assertIn("--linearize-proximal-terms", [a.flag for a in d.args])

    def test_lp_mip_only_solver_warns_on_a_quadratic_model(self):
        # cbc cannot take a quadratic objective AT ALL, so linearizing the
        # prox cannot rescue a model whose own objective is quadratic --
        # continuous QP included, which is why this is wider than the highs
        # case. Reporting a fix for a run that still dies is the bug.
        for deg, vi, cls in (("quadratic", 50, "MIQP"), ("quadratic", 0, "QP")):
            with self.subTest(cls=cls):
                facts = self._facts(ranks=6, solvers={"cbc"}, vars_int=vi,
                                    nonants_int=10 if vi else 0,
                                    model_degree=deg,
                                    user_flags={"--solver-name", "--lagrangian"},
                                    user_solver_name="cbc")
                d = ootb.recommend(facts, ootb.load_policy())
                self.assertEqual(d.problem_class, cls)
                self.assertNotIn("--linearize-proximal-terms",
                                 [a.flag for a in d.args])
                self.assertTrue(any(n.startswith("WARNING") for n in d.notes))
                self.assertFalse(
                    any("avoids the approximation" in m for m in
                        ootb.make_suggestions(d, facts, ootb.load_policy())),
                    msg="suggested a fix for a model the solver cannot solve")

    def test_lp_mip_only_solver_still_linearizes_for_lp_and_mip(self):
        for deg, vi, cls in (("linear", 0, "LP"), ("linear", 50, "MIP")):
            with self.subTest(cls=cls):
                facts = self._facts(ranks=6, solvers={"cbc"}, vars_int=vi,
                                    nonants_int=10 if vi else 0,
                                    model_degree=deg,
                                    user_flags={"--solver-name", "--lagrangian"},
                                    user_solver_name="cbc")
                d = ootb.recommend(facts, ootb.load_policy())
                self.assertIn("--linearize-proximal-terms",
                              [a.flag for a in d.args])

    def test_unsolvable_class_warning_names_the_right_solvers(self):
        # gurobi/cplex/xpress cannot solve a general NLP/MINLP either, so a
        # hard-coded list was wrong advice; the policy already knows.
        policy = ootb.load_policy()
        for deg, cls in (("nonlinear", "NLP"), ("quadratic", "QP")):
            with self.subTest(cls=cls):
                facts = self._facts(ranks=6, solvers={"cbc"},
                                    model_degree=deg,
                                    user_flags={"--solver-name", "--lagrangian"},
                                    user_solver_name="cbc")
                d = ootb.recommend(facts, policy)
                self.assertEqual(d.problem_class, cls)
                warn = [n for n in d.notes if n.startswith("WARNING")]
                self.assertTrue(warn)
                preferred = policy["solver"]["preference_order_by_class"][cls]
                self.assertIn(preferred[0], warn[0],
                              msg=f"warning for {cls} does not name {preferred[0]}")
                self.assertNotIn("quadratic objective", warn[0])

    def test_late_ef_fallback_drops_every_decomposition_note(self):
        # Matching only the notes choose() wrote left the prox warning, the
        # roster-trim line and the solver deferral on an EF trace.
        facts = self._facts(ranks=3, solvers={"cbc"}, vars_int=50,
                            nonants_int=10, model_degree="quadratic",
                            user_flags={"--solver-name", "--lagrangian",
                                        "--fwph", "--grad-rho"},
                            user_solver_name="cbc")
        d = ootb.recommend(facts, ootb.load_policy())
        self.assertTrue(d.run_ef)
        for n in d.notes:
            self.assertNotIn("PH prox", n)
            self.assertNotIn("spoke roster trimmed", n)
            self.assertFalse(n.startswith("--solver-name"))

    def test_late_ef_fallback_drops_decomposition_choices(self):
        # The incumbent-spoke gate can pick the EF after the decomposition
        # path already chose a solver, a prox linearization and spokes. An EF
        # run has no PH prox and reads EF_solver_name, so the trace and the
        # equivalent command line must not advertise a decomposition.
        facts = self._facts(ranks=3, solvers={"gurobi"},
                            user_flags={"--grad-rho", "--lagrangian", "--fwph"})
        d = ootb.recommend(facts, ootb.load_policy())
        self.assertTrue(d.run_ef)
        flags = [a.flag for a in d.args]
        self.assertNotIn("--solver-name", flags)
        self.assertNotIn("--linearize-proximal-terms", flags)
        self.assertIn("--EF-solver-name", flags)
        cl = d.command_line(facts)
        self.assertNotIn("--solver-name", cl)

    def test_fwph_objgap_hub_is_a_decomposition_request(self):
        self.assertIn("--fwph-objgap-hub", ootb.HUB_FLAGS)
        self.assertIn("--fwph-objgap-hub", ootb.DECOMPOSITION_FLAGS)
        d = ootb.recommend(self._facts(ranks=6,
                                       user_flags={"--fwph-objgap-hub"}),
                           ootb.load_policy())
        self.assertFalse(d.run_ef, msg="OOTB substituted the EF for a hub the "
                                       "user asked for")


class TestSpokeBuildOrderIsShared(unittest.TestCase):
    """OOTB must model the rank split in the order the run launches spokes.

    apportion_ranks breaks largest-remainder ties by position, so a different
    order gives each cylinder the right total under the wrong NAME -- the
    "rank split" note would tell the user a cylinder got ranks that went
    somewhere else. spokes.py assembles from the same tuple so the order is
    declared once.
    """

    def test_order_covers_every_spoke_flag(self):
        self.assertEqual(set(ootb.SPOKE_BUILD_ORDER), set(ootb.SPOKE_FLAGS))
        self.assertEqual(len(ootb.SPOKE_BUILD_ORDER),
                         len(set(ootb.SPOKE_BUILD_ORDER)))

    def test_build_spoke_list_assembles_from_the_tuple(self):
        # Both directions, from the source rather than from one run: the
        # runtime guard in spokes.py only sees the spokes that run enabled, so
        # a spoke added there but omitted from the tuple would stay invisible
        # until someone switched it on.
        import inspect
        import re
        from mpisppy.generic import spokes
        src = inspect.getsource(spokes.build_spoke_list)
        self.assertIn("SPOKE_BUILD_ORDER", src)
        built = set(re.findall(r'made\["(--[a-z-]+)"\]', src))
        self.assertTrue(built, "could not find the spokes build_spoke_list makes")
        self.assertEqual(built, set(ootb.SPOKE_BUILD_ORDER),
                         msg="build_spoke_list and SPOKE_BUILD_ORDER disagree "
                             "about which spokes exist")

    def test_modelled_split_follows_build_order(self):
        for flags, ranks in ((["--ph-dual"], 16), (["--relaxed-ph"], 12),
                             (["--ph-dual", "--xhatlshaped"], 16), ([], 16)):
            with self.subTest(flags=flags, ranks=ranks):
                facts = ootb.Facts("farmer", ranks, {"gurobi"}, 1000,
                                   effort="base", vars_int=50, vars_cont=50,
                                   nonants_total=20, nonants_int=10,
                                   model_degree="linear", user_flags=set(flags))
                d = ootb.recommend(facts, ootb.load_policy())
                got = [n for n in d.rank_split if n != "(hub)"]
                self.assertEqual(
                    got, [f for f in ootb.SPOKE_BUILD_ORDER if f in got])


class TestOffLadderSpokesAreInTheRankModel(unittest.TestCase):
    """A spoke the user set is a cylinder even when the policy ladder does not
    list it, so it has to be in the rank split. It was not: every rank was
    apportioned across the ladder cylinders while the run launched more, the
    printed split omitted the user's own spoke, and intra_ranks -- hence the
    bundle floor -- described a split that never happens."""

    @staticmethod
    def _decide(flags, num_ranks):
        facts = ootb.Facts("farmer", num_ranks, {"gurobi"}, 1000, effort="base",
                           vars_int=50, vars_cont=50, nonants_total=20,
                           nonants_int=10, model_degree="linear",
                           user_flags=set(flags))
        return ootb.recommend(facts, ootb.load_policy())

    def test_split_covers_every_cylinder(self):
        off = sorted(ootb.SPOKE_FLAGS - _ladder_flags())
        self.assertTrue(off, "no off-ladder spoke to test with")
        for flags, ranks in ((off[:1], 6), (off[:2], 8), (off, 8)):
            with self.subTest(flags=flags, ranks=ranks):
                d = self._decide(flags, ranks)
                # the property: one entry per cylinder the run will launch,
                # every requested spoke named, and no rank unaccounted for
                self.assertEqual(len(d.rank_split), d.num_cylinders)
                for f in flags:
                    self.assertIn(f, d.rank_split)
                self.assertEqual(sum(d.rank_split.values()), ranks)
                self.assertEqual(d.intra_ranks, max(d.rank_split.values()))

    def test_off_ladder_spoke_gets_the_default_ratio(self):
        off = sorted(ootb.SPOKE_FLAGS - _ladder_flags())[0]
        policy = ootb.load_policy()
        default = policy["rank_allocation"]["default_rank_ratio"]
        d = self._decide([off], 8)
        # no --<flag>-rank-ratio is emitted: it is the default, and for an
        # off-ladder spoke that option may not even be declared.
        self.assertNotIn(f"{off}-rank-ratio", [a.flag for a in d.args])
        # a full share: at the default ratio it is not starved relative to a
        # ladder spoke that also has the default.
        hub = d.rank_split["(hub)"]
        self.assertGreaterEqual(d.rank_split[off], hub if default == 1.0 else 1)

    def test_user_set_ratio_is_recorded(self):
        # "Every decision records why" covers the ones OOTB declines to make.
        facts = ootb.Facts("farmer", 6, {"gurobi"}, 1000, effort="base",
                           vars_int=50, vars_cont=50, nonants_total=20,
                           nonants_int=10, model_degree="linear",
                           user_flags={"--xhatshuffle-rank-ratio"},
                           user_args=[("--xhatshuffle-rank-ratio", "0.5")])
        d = ootb.recommend(facts, ootb.load_policy())
        self.assertTrue(any("--xhatshuffle-rank-ratio" in n and "defers" in n
                            for n in d.notes),
                        msg="the user's rank ratio left no trace")

    def test_unreadable_user_ratio_leaves_a_trace(self):
        # user_flags names the ratio but user_args has no value for it: OOTB
        # models the policy value, and must say so rather than fall silent.
        facts = ootb.Facts("farmer", 6, {"gurobi"}, 1000, effort="base",
                           vars_int=50, vars_cont=50, nonants_total=20,
                           nonants_int=10, model_degree="linear",
                           user_flags={"--xhatshuffle-rank-ratio"},
                           user_args=[])
        d = ootb.recommend(facts, ootb.load_policy())
        self.assertTrue(any("--xhatshuffle-rank-ratio" in n and "WARNING" in n
                            for n in d.notes),
                        msg="an unreadable ratio left no trace")

    def test_rejected_user_ratio_is_not_called_a_deferral(self):
        # The warning says OOTB is modelling the policy value; a "kept user's
        # value" note beside it would contradict that in the same trace.
        facts = ootb.Facts("farmer", 6, {"gurobi"}, 1000, effort="base",
                           vars_int=50, vars_cont=50, nonants_total=20,
                           nonants_int=10, model_degree="linear",
                           user_flags={"--ph-dual", "--ph-dual-rank-ratio"},
                           user_args=[("--ph-dual-rank-ratio", "0")])
        d = ootb.recommend(facts, ootb.load_policy())
        self.assertTrue(any("WARNING" in n and "--ph-dual-rank-ratio" in n
                            for n in d.notes))
        self.assertFalse(any("kept user's value" in n
                             and "--ph-dual-rank-ratio" in n for n in d.notes))

    def test_non_positive_user_ratio_is_reported_not_a_traceback(self):
        # apportion_ranks refuses a non-positive ratio and recommend() is not
        # wrapped, so passing one through killed the run with a bare traceback.
        facts = ootb.Facts("farmer", 6, {"gurobi"}, 1000, effort="base",
                           vars_int=50, vars_cont=50, nonants_total=20,
                           nonants_int=10, model_degree="linear",
                           user_flags={"--ph-dual", "--ph-dual-rank-ratio"},
                           user_args=[("--ph-dual-rank-ratio", "0")])
        d = ootb.recommend(facts, ootb.load_policy())      # must not raise
        self.assertTrue(any("--ph-dual-rank-ratio" in n and "WARNING" in n
                            for n in d.notes),
                        msg="no warning for a non-positive rank ratio")

    def test_unemitted_policy_ratio_is_not_modelled(self):
        # The RUN sees only the flags OOTB emits, never the policy file, and
        # OOTB emits a -rank-ratio only for a ladder rung. So a policy ratio on
        # an off-ladder spoke has no effect on the run, and modelling it would
        # put a split in the note (and an intra_ranks in the bundle floor) that
        # never happens.
        policy = copy.deepcopy(ootb.load_policy())
        # the premise: --ph-dual is NOT a ladder rung, so OOTB never emits a
        # ratio for it. If a later policy promotes it, say so here rather than
        # failing below with a message about unemitted ratios.
        self.assertNotIn("--ph-dual",
                         {r["flag"] for r in policy["spoke_ladder"]["rungs"]})
        policy["rank_allocation"]["rank_ratios"]["--ph-dual"] = 0.2
        facts = ootb.Facts("farmer", 12, {"gurobi"}, 1000, effort="base",
                           vars_int=50, vars_cont=50, nonants_total=20,
                           nonants_int=10, model_degree="linear",
                           user_flags={"--ph-dual"})
        d = ootb.recommend(facts, policy)
        self.assertNotIn("--ph-dual-rank-ratio", [a.flag for a in d.args])
        # a 0.2 share of 12 ranks would be 1; the default share is more
        self.assertGreater(d.rank_split["--ph-dual"], 1)
        self.assertEqual(d.intra_ranks, max(d.rank_split.values()))

    def test_widening_gate_counts_every_user_spoke(self):
        # A user-set LADDER rung beyond the widening loop's reach is
        # force-added afterwards, and an off-ladder spoke never enters
        # `chosen` at all; both spend rank budget the gate has to see.
        # --reduced-costs and --subgradient are rungs, --ph-dual is not.
        policy = ootb.load_policy()
        min_rpc = policy["rank_allocation"]["min_ranks_per_cylinder"]
        for flag in ("--reduced-costs", "--subgradient", "--ph-dual"):
            with self.subTest(flag=flag):
                facts = ootb.Facts("farmer", 12, {"gurobi"}, 1000,
                                   effort="base", vars_int=50, vars_cont=50,
                                   nonants_total=20, nonants_int=10,
                                   model_degree="linear", user_flags={flag})
                d = ootb.recommend(facts, policy)
                self.assertGreaterEqual(12 // d.num_cylinders, min_rpc,
                                        msg=f"{d.num_cylinders} cylinders for "
                                            f"12 ranks starves the roster")

    def test_split_is_deterministic(self):
        # off_ladder is a set; iterating it directly would make the split
        # depend on PYTHONHASHSEED.
        flags = sorted(ootb.SPOKE_FLAGS - _ladder_flags())
        first = list(self._decide(flags, 8).rank_split.items())
        for _ in range(5):
            self.assertEqual(list(self._decide(flags, 8).rank_split.items()),
                             first)


class TestRankLayoutMirrorsTheRun(unittest.TestCase):
    """_rank_layout must model what WheelSpinner actually does."""

    def test_branches_on_any_ratio_not_all_equal(self):
        """spin_the_wheel branches on any(r != 1.0). Uniform-but-not-1.0
        ratios apportion at run time, so modeling an equal split is wrong."""
        # 7 ranks is the discriminating case: apportionment spends the
        # remainder ([3, 2, 2]) while the equal split drops it ([2, 2, 2]),
        # so intra_ranks -- and the bundle floor it drives -- differ. At 6
        # ranks both paths give [2, 2, 2] and the test proves nothing.
        from mpisppy.utils.rank_apportionment import apportion_ranks
        ratios = [2.0, 2.0, 2.0]
        intra, split, _ = ootb._rank_layout(7, ratios, ["--a", "--b"])
        self.assertEqual(sorted(split.values()),
                         sorted(apportion_ranks(ratios, 7)))
        self.assertEqual(intra, 3)          # the equal split would say 2
        self.assertEqual(sum(split.values()), 7)

    def test_reports_indivisible_equal_split(self):
        """_make_comms raises "Need a multiple of N processes"."""
        _, _, divisible = ootb._rank_layout(5, [1.0, 1.0, 1.0],
                                            ["--a", "--b"])
        self.assertFalse(divisible)
        _, _, divisible = ootb._rank_layout(6, [1.0, 1.0, 1.0],
                                            ["--a", "--b"])
        self.assertTrue(divisible)


class TestPersistentSolverSuggestion(unittest.TestCase):
    def setUp(self):
        self.policy = ootb.load_policy()

    def _msg(self, solver, available=()):
        d = ootb.Decision(run_ef=False, chosen_solver=solver)
        facts = ootb.Facts("m", 6, set(available), 6)
        return ootb._sg_no_persistent_solver(d, facts, self.policy, None)

    def test_no_dead_end_advice(self):
        """cbc_persistent/glpk_persistent are not Pyomo solvers."""
        for solver in ("cbc", "glpk", "ipopt", "highs", "appsi_highs"):
            self.assertIsNone(self._msg(solver), f"bad advice for {solver}")

    def test_suggests_a_real_persistent_interface(self):
        self.assertIn("gurobi_persistent", self._msg("gurobi"))
        self.assertIsNone(self._msg("gurobi_persistent"))


class TestRosterFitsTheRanks(unittest.TestCase):
    """Count the spokes the RUN builds, not OOTB's own tally.

    d.num_cylinders was itself wrong, so asserting on it hid a regression:
    build_spoke_list appends on cfg.<spoke> whoever set it, so a user spoke
    OOTB did not pick is still a cylinder.
    """

    def setUp(self):
        self.policy = ootb.load_policy()

    @staticmethod
    def _built(flags, num_ranks):
        module = _farmer_module()
        facts = ootb.Facts("farmer", num_ranks, {"gurobi"}, 60, effort="base",
                           vars_int=0, vars_cont=5, nonants_int=0,
                           model_degree="linear", user_flags=set(flags))
        d = ootb.recommend(facts, ootb.load_policy())
        cfg = config.Config()
        parsing.add_driver_args(cfg, module)
        cfg.num_scens = 60
        cfg.module_name = "farmer"
        for f in flags:
            key = f[2:].replace("-", "_")
            if key in cfg:
                cfg[key] = True
        ootb.apply_decision(d, cfg)
        built = sum(1 for s in ootb.SPOKE_FLAGS
                    if cfg.get(s[2:].replace("-", "_"), ifmissing=False))
        return d, 1 + built

    def test_every_roster_fits_and_the_count_is_honest(self):
        combos = [set(), {"--lagrangian"}, {"--xhatxbar"}, {"--fwph"},
                  {"--ph-dual"}, {"--relaxed-ph"}, {"--subgradient"},
                  {"--reduced-costs"}, {"--xhatshuffle"},
                  {"--lagrangian", "--fwph"},
                  {"--lagrangian", "--fwph", "--xhatxbar"}]
        for n in range(1, 7):
            for flags in combos:
                d, cylinders = self._built(flags, n)
                if d.run_ef:
                    continue
                with self.subTest(ranks=n, flags=sorted(flags)):
                    self.assertLessEqual(cylinders, n)
                    self.assertEqual(cylinders, d.num_cylinders)

    def test_user_incumbent_option_keeps_a_best_xhat_spoke(self):
        """--grad-rho reads BEST_XHAT; the roster must publish one or not
        decompose. Trimming the xhat spoke under it killed the run."""
        for n in (2, 3, 4, 6):
            # both cases carry a real rho setter: --dynamic-rho-primal-crit
            # WITHOUT one is rejected by parse_args' own checker, so it can
            # never reach OOTB and is not a configuration worth asserting on.
            for flags in ({"--lagrangian", "--grad-rho"},
                          {"--fwph", "--grad-rho"}):
                d, _ = self._built(flags, n)
                if d.run_ef:
                    continue
                emitted = {a.flag for a in d.args} | set(flags)
                with self.subTest(ranks=n, flags=sorted(flags)):
                    self.assertTrue(emitted & ootb.BEST_XHAT_SPOKES)

    def test_user_xhat_spoke_keeps_ootb_rho_setter(self):
        """A user's own xhat spoke publishes BEST_XHAT, so OOTB need not drop
        --grad-rho.

        --xhatlshaped is the discriminating choice: it is NOT on the policy
        ladder, so it never lands in `chosen`. A ladder spoke like --xhatxbar
        does, and then the roster carries a BEST_XHAT spoke either way, so the
        test would pass without consulting the user's flags at all.
        """
        d, _ = self._built({"--xhatlshaped"}, 2)
        self.assertFalse(d.run_ef)
        self.assertIn("--grad-rho", {a.flag for a in d.args})


class TestEchoedValuesAreRunnable(unittest.TestCase):
    def test_values_with_spaces_are_quoted(self):
        """--solver-options is a space-delimited string; unquoted, the echoed
        line made argparse reject the trailing words."""
        cfg, _ = _farmer_cfg(solver_options="mipgap=0.001 threads=2")
        value = dict(ootb._user_args(cfg))["--solver-options"]
        facts = ootb.Facts("farmer", 1, set(), 6,
                           user_args=ootb._user_args(cfg))
        line = ootb.Decision().command_line(facts)
        self.assertEqual(shlex.split(f"x {value}")[1],
                         "mipgap=0.001 threads=2")
        # the whole line survives shell splitting as one token per value
        self.assertIn("mipgap=0.001 threads=2", shlex.split(line))

    def test_implicit_module_inputs_are_not_echoed(self):
        """command_line() always emits --module-name, and model_fname()
        refuses that together with either of these."""
        for key in ("mps_files_directory", "smps_dir"):
            # these are declared by the problem_io module's inparser_adder,
            # not by add_driver_args, so add them the way that module does
            cfg, _ = _farmer_cfg()
            cfg.add_to_config(key, description=key, domain=str, default=None)
            cfg[key] = "/tmp/x"
            self.assertNotIn(ootb._cfg_key_to_flag(key),
                             dict(ootb._user_args(cfg)))

    def test_abbreviation_through_real_parse_args(self):
        """The premise of the cfg-over-argv fix: argparse accepts prefixes."""
        module = _farmer_module()
        argv = sys.argv
        try:
            sys.argv = ["prog", "--module-name", "farmer", "--num-scens", "3",
                        "--solver-nam", "cbc"]
            cfg = parsing.parse_args(module)
        finally:
            sys.argv = argv
        self.assertEqual(cfg.solver_name, "cbc")
        self.assertIn("--solver-name", ootb._user_flags(cfg))


class TestApplyDecisionValidates(unittest.TestCase):
    def test_checker_runs_on_what_ootb_built(self):
        """parse_args checked the config before OOTB existed."""
        cfg, _ = _farmer_cfg()
        d = ootb.Decision()
        d.args = [ootb.ChosenArg("--dynamic-rho-primal-crit", None, "x")]
        with self.assertRaises(ValueError):    # no automated rho setter
            ootb.apply_decision(d, cfg)

    def test_coeff_rho_supersedes_dynamic_rho_in_the_policy(self):
        """So OOTB defers instead of building a pair checker rejects."""
        policy = ootb.load_policy()
        sup = policy["option_categories"]["dynamic_rho"]["superseded_by"]
        for f in ("--coeff-rho", "--reduced-costs-rho"):
            self.assertIn(f, sup)


class TestUserRankRatioIsModeled(unittest.TestCase):
    def test_user_ratio_changes_the_split(self):
        """12 ranks, not 6: at 6 the policy and user ratios both give
        [2, 3, 1] and the test would prove nothing."""
        policy = ootb.load_policy()
        facts = ootb.Facts(
            "m", 12, {"gurobi"}, 60, effort="base", vars_int=0, vars_cont=5,
            nonants_int=0, model_degree="linear",
            user_flags={"--lagrangian", "--lagrangian-rank-ratio"},
            user_args=[("--lagrangian-rank-ratio", "3.0")])
        d = ootb.recommend(facts, policy)
        self.assertFalse(d.run_ef)
        base = ootb.recommend(
            ootb.Facts("m", 12, {"gurobi"}, 60, effort="base", vars_int=0,
                       vars_cont=5, nonants_int=0, model_degree="linear",
                       user_flags={"--lagrangian"}),
            policy)
        # the user's 3.0 widens --lagrangian and moves the bundle floor
        self.assertGreater(d.rank_split["--lagrangian"],
                           base.rank_split["--lagrangian"])
        self.assertNotEqual(d.intra_ranks, base.intra_ranks)


class TestProbeIsQuiet(unittest.TestCase):
    def test_missing_solvers_do_not_log(self):
        """Pyomo logs a WARNING plus a traceback per missing ASL solver."""
        import logging
        records = []

        class _Catch(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Catch()
        for name in ("pyomo.opt", "pyomo.solvers", "pyomo.common"):
            logging.getLogger(name).addHandler(handler)
        try:
            ootb._detect_available_solvers(["bonmin", "couenne", "not_a_solver"])
        finally:
            for name in ("pyomo.opt", "pyomo.solvers", "pyomo.common"):
                logging.getLogger(name).removeHandler(handler)
        self.assertEqual([], [r.getMessage() for r in records])

    def test_levels_are_restored(self):
        import logging
        lg = logging.getLogger("pyomo.opt")
        before = lg.level
        ootb._detect_available_solvers(["bonmin"])
        self.assertEqual(before, lg.level)


if __name__ == "__main__":
    unittest.main()
