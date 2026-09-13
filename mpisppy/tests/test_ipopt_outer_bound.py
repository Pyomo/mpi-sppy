###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Tests for the ipopt_outer_bound spoke.
#
#     python -m pytest mpisppy/tests/test_ipopt_outer_bound.py
#     mpiexec -np 2 python -m mpi4py -m pytest mpisppy/tests/test_ipopt_outer_bound.py
#
# The wiring tests need neither a solver nor MPI: option routing is the part
# most likely to break silently, and it is checkable by inspecting the spoke
# dict the factory builds. The end-to-end test needs both Ipopt and two ranks
# and skips cleanly without them.

import math
import unittest
import warnings

import pyomo.environ as pyo

import mpisppy.tests.examples.farmer as farmer
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.utils import config
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils.dual_certificate import CertificateError
from mpisppy.tests.utils import announce_hsl_if_used, get_solver

from mpi4py import MPI

comm = MPI.COMM_WORLD

ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)

if ipopt_available:
    announce_hsl_if_used()
mip_available, mip_solver_name, *_ = get_solver()


def _reports_dual_bound(name):
    """True if `name` actually fills in a dual bound on a solved LP.

    Being available is not enough, and neither is solving to optimality. cbc on
    the CI runner returns `status=ok, TerminationCondition=optimal` and leaves
    Problem[0].Lower_bound empty, so the Lagrangian spoke gets nothing to send
    and its reported bound stays nan. Asking the solver directly is the only
    honest test; anything else guesses.
    """
    try:
        if not pyo.SolverFactory(name).available(exception_flag=False):
            return False
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10), initialize=0.0)
        m.o = pyo.Objective(expr=2.0 * m.x, sense=pyo.minimize)
        m.c = pyo.Constraint(expr=m.x >= 1)
        results = pyo.SolverFactory(name).solve(m, load_solutions=False)
        bound = results.Problem[0].Lower_bound
        return bound is not None and float(bound) == float(bound)  # not NaN
    except Exception:
        return False


def _dual_bound_solver():
    """A solver that actually reports a dual bound, for the Lagrangian spoke.

    Ipopt reports none -- that is the entire reason this spoke exists -- so the
    comparison in TestAgreesWithLagrangian needs a second solver. cbc is tried
    as a fallback because it ships in the same idaes-ext bundle that supplies
    Ipopt, but it is only used if it demonstrably reports a bound here.
    """
    candidates = []
    if mip_available:
        # The persistent interfaces need set_instance before a solve, which the
        # probe below does not do, so try the plain name as well.
        candidates += [mip_solver_name, mip_solver_name.replace("_persistent", "")]
    # glpk and cbc are both plausible in CI: glpk is a small apt package and cbc
    # ships in the idaes-ext bundle alongside Ipopt. Neither is assumed to work
    # -- each is probed. (glpk cannot handle a PH proximal term, but the
    # Lagrangian spoke runs with attach_prox=False, so it is fine here.)
    candidates += ["glpk", "cbc"]
    for name in candidates:
        if name and _reports_dual_bound(name):
            return name
    return None


dual_bound_solver_name = _dual_bound_solver()

# The three-scenario farmer optimum, from an EF solve. farmer is linear, hence
# convex, so it is inside this spoke's assumptions and its bound must not exceed
# this value.
FARMER_EF_OPT = -108390.0


def _cfg(num_scens=3, hub_solver="ipopt"):
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.lagrangian_args()
    cfg.ipopt_outer_bound_args()
    cfg.num_scens = num_scens
    cfg.max_iterations = 5
    cfg.default_rho = 1.0
    cfg.solver_name = hub_solver
    return cfg


def _beans(cfg):
    all_scenario_names = farmer.scenario_names_creator(cfg.num_scens)
    kwargs = farmer.kw_creator(cfg)
    beans = (cfg, farmer.scenario_creator, farmer.scenario_denouement,
             all_scenario_names)
    return beans, kwargs


def _spoke_options(cfg, **kw):
    beans, kwargs = _beans(cfg)
    spoke = vanilla.ipopt_outer_bound_spoke(*beans,
                                            scenario_creator_kwargs=kwargs, **kw)
    return spoke["opt_kwargs"]["options"]


class TestConfigSurface(unittest.TestCase):

    def test_negative_cushion_is_rejected(self):
        # The cushion is SUBTRACTED, so a negative one raises the reported
        # value above the certified quantity and what comes back is not an
        # outer bound. domain=float allowed it; NonNegativeFloat does not.
        cfg = _cfg()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            cfg.ipopt_outer_bound_cushion = -0.05
        cfg.ipopt_outer_bound_cushion = 0.0      # zero still disables it
        self.assertEqual(cfg.ipopt_outer_bound_cushion, 0.0)

    def test_warmstart_does_not_reach_the_spoke(self):
        # shared_options copies --warmstart-subproblems in and solve_one turns
        # it into a `warmstart=` keyword the shell Ipopt interface rejects, so
        # a run that merely asked the HUB's solver to warmstart would kill the
        # spoke on its first solve. The hub's own setting is untouched.
        cfg = _cfg(hub_solver="gurobi_persistent")
        cfg.warmstart_subproblems = True
        self.assertFalse(_spoke_options(cfg)["warmstart_subproblems"])
        beans, kwargs = _beans(cfg)
        hub = vanilla.ph_hub(*beans, scenario_creator_kwargs=kwargs)
        self.assertTrue(hub["opt_kwargs"]["options"]["warmstart_subproblems"])

    def test_presolve_and_obbt_do_not_reach_the_spoke(self):
        # SPOpt runs SPPresolve at CONSTRUCTION, before any guard can reject
        # anything, and shared_options hands obbt_options the GLOBAL solver --
        # so a MIP solver would be invoked on this spoke's convex NLPs from
        # inside its constructor. The hub keeps both.
        cfg = _cfg(hub_solver="gurobi")
        cfg.presolve_args()
        cfg.presolve = True
        cfg.obbt = True
        spoke_opts = _spoke_options(cfg)
        self.assertFalse(spoke_opts["presolve"])
        self.assertNotIn("presolve_options", spoke_opts)
        beans, kwargs = _beans(cfg)
        hub = vanilla.ph_hub(*beans, scenario_creator_kwargs=kwargs)
        self.assertTrue(hub["opt_kwargs"]["options"]["presolve"])
        self.assertEqual(
            hub["opt_kwargs"]["options"]["presolve_options"]
                ["obbt_options"]["solver_name"], "gurobi")

    def test_flags_exist_with_expected_defaults(self):
        cfg = _cfg()
        self.assertFalse(cfg.ipopt_outer_bound)
        self.assertEqual(cfg.ipopt_outer_bound_rank_ratio, 1.0)
        self.assertEqual(cfg.ipopt_outer_bound_cushion, 1e-9)
        # Scoped to Ipopt, but the name is still overridable.
        self.assertIn("ipopt_outer_bound_solver_name", cfg)

    def test_no_mipgap_flags(self):
        # Ipopt is not a branch-and-bound solver; offering mipgap flags would
        # imply otherwise.
        cfg = _cfg()
        self.assertNotIn("ipopt_outer_bound_starting_mipgap", cfg)
        self.assertNotIn("ipopt_outer_bound_iter0_mipgap", cfg)


class TestFactoryWiring(unittest.TestCase):

    def test_solver_defaults_to_ipopt(self):
        # Even when the hub runs something else entirely, the spoke must land
        # on ipopt rather than inheriting -- its own setup guard would reject
        # anything else.
        options = _spoke_options(_cfg(hub_solver="gurobi"))
        self.assertEqual(options["solver_name"], "ipopt")

    def test_explicit_solver_name_is_honored(self):
        cfg = _cfg()
        cfg.ipopt_outer_bound_solver_name = "ipopt_v2"
        self.assertEqual(_spoke_options(cfg)["solver_name"], "ipopt_v2")

    def test_cushion_is_threaded_through(self):
        cfg = _cfg()
        cfg.ipopt_outer_bound_cushion = 1e-7
        self.assertEqual(
            _spoke_options(cfg)["ipopt_outer_bound_cushion"], 1e-7)

    def test_global_solver_options_do_not_leak(self):  # noqa: D401
        # The point of this test: Ipopt hard-fails on an unrecognized keyword
        # rather than ignoring it, so inheriting the global --solver-options
        # (meant for the hub's MIP solver) would kill this spoke on its first
        # solve, with an error naming Ipopt rather than the option routing.
        cfg = _cfg()
        cfg.solver_options = "mipgap=0.01"
        options = _spoke_options(cfg)
        self.assertNotIn("mipgap", options["iter0_solver_options"])
        self.assertNotIn("mipgap", options["iterk_solver_options"])
        self.assertEqual(options["solver_options_layers"], [])

    def test_other_spokes_still_inherit_global_options(self):
        # The contrast that makes the previous test meaningful: not inheriting
        # is special to this spoke, not a change in how spokes work.
        cfg = _cfg()
        cfg.solver_options = "mipgap=0.01"
        beans, kwargs = _beans(cfg)
        lag = vanilla.lagrangian_spoke(*beans, scenario_creator_kwargs=kwargs)
        self.assertIn("mipgap",
                      lag["opt_kwargs"]["options"]["iter0_solver_options"])

    def test_max_solver_threads_does_not_leak(self):
        # --max-solver-threads is re-applied by apply_solver_specs *after* the
        # factory clears the global layers, so clearing alone is not enough.
        # Ipopt has no `threads` option and translate_solver_options has no
        # mapping for it, so it would reach the solver verbatim and hard-fail
        # the spoke's first solve -- taking the whole run with it.
        cfg = _cfg()
        cfg.max_solver_threads = 2
        options = _spoke_options(cfg)
        self.assertNotIn("threads", options["iter0_solver_options"])
        self.assertNotIn("threads", options["iterk_solver_options"])
        for layer in options["solver_options_layers"]:
            self.assertNotIn("threads", layer["options"])

    def test_max_solver_threads_stripped_but_spoke_options_kept(self):
        # Stripping the cap must not take the spoke's own options with it.
        cfg = _cfg()
        cfg.max_solver_threads = 2
        cfg.ipopt_outer_bound_solver_options = "max_iter=42"
        options = _spoke_options(cfg)
        self.assertNotIn("threads", options["iterk_solver_options"])
        self.assertEqual(options["iterk_solver_options"].get("max_iter"), 42)

    def test_per_spoke_solver_options_do_apply(self):
        # Not inheriting the global layer must not mean ignoring the spoke's
        # own options, which is how Ipopt settings are meant to arrive.
        cfg = _cfg()
        cfg.solver_options = "mipgap=0.01"
        cfg.ipopt_outer_bound_solver_options = "max_iter=42"
        options = _spoke_options(cfg)
        self.assertEqual(options["iterk_solver_options"].get("max_iter"), 42)
        self.assertNotIn("mipgap", options["iterk_solver_options"])


def _certifiable_scenario(name="Scen0"):
    """A minimal model that passes check_model_is_certifiable."""
    m = pyo.ConcreteModel(name=name)
    m.x = pyo.Var(bounds=(0, 10), initialize=1.0)
    m.c = pyo.Constraint(expr=m.x >= 1)
    m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
    return m


class _SerialComm:
    """Stand-in for cylinder_comm on a one-rank stub.

    The spoke's diagnostics are collective by design -- their conditions are
    rank-local and Ebound is not -- so a stub that exercises them needs a comm.
    """

    size = 1

    def Get_rank(self):
        return 0

    def allreduce(self, value, op=None):
        return value

    def bcast(self, value, root=0):
        return value

    def allgather(self, value):
        return [value]


class TestSetupGuards(unittest.TestCase):
    """The guards that belong to the spoke rather than the certificate engine.

    Constructed without running the cylinders: the guard reads self.opt.options, so a
    lightweight stand-in exercises it without a solve.
    """

    def _guard_with_solver(self, solver_name, scenarios=None):
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound

        # A real scenario by default. With local_scenarios = {} both
        # per-scenario loops in _check_setup_guards have empty bodies, so the
        # guard tests passed without exercising the guards at all.
        if scenarios is None:
            scenarios = {"Scen0": _certifiable_scenario()}

        class _Stub:
            options = {"solver_name": solver_name}
            local_scenarios = scenarios

        spoke = IpoptOuterBound.__new__(IpoptOuterBound)
        spoke.opt = _Stub()
        spoke.cylinder_rank = 0
        spoke.cylinder_comm = _SerialComm()
        spoke._warned = set()
        return spoke

    def test_non_ipopt_solver_is_rejected(self):
        spoke = self._guard_with_solver("gurobi")
        with self.assertRaisesRegex(CertificateError, "scoped to Ipopt"):
            spoke._check_setup_guards()

    def test_the_measured_ipopt_is_accepted(self):
        self._guard_with_solver("ipopt")._check_setup_guards()
        # and the name is normalized before the test
        self._guard_with_solver("  IPOPT ")._check_setup_guards()

    def test_unmeasured_ipopt_variants_are_rejected(self):
        # These all contain "ipopt", which an earlier substring test accepted.
        # ipopt_v2 and appsi_ipopt run a linear presolve that eliminates rows
        # and then cannot load their duals; cyipopt's sign convention has never
        # been measured against this certificate. Each one fails at solve time
        # with an error naming something other than the solver choice, so the
        # guard has to catch them here.
        for name in ("ipopt_v2", "appsi_ipopt", "cyipopt"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(CertificateError, "scoped to Ipopt"):
                    self._guard_with_solver(name)._check_setup_guards()

    def test_missing_solver_name_is_rejected(self):
        spoke = self._guard_with_solver(None)
        with self.assertRaisesRegex(CertificateError, "scoped to Ipopt"):
            spoke._check_setup_guards()

    def test_fbbt_infeasibility_does_not_take_down_the_run(self):
        """An infeasible scenario is the model's problem, not this spoke's.

        The setup guard runs fbbt to tighten the box and to build the
        unbounded-variable diagnostic. fbbt signals infeasibility by raising,
        and letting that out of a cylinder MPI_Aborts the hub and every other
        spoke over a call this spoke makes for its own convenience.
        """
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1), initialize=0.5)
        m.c = pyo.Constraint(expr=m.x >= 5)
        m.obj = pyo.Objective(expr=m.x)

        spoke = self._guard_with_solver("ipopt")
        spoke.opt.local_scenarios = {"Scen0": m}
        spoke._warned = set()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spoke._check_setup_guards()          # must not raise
        self.assertTrue(any("infeasible" in str(w.message) for w in caught))


class TestInheritsTheLagrangianDriver(unittest.TestCase):
    """The spoke used to fork LagrangianOuterBound.main() rather than subclass
    it, and the copy had drifted: it lost _PreLoopXhatMixin, it lost the
    `else: do_while_waiting_for_new_Ws(...)` branch (which made
    --subgradient-while-waiting a silent no-op), and its `outer_bound_only`
    was inert. Assert the wiring, since none of that failed loudly."""

    def test_it_is_a_lagrangian_outer_bound(self):
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound
        from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
        from mpisppy.cylinders._preloop_xhat_mixin import _PreLoopXhatMixin
        self.assertTrue(issubclass(IpoptOuterBound, LagrangianOuterBound))
        self.assertTrue(issubclass(IpoptOuterBound, _PreLoopXhatMixin))

    def test_the_driver_is_not_reimplemented(self):
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound
        from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
        for name in ("main", "do_while_waiting_for_new_Ws",
                     "_set_weights_and_solve"):
            with self.subTest(name=name):
                self.assertNotIn(name, IpoptOuterBound.__dict__)
                self.assertIs(getattr(IpoptOuterBound, name),
                              getattr(LagrangianOuterBound, name))

    def test_only_the_bound_computation_is_overridden(self):
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound
        self.assertIn("lagrangian", IpoptOuterBound.__dict__)
        self.assertIn("lagrangian_prep", IpoptOuterBound.__dict__)

    def test_jensens_is_declined(self):
        # The inherited main() would offer a Jensen's bound taken from
        # results.problem.lower_bound -- the solver dual bound Ipopt does not
        # produce, and the reason this spoke exists.
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound
        spoke = IpoptOuterBound.__new__(IpoptOuterBound)
        spoke.opt = type("_O", (), {"options": {"jensens": {}}})()
        self.assertFalse(spoke._jensens_enabled())

    def test_the_certificate_needs_the_solution_loaded(self):
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound
        from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
        # The base spoke skips loading the solution; the certificate reads the
        # point AND the duals off the solved model, so it cannot.
        self.assertTrue(LagrangianOuterBound.outer_bound_only)
        self.assertFalse(IpoptOuterBound.outer_bound_only)


class TestDualSuffixGuard(unittest.TestCase):
    """The certificate reads the solver's duals, so the Suffix has to import."""

    def _spoke_over(self, scenario):
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound
        spoke = IpoptOuterBound.__new__(IpoptOuterBound)
        spoke.opt = type("_O", (), {"local_scenarios": {"Scen0": scenario}})()
        # The reject path is collective -- see _raise_collectively -- so even a
        # one-rank stub needs a comm.
        spoke.cylinder_rank = 0
        spoke.cylinder_comm = _SerialComm()
        return spoke

    def test_a_suffix_is_attached_when_there_is_none(self):
        m = _certifiable_scenario()
        self._spoke_over(m)._attach_dual_suffixes()
        self.assertTrue(m.dual.import_enabled())

    def test_an_importing_suffix_is_left_alone(self):
        m = _certifiable_scenario()
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        original = m.dual
        self._spoke_over(m)._attach_dual_suffixes()
        self.assertIs(m.dual, original)

    def test_an_export_only_suffix_is_rejected(self):
        # A scenario_creator supplying dual warm starts attaches EXPORT;
        # reusing it would import nothing and the certificate would see no
        # multipliers at all.
        # Unnamed, keyed "Scen0", for the same reason as the test below:
        # _certifiable_scenario() names the model "Scen0" too, so this passed
        # whether the message used the key or the model name.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10), initialize=1.0)
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
        m.dual = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        with self.assertRaises(CertificateError) as ctx:
            self._spoke_over(m)._attach_dual_suffixes()
        message = str(ctx.exception)
        self.assertIn("does not import", message)
        self.assertIn("Scen0", message)
        self.assertNotIn("unknown", message)

    def test_a_dual_that_is_not_a_suffix_is_rejected_by_name(self):
        # getattr returns whatever the scenario_creator declared, and `dual` is
        # an ordinary enough name for a Var. Calling import_enabled() on it
        # raised AttributeError out of lagrangian_prep, naming neither this
        # spoke nor the component; every other setup problem here is a
        # CertificateError naming the scenario.
        # Deliberately UNNAMED, while the local_scenarios key is "Scen0".
        # _certifiable_scenario() names the model "Scen0" too, which would let
        # this pass whether the message used the key or the model name.
        m = pyo.ConcreteModel()
        self.assertEqual(m.name, "unknown")       # the premise
        m.x = pyo.Var(bounds=(0, 10), initialize=1.0)
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
        m.dual = pyo.Var(initialize=0.0)
        with self.assertRaises(CertificateError) as ctx:
            self._spoke_over(m)._attach_dual_suffixes()
        message = str(ctx.exception)
        self.assertIn("not a Suffix", message)
        self.assertIn("ScalarVar", message)
        # By the local_scenarios KEY. s.name is the Pyomo model name, which
        # SPBase never sets -- an unnamed ConcreteModel reports "unknown".
        self.assertIn("Scen0", message)
        self.assertNotIn("unknown", message)


class TestNewlyFixedNonants(unittest.TestCase):
    """Fixing a nonant after setup restricts the subproblem, so its minimum
    bounds the restricted problem and not the original."""

    def _spoke(self, fixed_now, fixed_at_setup):
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound
        m = _certifiable_scenario()
        m.x.fixed = fixed_now
        data = type("_D", (), {})()
        data.nonant_indices = {("ROOT", 0): m.x}
        m._mpisppy_data = data
        spoke = IpoptOuterBound.__new__(IpoptOuterBound)
        spoke.opt = type("_O", (), {"local_scenarios": {"Scen0": m}})()
        spoke.cylinder_rank = 0
        spoke.cylinder_comm = _SerialComm()
        spoke._warned = set()
        spoke._fixed_at_setup = {("Scen0", ("ROOT", 0)): fixed_at_setup}
        return spoke

    def test_newly_fixed_is_detected(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertTrue(self._spoke(True, False)._nonants_newly_fixed())
        self.assertTrue(any("fixed after" in str(w.message) for w in caught))

    def test_fixed_by_the_scenario_creator_is_fine(self):
        # Part of the problem as stated, not a restriction of it.
        self.assertFalse(self._spoke(True, True)._nonants_newly_fixed())

    def test_nothing_fixed_is_fine(self):
        self.assertFalse(self._spoke(False, False)._nonants_newly_fixed())

    def test_the_answer_is_global_so_the_branch_cannot_diverge(self):
        """lagrangian() branches on this to skip the rest of the iteration.

        A rank-local answer means the rank with a fixed nonant returns into
        Ebound's Allreduce while its peers enter two more collectives on the
        same communicator -- a hang with no error. This rank saw nothing; its
        peer did; it must still say True.
        """
        from mpisppy import MPI
        spoke = self._spoke(False, False)          # nothing fixed HERE

        class _TwoRanks:
            size = 2

            def Get_rank(self):
                return 0

            def allreduce(self, value, op=None):
                assert op is MPI.MIN
                return min(value, 1)               # rank 1 saw one

        spoke.cylinder_comm = _TwoRanks()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.assertTrue(spoke._nonants_newly_fixed())


class TestProxGuard(unittest.TestCase):
    """The bound is a LAGRANGIAN bound, so a proximal subproblem would not
    produce it. The old guard tested prox_on, a Param attach_Ws_and_prox
    always creates at 0, so it could never fire."""

    def _spoke(self, attach_prox):
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound

        class _Opt:
            options = {"solver_name": "ipopt"}
            local_scenarios = {}

        spoke = IpoptOuterBound.__new__(IpoptOuterBound)
        spoke.opt = _Opt()
        spoke.opt._attach_prox = attach_prox
        spoke.cylinder_rank = 0
        spoke.cylinder_comm = _SerialComm()
        spoke._warned = set()
        return spoke

    def test_prox_on_is_rejected(self):
        with self.assertRaisesRegex(CertificateError, "proximal term to be off"):
            self._spoke(True)._check_setup_guards()

    def test_prox_off_passes(self):
        self._spoke(False)._check_setup_guards()


class TestFbbtExceptionsStandDown(unittest.TestCase):
    """fbbt raises more than the infeasibility the guard asks it about.

    unbounded_variables(do_fbbt=True) is called to TIGHTEN the box and to build
    a diagnostic. Nothing it raises is worth aborting the run, and an
    exception escaping lagrangian_prep does exactly that.
    """

    def _spoke_over(self, scenario):
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound

        class _Stub:
            options = {"solver_name": "ipopt"}
            local_scenarios = {"Scen0": scenario}

        spoke = IpoptOuterBound.__new__(IpoptOuterBound)
        spoke.opt = _Stub()
        spoke.opt._attach_prox = False
        spoke.cylinder_rank = 0
        spoke.cylinder_comm = _SerialComm()
        spoke._warned = set()
        return spoke

    # Models fbbt cannot analyze, one per exception class it is known to
    # raise. Enumerating exception classes in the GUARD was tried twice and was
    # wrong twice -- IntervalException was caught, then PyomoException, and
    # OverflowError is neither -- so the guard now catches Exception and this
    # list is what keeps that honest. Each entry must be admitted by
    # check_model_is_certifiable, or it would never reach the fbbt call.
    def _negative_base_scenario(self):
        """IntervalException: a negative base raised to a variable power."""
        m = pyo.ConcreteModel(name="Scen0")
        m.x = pyo.Var(bounds=(-5, -1), initialize=-2.0)
        m.y = pyo.Var(bounds=(0.5, 2), initialize=1.0)
        m.z = pyo.Var(initialize=0.0)             # deliberately unbounded
        m.c = pyo.Constraint(expr=m.x ** m.y <= 10)
        m.obj = pyo.Objective(expr=m.x + m.y + m.z, sense=pyo.minimize)
        return m

    def _overflowing_power_scenario(self):
        """OverflowError: interval.power computes xu**yu unguarded.

        Not a PyomoException, which is what makes it the case that a
        PyomoException catch still let through.
        """
        m = pyo.ConcreteModel(name="Scen0")
        m.x = pyo.Var(bounds=(1.0, 1e200), initialize=1.0)
        m.y = pyo.Var(bounds=(0, None), initialize=1.0)
        m.z = pyo.Var(initialize=0.0)             # deliberately unbounded
        m.c = pyo.Constraint(expr=m.x ** 3 <= m.y)
        m.obj = pyo.Objective(expr=m.y + m.z, sense=pyo.minimize)
        return m

    def _unanalyzable_scenarios(self):
        """label -> (builder, the class fbbt raises on it).

        The class is carried, not just asserted to be "some exception": two
        rows that collapsed onto the same class would silently stop covering
        the other one, which is the failure this table exists to prevent.
        """
        from pyomo.common.errors import IntervalException
        return {
            "negative base to a variable power":
                (self._negative_base_scenario, IntervalException),
            "overflow in interval.power":
                (self._overflowing_power_scenario, OverflowError),
        }

    def test_the_scenarios_are_admitted_by_the_certifiability_check(self):
        # If this ever stops holding, the fbbt call is unreachable for that
        # model and its case below stops testing anything.
        from mpisppy.utils.dual_certificate import check_model_is_certifiable
        for label, (build, _) in self._unanalyzable_scenarios().items():
            with self.subTest(label):
                check_model_is_certifiable(build())

    def test_fbbt_raises_the_expected_class_on_each_of_them(self):
        # The premise of the tests below, twice over: that fbbt raises at all
        # (no exception, nothing being caught) and that the two rows still
        # cover two DIFFERENT classes. Note the build is outside assertRaises:
        # a model that started raising at CONSTRUCTION would otherwise satisfy
        # the assertion while testing nothing.
        from mpisppy.utils.dual_certificate import unbounded_variables
        table = self._unanalyzable_scenarios()
        seen = {}
        for label, (build, expected) in table.items():
            with self.subTest(label):
                scenario = build()
                with self.assertRaises(expected) as ctx:
                    unbounded_variables(scenario, do_fbbt=True)
                # the class RAISED, not the class declared in the table --
                # recording the latter makes the distinctness check below
                # test the dict literal rather than Pyomo's behavior
                seen[label] = type(ctx.exception)
        self.assertEqual(len(seen), len(table),
                         "a row did not raise, so it never reached the check")
        self.assertEqual(len(set(seen.values())), len(seen),
                         f"two rows collapsed onto one exception class: {seen}")

    def test_fbbt_raising_does_not_escape_the_guard(self):
        for label, (build, _) in self._unanalyzable_scenarios().items():
            with self.subTest(label):
                spoke = self._spoke_over(build())
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    spoke._check_setup_guards()   # must not raise
                self.assertTrue(
                    any("could not analyze" in str(w.message) for w in caught),
                    "stood down silently; it should say the box is untightened")

    def test_the_warning_names_the_exception_class(self):
        # The wide catch swallows a genuine bug in our own code too, so the
        # message has to carry enough to recognize one.
        spoke = self._spoke_over(self._overflowing_power_scenario())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spoke._check_setup_guards()
        self.assertTrue(
            any("OverflowError" in str(w.message) for w in caught))

    def test_the_unbounded_diagnostic_survives_fbbt_bailing(self):
        # Losing the tightening costs looseness. Losing the diagnostic would
        # cost the user the one message that explains an empty 'N' column, so
        # the scan is redone without fbbt.
        for label, (build, _) in self._unanalyzable_scenarios().items():
            with self.subTest(label):
                spoke = self._spoke_over(build())
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    spoke._check_setup_guards()
                self.assertTrue(
                    any("no finite bound" in str(w.message) for w in caught))


class TestCollectiveRaise(unittest.TestCase):
    """A hard error at setup must reach every rank, or none.

    The conditions are rank-local -- a discrete variable, a `dual` of the wrong
    type, sit in one scenario on one rank -- while everything after them is
    collective. A bare raise on the one offending rank leaves its peers in the
    next allreduce with no partner: a hang, not an error. Today it dies rather
    than hangs only because WheelSpinner.run wraps the wheel in MPI_Abort
    (#852), which is not a property of this spoke.
    """

    def _spoke(self, rank, peer_saw_it):
        from mpisppy import MPI
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound

        class _TwoRanks:
            size = 2

            def Get_rank(self):
                return rank

            def allreduce(self, value, op=None):
                assert op is MPI.MIN
                return min(value, 1) if peer_saw_it else value

            def bcast(self, value, root=0):
                # rank 1 is the one with the bad scenario in these tests
                return value if root == rank else "scenario Scen1: a discrete variable"

        spoke = IpoptOuterBound.__new__(IpoptOuterBound)
        spoke.cylinder_rank = rank
        spoke.cylinder_comm = _TwoRanks()
        return spoke

    def test_no_rank_raises_when_no_rank_saw_a_problem(self):
        self._spoke(rank=0, peer_saw_it=False)._raise_collectively(None)

    def test_a_clean_rank_raises_when_its_peer_saw_the_problem(self):
        """This is the fix. Before it, rank 0 returned and blocked."""
        spoke = self._spoke(rank=0, peer_saw_it=True)
        with self.assertRaises(CertificateError) as ctx:
            spoke._raise_collectively(None)       # nothing wrong HERE
        # and it names the rank and the real cause, not "some other rank"
        message = str(ctx.exception)
        self.assertIn("rank 1", message)
        self.assertIn("Scen1", message)

    def test_the_offending_rank_raises_its_own_message(self):
        spoke = self._spoke(rank=0, peer_saw_it=False)
        with self.assertRaisesRegex(CertificateError, "rank 0: my own problem"):
            spoke._raise_collectively("my own problem")


@unittest.skipUnless(comm.size == 2, "needs exactly two ranks")
class TestCollectiveRaiseOnRealRanks(unittest.TestCase):
    """The stubbed version above pins the logic; this pins the MPI calls.

    A stub cannot catch a wrong `op`, a bcast whose root disagrees between
    ranks, or an allreduce that the ranks enter with different types -- and
    those are exactly the mistakes that turn a guard meant to prevent a hang
    into one.
    """

    def _spoke(self):
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound
        spoke = IpoptOuterBound.__new__(IpoptOuterBound)
        spoke.cylinder_comm = comm
        spoke.cylinder_rank = comm.Get_rank()
        return spoke

    def test_one_rank_with_a_problem_raises_on_both(self):
        # Only rank 1 has the bad scenario. Both ranks must raise; before the
        # fix rank 0 returned and blocked in the next collective.
        rank = comm.Get_rank()
        problem = "scenario Scen1: has a discrete variable" if rank == 1 else None
        with self.assertRaises(CertificateError) as ctx:
            self._spoke()._raise_collectively(problem)
        # and BOTH tracebacks name the scenario that actually caused it
        self.assertIn("rank 1", str(ctx.exception))
        self.assertIn("Scen1", str(ctx.exception))

    def test_no_rank_with_a_problem_raises_on_neither(self):
        self._spoke()._raise_collectively(None)
        # If either rank had raised, the other would hang here rather than
        # reach the barrier.
        comm.Barrier()


class TestCollectiveWarning(unittest.TestCase):
    """The conditions this spoke warns about are rank-local; Ebound is not.

    Gating on `cylinder_rank == 0` silenced the rank that actually saw the
    problem, leaving an empty bound column and no explanation.
    """

    def _spoke(self, rank, flags):
        """A spoke on `rank` of a comm where `flags[r]` says whether rank r
        saw the condition. The comm performs the REAL reduction over all of
        them rather than returning a canned answer, so an inverted ternary or
        the wrong operator fails here instead of passing."""
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound
        from mpisppy import MPI

        class _Comm:
            size = len(flags)

            def Get_rank(self):
                return rank

            def allreduce(self, value, op=None):
                assert op is MPI.MIN, "the helper must reduce with MIN"
                contributions = [
                    r if flags[r] else len(flags) for r in range(len(flags))
                ]
                assert value == contributions[rank], "this rank's contribution"
                return min(contributions)

        spoke = IpoptOuterBound.__new__(IpoptOuterBound)
        spoke.cylinder_rank = rank
        spoke.cylinder_comm = _Comm()
        spoke._warned = set()
        return spoke

    def _warn(self, spoke, local_flag):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._returned = spoke._warn_once_collectively(
                "k", local_flag, lambda: "saw it")
        return [str(w.message) for w in caught]

    def test_the_rank_that_saw_it_speaks_even_when_it_is_not_rank_zero(self):
        spoke = self._spoke(rank=1, flags=[False, True])
        self.assertEqual(self._warn(spoke, True), ["saw it"])

    def test_the_other_ranks_stay_quiet(self):
        spoke = self._spoke(rank=0, flags=[False, True])
        self.assertEqual(self._warn(spoke, False), [])

    def test_silence_when_no_rank_saw_it(self):
        spoke = self._spoke(rank=0, flags=[False, False])
        self.assertEqual(self._warn(spoke, False), [])
        self.assertNotIn("k", spoke._warned)   # not consumed, can still fire

    def test_key_is_consumed_on_every_rank_so_it_fires_only_once(self):
        spoke = self._spoke(rank=0, flags=[False, True])
        self._warn(spoke, False)
        self.assertIn("k", spoke._warned)
        spoke2 = self._spoke(rank=1, flags=[False, True])
        self.assertEqual(self._warn(spoke2, True), ["saw it"])
        self.assertEqual(self._warn(spoke2, True), [])

    def test_the_answer_is_global_not_rank_local(self):
        """Callers branch on this. A rank-local answer sends some ranks down a
        path that skips collectives the others enter, which hangs the run."""
        quiet = self._spoke(rank=0, flags=[False, True])
        self._warn(quiet, False)
        self.assertTrue(self._returned)          # False locally, True globally
        nobody = self._spoke(rank=0, flags=[False, False])
        self._warn(nobody, False)
        self.assertFalse(self._returned)

    def test_the_global_answer_survives_the_warn_once(self):
        # Second call: the key is consumed, but the reduction must still run
        # and still report the global condition, or the branch flips on
        # iteration two and the ranks diverge.
        spoke = self._spoke(rank=0, flags=[False, True])
        self._warn(spoke, False)
        self._warn(spoke, False)
        self.assertTrue(self._returned)


class TestCertificateFailureStandsDown(unittest.TestCase):
    """Evaluating phi at the returned point can raise things that are not
    CertificateError, and none of them is worth aborting the run."""

    def _spoke_over(self, scenario):
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound

        class _Opt:
            options = {"verbose": False, "tee-rank0-solves": False,
                       "ipopt_outer_bound_cushion": 1e-9}
            local_scenarios = {"Scen0": scenario}
            _PHIter = 1

            def _effective_solver_options(self, iteration):
                return {}

            def solve_loop(self, **kwargs):
                pass                              # the solve is not under test

            def Ebound(self, verbose):
                return "EBOUND"

        spoke = IpoptOuterBound.__new__(IpoptOuterBound)
        spoke.opt = _Opt()
        spoke.cylinder_rank = 0
        spoke.cylinder_comm = _SerialComm()
        spoke._warned = set()
        spoke.receive_nonant_bounds = lambda: None
        spoke._nonants_newly_fixed = lambda: False
        return spoke

    def _scenario_with_uninitialized_var(self):
        # No initialize=, so evaluating phi raises ValueError rather than
        # CertificateError. bound_relax_factor putting an iterate a hair
        # outside a log or a sqrt raises the same class.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 4))
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.dual[m.c] = 0.0
        m.obj = pyo.Objective(expr=m.x)
        m.name = "Scen0"
        m._mpisppy_data = type(
            "_D", (), {"solution_available": True, "outer_bound": "UNSET"})()
        return m

    def _scenario_differentiate_cannot_handle(self):
        """A model that PASSES setup and still cannot be differentiated.

        abs is deliberately the case here, not cosh. The structurally
        unsupported functions -- cosh, sinh, tanh, ceil, floor, Expr_if -- are
        now a hard error in check_model_is_certifiable, so a scenario carrying
        one never reaches the iteration loop and could not exercise the
        stand-down. abs is the case that survives that guard: differentiate
        handles it everywhere except exactly at the kink, which depends on the
        POINT and not on the model, so it cannot be decided at setup. The
        iterate landing on x=0 raises DifferentiationException, which derives
        straight from Exception -- not ValueError, not ArithmeticError, not
        even PyomoException.
        """
        m = pyo.ConcreteModel()                   # unnamed on purpose
        m.x = pyo.Var(bounds=(-1, 1), initialize=0.0)   # exactly on the kink
        m.y = pyo.Var(bounds=(0, 100), initialize=5.0)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.c = pyo.Constraint(expr=abs(m.x) <= m.y)
        m.dual[m.c] = -1.0
        m.obj = pyo.Objective(expr=m.y)
        m._mpisppy_data = type(
            "_D", (), {"solution_available": True, "outer_bound": "UNSET"})()
        return m

    def test_the_model_reaches_the_iteration_loop_at_all(self):
        """Otherwise the stand-down test below exercises an unreachable path.

        The setup guard must NOT reject this one: if it ever does, the case is
        being caught earlier and this class stops testing the runtime catch.
        """
        from mpisppy.utils.dual_certificate import check_model_is_certifiable
        check_model_is_certifiable(self._scenario_differentiate_cannot_handle())

    def test_a_differentiation_failure_becomes_no_bound(self):
        from mpisppy.utils.dual_certificate import certified_lower_bound
        # The premise, on its OWN model: that the call really does raise, and
        # really is outside the classes the old enumerated catch listed.
        with self.assertRaises(Exception) as ctx:
            certified_lower_bound(self._scenario_differentiate_cannot_handle(),
                                  sign_convention="ipopt", eps_rel=1e-9)
        self.assertNotIsInstance(
            ctx.exception, (CertificateError, ValueError, ArithmeticError),
            "no longer outside the old enumerated catch; pick another model")

        # and now the model the spoke actually runs, so the assertions below
        # are about the object it touched
        scenario = self._scenario_differentiate_cannot_handle()
        spoke = self._spoke_over(scenario)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = spoke.lagrangian()           # must not raise
        self.assertEqual(result, "EBOUND")
        self.assertIsNone(scenario._mpisppy_data.outer_bound)
        matching = [str(w.message) for w in caught
                    if "no certificate" in str(w.message)]
        # assertion, not StopIteration: a missing warning should say so
        self.assertEqual(len(matching), 1, f"warnings were: {matching}")
        message = matching[0]
        # the class is reported, so a genuine bug stays legible
        self.assertIn("DifferentiationException", message)
        # and the scenario is named by its local_scenarios KEY, not s.name
        self.assertIn("Scen0", message)
        self.assertNotIn("unknown", message)

    def test_a_later_failure_of_a_different_class_still_warns(self):
        """The wide catch is only acceptable if a genuine bug stays audible.

        The warning used to be keyed "certificate_failed", which
        _warn_once_collectively consumes on the FIRST failure of the run --
        typically a routine CertificateError. A DifferentiationException
        arriving on a later iteration was then silent for the rest of the run,
        and the spoke reported no bound with no explanation. Before the catch
        was widened it at least aborted loudly.
        """
        def warnings_from(spoke):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                spoke.lagrangian()
            return [str(w.message) for w in caught]

        # Iteration 1: a routine ValueError. _scenario_with_uninitialized_var
        # DOES attach a dual suffix; what it leaves out is the Var's value, so
        # evaluating phi raises. The class matters and only the class matters
        # -- it has to differ from the one below for the premise to hold.
        routine = self._scenario_with_uninitialized_var()
        spoke = self._spoke_over(routine)
        first = warnings_from(spoke)
        self.assertTrue(any("no certificate" in m for m in first))
        # the premise, asserted rather than described: two DIFFERENT classes
        self.assertTrue(any("ValueError" in m for m in first), first)

        # iteration 2, same spoke so _warned persists: a different class
        structural = self._scenario_differentiate_cannot_handle()
        spoke.opt.local_scenarios = {"Scen0": structural}
        second = warnings_from(spoke)
        self.assertTrue(
            any("DifferentiationException" in m for m in second),
            "a new failure class was swallowed silently; before the catch was "
            f"widened this aborted loudly. warnings were: {second}")

        # and the routine class does NOT warn twice
        spoke.opt.local_scenarios = {"Scen0": self._scenario_with_uninitialized_var()}
        third = warnings_from(spoke)
        self.assertFalse(any("no certificate" in m for m in third),
                         "once per class, not once per iteration")

    def _scenario_with_no_bound_and_a_missing_dual(self):
        """certified_lower_bound RETURNS None here, without raising.

        An unbounded variable whose gradient component in phi is nonzero is an
        ordinary outcome for this spoke -- the unbounded_variables warning at
        setup exists for exactly it -- and the constraint has no imported dual,
        so missing_duals is populated on the way to returning None.
        """
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(None, None), initialize=1.0)
        m.y = pyo.Var(bounds=(0, 10), initialize=1.0)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.c = pyo.Constraint(expr=m.x + m.y >= 1)     # no dual imported
        m.obj = pyo.Objective(expr=m.x + m.y)
        m._mpisppy_data = type(
            "_D", (), {"solution_available": True, "outer_bound": "UNSET"})()
        return m

    def test_a_scenario_with_no_bound_does_not_claim_a_looser_one(self):
        from mpisppy.utils.dual_certificate import certified_lower_bound
        # the premise: returns None WITHOUT raising, having populated the list
        probe, missing = self._scenario_with_no_bound_and_a_missing_dual(), []
        self.assertIsNone(certified_lower_bound(
            probe, sign_convention="ipopt", eps_rel=1e-9,
            missing_duals=missing))
        self.assertEqual(missing, ["c"])

        scenario = self._scenario_with_no_bound_and_a_missing_dual()
        spoke = self._spoke_over(scenario)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spoke.lagrangian()
        self.assertIsNone(scenario._mpisppy_data.outer_bound)
        # "looser than it could be but still valid" is false of a scenario
        # that produced no bound at all, and saying it burns the warn-once
        # key so a real tightness loss later is never reported.
        self.assertFalse(
            any("still valid" in str(w.message) for w in caught),
            [str(w.message) for w in caught])

    def _scenario_with_a_bound_and_a_missing_dual(self):
        """Produces a bound, and one constraint has no imported dual.

        The positive half of the gate. Everything bounded, so the correction
        is finite; the dual-less row is taken with multiplier zero, which is
        what "looser than it could be but still valid" describes.
        """
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10), initialize=1.0)
        m.y = pyo.Var(bounds=(0, 10), initialize=1.0)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.c = pyo.Constraint(expr=m.x + m.y >= 1)     # no dual imported
        m.obj = pyo.Objective(expr=m.x + m.y)
        m._mpisppy_data = type(
            "_D", (), {"solution_available": True, "outer_bound": "UNSET"})()
        return m

    def test_a_scenario_that_does_produce_a_bound_reports_the_missing_dual(self):
        """Without this the gate is unguarded in the direction that matters.

        Deleting the merge outright, or inverting the condition to `is None`,
        left the whole suite green: only the negative half was asserted.
        """
        scenario = self._scenario_with_a_bound_and_a_missing_dual()
        spoke = self._spoke_over(scenario)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spoke.lagrangian()
        messages = [str(w.message) for w in caught]
        self.assertIsNotNone(scenario._mpisppy_data.outer_bound, messages)
        self.assertTrue(any("still valid" in m for m in messages), messages)

    def test_a_scenario_with_no_bound_is_not_silent(self):
        """Dropping the false message is only an improvement if something
        true replaces it. It reached neither failure list, so nothing did."""
        scenario = self._scenario_with_no_bound_and_a_missing_dual()
        spoke = self._spoke_over(scenario)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spoke.lagrangian()
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("no bound" in m for m in messages),
                        f"silent: an empty 'N' column with no reason. {messages}")
        # and it names WHICH cause, with advice that fits it
        self.assertTrue(
            any("unbounded below" in m for m in messages), messages)
        self.assertTrue(any("finite bound is the fix" in m for m in messages),
                        messages)
        # and it names the variable and the side, from the engine
        self.assertTrue(any("no finite lower bound" in m for m in messages),
                        messages)

    def test_a_non_finite_value_is_not_blamed_on_unbounded_variables(self):
        """One flat key, or one message naming two causes, sends the user
        hunting for an unbounded variable that does not exist."""
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10), initialize=1.0)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.dual[m.c] = float("nan")                # a diverged solve's duals
        m.obj = pyo.Objective(expr=m.x)
        m._mpisppy_data = type(
            "_D", (), {"solution_available": True, "outer_bound": "UNSET"})()

        spoke = self._spoke_over(m)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spoke.lagrangian()
        messages = [str(w.message) for w in caught]
        self.assertIsNone(m._mpisppy_data.outer_bound, messages)
        self.assertTrue(any("non-finite value" in m for m in messages),
                        messages)
        # every variable here IS bounded; saying otherwise is a wild goose chase
        self.assertFalse(any("finite bound is the fix" in m for m in messages),
                         messages)

    def test_a_failed_solution_load_does_not_take_down_the_run(self):
        """need_solution=False, so spopt hands the load failure back.

        With True it re-raises from inside the iteration loop, which
        MPI_Aborts the hub and every other cylinder -- an optional source of a
        bound should not do that. spopt instead sets solution_available=False,
        which this loop already reports as a cause.

        The flag governs ONLY the load step: a solve that fails outright still
        re-raises its solver_exception from the not_good_enough_results branch,
        which need_solution does not gate. This test pins the reach, so the
        comment in the spoke cannot drift from it.
        """
        import inspect
        from mpisppy import spopt
        recorded = {}
        scenario = self._scenario_with_a_bound_and_a_missing_dual()
        spoke = self._spoke_over(scenario)
        spoke.opt.solve_loop = lambda **kw: recorded.update(kw)
        spoke.lagrangian()
        self.assertIs(recorded.get("need_solution"), False,
                      f"solve_loop was called with {recorded}")

        # and the reach: need_solution guards the load, not the solve
        source = inspect.getsource(spopt.SPOpt.solve_one)
        self.assertIn("if need_solution:", source)
        self.assertIn("raise solver_exception", source,
                      "spopt no longer re-raises there; the spoke comment "
                      "about the reach of need_solution needs revisiting")

    def test_a_scenario_with_no_solution_is_not_silent(self):
        """solve_loop reports the failed solve; only this cylinder can report
        that Ebound is all-or-nothing so the whole spoke stood down."""
        m = self._scenario_with_a_bound_and_a_missing_dual()
        m._mpisppy_data.solution_available = False

        spoke = self._spoke_over(m)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spoke.lagrangian()
        messages = [str(w.message) for w in caught]
        self.assertIsNone(m._mpisppy_data.outer_bound)
        self.assertTrue(any("no loadable solution" in m for m in messages),
                        f"the last silent path. {messages}")

    def test_a_second_no_bound_cause_still_warns_on_the_same_spoke(self):
        """The reason the key carries the cause.

        _warn_once_collectively consumes a key for the life of the run, so one
        flat "no_bound_returned" key is spent by whichever cause arrives first
        and every other cause is silent from then on -- the burnt-key problem
        that keying failures_by_class per class was meant to end. A fresh
        spoke per test cannot see this: it takes two iterations on ONE spoke.
        """
        def warnings_from(spoke):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                spoke.lagrangian()
            return [str(w.message) for w in caught]

        # iteration 1: an unbounded variable
        spoke = self._spoke_over(self._scenario_with_no_bound_and_a_missing_dual())
        first = warnings_from(spoke)
        self.assertTrue(any("unbounded below" in m for m in first), first)

        # iteration 2, same spoke so _warned persists: a different cause
        nan_scenario = pyo.ConcreteModel()
        nan_scenario.x = pyo.Var(bounds=(0, 10), initialize=1.0)
        nan_scenario.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        nan_scenario.c = pyo.Constraint(expr=nan_scenario.x >= 1)
        nan_scenario.dual[nan_scenario.c] = float("nan")
        nan_scenario.obj = pyo.Objective(expr=nan_scenario.x)
        nan_scenario._mpisppy_data = type(
            "_D", (), {"solution_available": True, "outer_bound": "UNSET"})()
        spoke.opt.local_scenarios = {"Scen0": nan_scenario}
        second = warnings_from(spoke)
        self.assertTrue(
            any("non-finite value" in m for m in second),
            f"a second cause was swallowed by the first one's key: {second}")

        # and the first cause does not warn twice
        spoke.opt.local_scenarios = {
            "Scen0": self._scenario_with_no_bound_and_a_missing_dual()}
        third = warnings_from(spoke)
        self.assertFalse(any("unbounded below" in m for m in third),
                         f"once per cause, not once per iteration: {third}")

    def test_a_non_finite_value_alongside_an_irrelevant_unbounded_var(self):
        """The configuration the old scan-based classifier misread.

        `z` has no upper bound but is absent from phi's gradient, so it is NOT
        why there is no bound -- the NaN dual is. Scanning the model for any
        variable with an infinite bound found `z` and reported the unbounded
        cause, with the advice for it, and burnt that key; the real cause then
        had no way to be reported for the rest of the run even though it
        recurs every iteration. The engine reports which site fired, so this
        cannot be got wrong by re-derivation.
        """
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10), initialize=1.0)
        m.z = pyo.Var(bounds=(0, None), initialize=0.0)   # unbounded, unused
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.dual[m.c] = float("nan")
        m.obj = pyo.Objective(expr=m.x)                   # z not in the objective
        m._mpisppy_data = type(
            "_D", (), {"solution_available": True, "outer_bound": "UNSET"})()

        spoke = self._spoke_over(m)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spoke.lagrangian()
        messages = [str(w.message) for w in caught]
        self.assertIsNone(m._mpisppy_data.outer_bound, messages)
        self.assertTrue(any("non-finite value" in m for m in messages),
                        messages)
        self.assertFalse(any("unbounded below" in m for m in messages),
                         f"blamed z, which is not why there is no bound: "
                         f"{messages}")

    def test_a_nan_gradient_on_a_one_sided_variable_is_not_called_unbounded(self):
        """NaN answers False to every comparison, which is how it got mislabelled.

        `g == 0.0` and `g > 0.0` are both False for NaN, so a NaN gradient took
        the `v.ub` branch, and on NonNegativeReals -- ub is None -- reported an
        unbounded box: a diverged solve dressed as a missing bound, with the
        advice for the wrong one, burning the wrong key. The two existing
        non-finite fixtures both bound x on BOTH sides, which routes the NaN to
        a finite ub and is exactly why neither caught this.
        """
        m = pyo.ConcreteModel()
        m.x = pyo.Var(within=pyo.NonNegativeReals, initialize=1.0)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.dual[m.c] = float("nan")
        m.obj = pyo.Objective(expr=m.x)
        m._mpisppy_data = type(
            "_D", (), {"solution_available": True, "outer_bound": "UNSET"})()

        spoke = self._spoke_over(m)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spoke.lagrangian()
        messages = [str(w.message) for w in caught]
        self.assertIsNone(m._mpisppy_data.outer_bound, messages)
        self.assertTrue(any("non-finite value" in m for m in messages),
                        messages)
        self.assertFalse(
            any("unbounded below" in m for m in messages),
            f"a diverged solve reported as a missing bound: {messages}")

    def test_an_unlabelled_no_bound_is_not_attributed_to_a_known_cause(self):
        """Unreachable today, which is the reason to pin it.

        Both return-None sites record a tag, so nothing exercises the
        fallback. If a third is ever added without one, the fallback decides
        what the user is told -- and guessing "non_finite" would print a
        specific cause, and advice for it, for something nobody classified.
        """
        from unittest import mock
        scenario = self._scenario_with_a_bound_and_a_missing_dual()
        spoke = self._spoke_over(scenario)
        with mock.patch(
            "mpisppy.cylinders.ipopt_outer_bound.certified_lower_bound",
            return_value=None,          # returns None, records no reason
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                spoke.lagrangian()
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("recorded no reason" in m for m in messages),
                        messages)
        self.assertFalse(any("bounds will not help" in m for m in messages),
                         f"asserted a cause nobody classified: {messages}")

    def test_a_nan_beats_an_unbounded_variable_regardless_of_order(self):
        """Screening per component only moved the misclassification.

        The bound-selection loop returns on the first UNBOUNDED component too,
        so whichever came first in vlist won. Here `x` is unbounded above and
        appears ONLY in the objective, so its gradient stays finite while the
        NaN dual on a constraint over `y` poisons a different component -- and
        `x` is visited first. The user was told to bound `x`, which does not
        help, and the unbounded_box key was burnt so the NaN could never be
        reported for the rest of the run.
        """
        m = pyo.ConcreteModel()
        m.x = pyo.Var(within=pyo.NonNegativeReals, initialize=1.0)
        m.y = pyo.Var(bounds=(0, 10), initialize=1.0)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.c = pyo.Constraint(expr=m.y >= 1)       # x is NOT in it
        m.dual[m.c] = float("nan")
        m.obj = pyo.Objective(expr=-m.x + m.y)    # x needs an ub it lacks
        m._mpisppy_data = type(
            "_D", (), {"solution_available": True, "outer_bound": "UNSET"})()

        spoke = self._spoke_over(m)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spoke.lagrangian()
        messages = [str(w.message) for w in caught]
        self.assertIsNone(m._mpisppy_data.outer_bound, messages)
        self.assertTrue(any("non-finite value" in m for m in messages),
                        messages)
        self.assertFalse(
            any("unbounded below" in m for m in messages),
            f"the earlier unbounded component won the race: {messages}")

    def test_an_unknown_tag_does_not_abort_the_run(self):
        """A new return-None site recording a NEW tag used to KeyError inside
        lagrangian(), aborting the run every iteration."""
        from unittest import mock
        scenario = self._scenario_with_a_bound_and_a_missing_dual()
        spoke = self._spoke_over(scenario)

        def _records_an_unknown_tag(*args, **kwargs):
            kwargs["no_bound_reason"].append(("a_tag_from_the_future", "hi"))
            return None

        with mock.patch(
            "mpisppy.cylinders.ipopt_outer_bound.certified_lower_bound",
            side_effect=_records_an_unknown_tag,
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = spoke.lagrangian()       # must not raise
        self.assertEqual(result, "EBOUND")
        self.assertTrue(
            any("recorded no reason" in str(w.message) for w in caught),
            [str(w.message) for w in caught])

    def test_value_error_becomes_no_bound(self):
        scenario = self._scenario_with_uninitialized_var()
        spoke = self._spoke_over(scenario)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = spoke.lagrangian()           # must not raise
        self.assertEqual(result, "EBOUND")
        self.assertIsNone(scenario._mpisppy_data.outer_bound)
        self.assertTrue(any("no certificate" in str(w.message) for w in caught))


@unittest.skipUnless(ipopt_available, "ipopt is not available")
@unittest.skipUnless(comm.size == 2, "needs exactly two ranks")
class TestAgainstEFOptimum(unittest.TestCase):
    """End-to-end: a PH hub on a MIP solver, this spoke on Ipopt.

    This also exercises the routing claim in the design -- the hub and the spoke
    really do run different solvers on their own copies of the models -- and the
    Ebound reduction across the spoke's rank.
    """

    def _spin(self, hub_solver):
        cfg = _cfg(hub_solver=hub_solver)
        if hub_solver in ("glpk", "cbc"):
            # Neither can handle the PH hub's quadratic proximal term.
            cfg.linearize_proximal_terms = True
        beans, kwargs = _beans(cfg)
        hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=kwargs)
        spoke = vanilla.ipopt_outer_bound_spoke(
            *beans, scenario_creator_kwargs=kwargs)
        wheel = WheelSpinner(hub_dict, [spoke])
        wheel.spin()
        return wheel

    def _assert_valid_and_useful(self, wheel):
        if wheel.global_rank != 1:
            return
        bound = wheel.spcomm.bound
        self.assertIsNotNone(bound)
        # An outer bound on a minimization must not exceed the optimum.
        self.assertLessEqual(bound, FARMER_EF_OPT + 1e-6)
        # And it must be useful, not merely valid: farmer's Lagrangian bound
        # sits in the same neighborhood as the optimum, so a wildly negative
        # number would mean the certificate had collapsed.
        self.assertGreater(bound, 2.0 * FARMER_EF_OPT)

    def test_bound_does_not_exceed_the_ef_optimum(self):
        # farmer is an LP, so Ipopt can drive the hub too; this keeps the test
        # runnable anywhere Ipopt is, with no MIP solver needed.
        self._assert_valid_and_useful(self._spin("ipopt"))

    @unittest.skipUnless(dual_bound_solver_name, "no second solver available")
    def test_hub_and_spoke_can_use_different_solvers(self):
        # The routing claim in the design: each cylinder solves its own copy of
        # the models with its own solver, and the only coupling is the numeric
        # exchange of W and bounds. Here the hub runs something that is not
        # Ipopt while the spoke runs Ipopt.
        #
        # Gated on any working second solver rather than on a commercial MIP
        # solver. The claim under test is "different solvers", and requiring a
        # commercial one skipped this everywhere: the ipopt-tests job installs
        # none. glpk and cbc cannot take the PH hub's quadratic proximal term,
        # so _spin linearizes it when the hub solver is one of those.
        self.assertNotEqual(dual_bound_solver_name, "ipopt")
        self._assert_valid_and_useful(self._spin(dual_bound_solver_name))


@unittest.skipUnless(ipopt_available, "ipopt is not available")
@unittest.skipUnless(comm.size == 2, "needs exactly two ranks")
@unittest.skipUnless(dual_bound_solver_name, "no dual-bound-reporting solver")
class TestAgreesWithLagrangian(unittest.TestCase):
    """The strongest correctness check available: on a linear problem, compare
    this spoke's bound against the ordinary Lagrangian spoke's.

    farmer is an LP, so an LP solver's dual bound *is* the Lagrangian dual value
    -- exact, and arrived at by a completely different route than the tangent
    plane over the variable box that this spoke computes from Ipopt's duals.
    Two independent computations of the same quantity is a much sharper test
    than "the bound does not exceed the optimum", which a badly broken
    certificate could still pass by being very negative.

    Both legs use the same hub, same rho and same iteration count, so the hub
    walks the same W trajectory and the two spokes are asked for a bound on the
    same relaxations. Spoke bounds do not feed back into W.
    """

    TOL = 1e-2   # measured agreement is ~1.2e-4

    def _bound_from(self, spoke_factory, **cfg_overrides):
        cfg = _cfg(hub_solver="ipopt")
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
        beans, kwargs = _beans(cfg)
        hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=kwargs)
        spoke = spoke_factory(*beans, scenario_creator_kwargs=kwargs)
        wheel = WheelSpinner(hub_dict, [spoke])
        wheel.spin()
        return wheel

    def test_certificate_reproduces_the_lagrangian_bound(self):
        lag = self._bound_from(vanilla.lagrangian_spoke,
                               lagrangian_solver_name=dual_bound_solver_name)
        cert = self._bound_from(vanilla.ipopt_outer_bound_spoke)

        if lag.global_rank != 1:
            return
        lag_bound, cert_bound = lag.spcomm.bound, cert.spcomm.bound
        self.assertIsNotNone(lag_bound)
        self.assertIsNotNone(cert_bound)
        # A spoke that never sent anything leaves its bound at nan. That means
        # the comparison solver produced no dual bound after all, which is a
        # broken premise for this test rather than a failure of the spoke under
        # test -- say so instead of reporting a bogus mismatch.
        if math.isnan(lag_bound):
            self.skipTest(
                f"{dual_bound_solver_name} reported no dual bound for the "
                "Lagrangian spoke, so there is nothing to compare against")
        self.assertFalse(math.isnan(cert_bound),
                         "ipopt_outer_bound sent no bound at all")

        # Two independent computations of the same number.
        self.assertAlmostEqual(cert_bound, lag_bound, delta=self.TOL)

        # And in the safe direction: the certificate carries the box-correction
        # term and the cushion, so it may be slightly looser than the exact LP
        # dual bound but must never be more optimistic than it.
        self.assertLessEqual(cert_bound, lag_bound + self.TOL)

        # Both remain valid outer bounds.
        self.assertLessEqual(cert_bound, FARMER_EF_OPT + 1e-6)
        self.assertLessEqual(lag_bound, FARMER_EF_OPT + 1e-6)


if __name__ == "__main__":
    unittest.main()
