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


class TestSetupGuards(unittest.TestCase):
    """The guards that belong to the spoke rather than the certificate engine.

    Constructed without running a wheel: the guard reads self.opt.options, so a
    lightweight stand-in exercises it without a solve.
    """

    def _guard_with_solver(self, solver_name):
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound

        class _Stub:
            options = {"solver_name": solver_name}
            local_scenarios = {}

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


class TestCollectiveWarning(unittest.TestCase):
    """The conditions this spoke warns about are rank-local; Ebound is not.

    Gating on `cylinder_rank == 0` silenced the rank that actually saw the
    problem, leaving an empty bound column and no explanation.
    """

    def _spoke(self, rank, size, speaking_rank):
        from mpisppy.cylinders.ipopt_outer_bound import IpoptOuterBound

        class _Comm:
            def __init__(self):
                self.size = size

            def Get_rank(self):
                return rank

            def allreduce(self, value, op=None):
                return speaking_rank

        spoke = IpoptOuterBound.__new__(IpoptOuterBound)
        spoke.cylinder_rank = rank
        spoke.cylinder_comm = _Comm()
        spoke._warned = set()
        return spoke

    def _warn(self, spoke, local_flag):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spoke._warn_once_collectively("k", local_flag, lambda: "saw it")
        return [str(w.message) for w in caught]

    def test_the_rank_that_saw_it_speaks_even_when_it_is_not_rank_zero(self):
        spoke = self._spoke(rank=1, size=2, speaking_rank=1)
        self.assertEqual(self._warn(spoke, True), ["saw it"])

    def test_the_other_ranks_stay_quiet(self):
        spoke = self._spoke(rank=0, size=2, speaking_rank=1)
        self.assertEqual(self._warn(spoke, False), [])

    def test_silence_when_no_rank_saw_it(self):
        spoke = self._spoke(rank=0, size=2, speaking_rank=2)   # == size
        self.assertEqual(self._warn(spoke, False), [])
        self.assertNotIn("k", spoke._warned)   # not consumed, can still fire

    def test_key_is_consumed_on_every_rank_so_it_fires_only_once(self):
        spoke = self._spoke(rank=0, size=2, speaking_rank=1)
        self._warn(spoke, False)
        self.assertIn("k", spoke._warned)
        spoke2 = self._spoke(rank=1, size=2, speaking_rank=1)
        self.assertEqual(self._warn(spoke2, True), ["saw it"])
        self.assertEqual(self._warn(spoke2, True), [])


class TestCertificateFailureStandsDown(unittest.TestCase):
    """Evaluating phi at the returned point can raise things that are not
    CertificateError, and none of them is worth aborting the wheel."""

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

    def test_value_error_becomes_no_bound(self):
        scenario = self._scenario_with_uninitialized_var()
        spoke = self._spoke_over(scenario)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = spoke._solve_and_certify()   # must not raise
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

    @unittest.skipUnless(mip_available, "no MIP solver available")
    def test_hub_and_spoke_can_use_different_solvers(self):
        # The routing claim in the design: each cylinder solves its own copy of
        # the models with its own solver, and the only coupling is the numeric
        # exchange of W and bounds. Here the hub runs a MIP solver while the
        # spoke runs Ipopt.
        self._assert_valid_and_useful(self._spin(mip_solver_name))


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
