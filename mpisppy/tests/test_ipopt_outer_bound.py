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
        return spoke

    def test_non_ipopt_solver_is_rejected(self):
        spoke = self._guard_with_solver("gurobi")
        with self.assertRaisesRegex(CertificateError, "scoped to Ipopt"):
            spoke._check_setup_guards()

    def test_ipopt_variants_are_accepted(self):
        # ipopt_v2 and similar names still name Ipopt.
        for name in ("ipopt", "ipopt_v2"):
            with self.subTest(name=name):
                self._guard_with_solver(name)._check_setup_guards()

    def test_missing_solver_name_is_rejected(self):
        spoke = self._guard_with_solver(None)
        with self.assertRaisesRegex(CertificateError, "scoped to Ipopt"):
            spoke._check_setup_guards()


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
