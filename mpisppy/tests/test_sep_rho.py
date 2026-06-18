###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for the SEP rho setter in mpisppy/extensions/sep_rho.py.

SepRho (Watson & Woodruff, 2011) sets, per non-anticipative variable with a
nonzero objective coefficient c:

    integer    var:  rho = |c| / (xmax - xmin + 1) * multiplier
    continuous var:  rho = |c| / max(1, E|x - xbar|) * multiplier

and leaves rho untouched where c == 0.  These tests drive the real extension
code (including its single-rank MPI all-reduces) with a controlled stand-in PH
so the per-variable arithmetic can be checked exactly, with no solver.
"""

import types
import warnings
import unittest

import pyomo.environ as pyo

import mpisppy.MPI as MPI
from mpisppy.extensions.sep_rho import SepRho

NDN = "ROOT"
NLEN = 4
# index -> is this nonant integer-valued?
INTEGER_FLAGS = [False, True, False, False]
# the same objective coefficients are used for every scenario
COSTS = [10.0, -15.0, 7.0, 0.0]
# nonant 3 has c == 0, so the heuristic cannot determine its rho; it falls back
# to the positive default rho (DEFAULT_PH_RHO), overwriting this seed (#560)
INIT_RHOS = [99.0, 99.0, 99.0, 42.0]
DEFAULT_PH_RHO = 1.0
# probability-weighted mean of each nonant across the two scenarios below
XBARS = [15.0, 6.0, 10.25, 1.5]
# per-scenario nonant values (rows are scenarios)
SCEN_VALS = {
    "Scen1": [10.0, 4, 10.0, 1.0],
    "Scen2": [20.0, 8, 10.5, 2.0],
}


class _Holder:
    """Mimics a mutable Pyomo Param data object (only ._value is read/written)."""
    def __init__(self, value):
        self._value = value


class _Scenario:
    """Hashable scenario stand-in (SepRho caches cost coeffs keyed on the
    scenario object, and types.SimpleNamespace is unhashable)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _make_scenario(name, prob):
    sm = pyo.ConcreteModel()

    def _domain(m, i):
        return pyo.Integers if INTEGER_FLAGS[i] else pyo.Reals

    sm.v = pyo.Var(range(NLEN), domain=_domain)
    for i, val in enumerate(SCEN_VALS[name]):
        sm.v[i]._value = val

    nonant_vardata_list = [sm.v[i] for i in range(NLEN)]
    node = types.SimpleNamespace(name=NDN, nonant_vardata_list=nonant_vardata_list)

    data = types.SimpleNamespace(
        nlens={NDN: NLEN},
        nonant_indices={(NDN, i): sm.v[i] for i in range(NLEN)},
        nonant_cost_coeffs={(NDN, i): COSTS[i] for i in range(NLEN)},
    )
    model = types.SimpleNamespace(
        xbars={(NDN, i): _Holder(XBARS[i]) for i in range(NLEN)},
        rho={(NDN, i): _Holder(INIT_RHOS[i]) for i in range(NLEN)},
    )
    # keep a handle to the Pyomo model alive so the Vars are not gc'd
    return _Scenario(
        _pyomo_model=sm,
        _mpisppy_node_list=[node],
        _mpisppy_data=data,
        _mpisppy_model=model,
        _mpisppy_probability=prob,
    )


def _make_ph(multiplier=1.0):
    scenarios = {
        "Scen1": _make_scenario("Scen1", 0.5),
        "Scen2": _make_scenario("Scen2", 0.5),
    }
    return types.SimpleNamespace(
        cylinder_rank=0,
        _PHIter=0,
        local_scenarios=scenarios,
        comms={NDN: MPI.COMM_WORLD},
        options={"defaultPHrho": DEFAULT_PH_RHO,
                 "sep_rho_options": {"cfg": types.SimpleNamespace(),
                                     "multiplier": multiplier}},
    )


def _make_ext(ph):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return SepRho(ph)


@unittest.skipIf(MPI.COMM_WORLD.Get_size() > 1,
                 "serial unit test; run without mpiexec")
class Test_sep_rho_reductions(unittest.TestCase):
    """The cross-scenario all-reduce helpers compute xmax, xmin, and E|x-xbar|."""

    def setUp(self):
        self.ph = _make_ph()
        self.ext = _make_ext(self.ph)

    def test_xmax(self):
        xmax = SepRho._compute_xmax(self.ph)
        self.assertEqual(xmax[(NDN, 0)], 20.0)
        self.assertEqual(xmax[(NDN, 1)], 8.0)
        self.assertEqual(xmax[(NDN, 2)], 10.5)

    def test_xmin(self):
        xmin = SepRho._compute_xmin(self.ph)
        self.assertEqual(xmin[(NDN, 0)], 10.0)
        self.assertEqual(xmin[(NDN, 1)], 4.0)
        self.assertEqual(xmin[(NDN, 2)], 10.0)

    def test_primal_residual_is_probability_weighted_abs_gap(self):
        resid = self.ext._compute_primal_residual_norm(self.ph)
        # 0.5*|10-15| + 0.5*|20-15| = 5.0
        self.assertAlmostEqual(resid[(NDN, 0)], 5.0)
        # 0.5*|4-6| + 0.5*|8-6| = 2.0
        self.assertAlmostEqual(resid[(NDN, 1)], 2.0)
        # 0.5*|10-10.25| + 0.5*|10.5-10.25| = 0.25
        self.assertAlmostEqual(resid[(NDN, 2)], 0.25)


@unittest.skipIf(MPI.COMM_WORLD.Get_size() > 1,
                 "serial unit test; run without mpiexec")
class Test_sep_rho_formula(unittest.TestCase):
    def _run(self, multiplier=1.0):
        ph = _make_ph(multiplier=multiplier)
        ext = _make_ext(ph)
        ext.compute_and_update_rho()
        return ph

    def _rho(self, ph, sname, i):
        return ph.local_scenarios[sname]._mpisppy_model.rho[(NDN, i)]._value

    def test_continuous_nonant_uses_residual(self):
        ph = self._run()
        # |10| / max(1, 5.0) = 2.0
        for sname in ("Scen1", "Scen2"):
            self.assertAlmostEqual(self._rho(ph, sname, 0), 2.0)

    def test_continuous_residual_clamped_at_one(self):
        ph = self._run()
        # residual 0.25 < 1, so denominator is max(1, 0.25) = 1 -> |7|/1 = 7.0
        self.assertAlmostEqual(self._rho(ph, "Scen1", 2), 7.0)

    def test_integer_nonant_uses_range_and_abs_cost(self):
        ph = self._run()
        # |-15| / (8 - 4 + 1) = 3.0  (note abs of a negative coefficient)
        for sname in ("Scen1", "Scen2"):
            self.assertAlmostEqual(self._rho(ph, sname, 1), 3.0)

    def test_zero_cost_coeff_uses_default_rho(self):
        ph = self._run()
        # c == 0 -> heuristic is uninformative -> positive default rho
        # (issue #560: fall back and report, rather than leave a stale seed)
        self.assertEqual(self._rho(ph, "Scen1", 3), DEFAULT_PH_RHO)
        self.assertEqual(self._rho(ph, "Scen2", 3), DEFAULT_PH_RHO)

    def test_multiplier_scales_every_updated_rho(self):
        ph = self._run(multiplier=2.0)
        self.assertAlmostEqual(self._rho(ph, "Scen1", 0), 4.0)   # 2.0 * 2
        self.assertAlmostEqual(self._rho(ph, "Scen1", 1), 6.0)   # 3.0 * 2
        self.assertAlmostEqual(self._rho(ph, "Scen1", 2), 14.0)  # 7.0 * 2
        # the zero-cost nonant falls back to the (unscaled) default rho
        self.assertEqual(self._rho(ph, "Scen1", 3), DEFAULT_PH_RHO)

    def test_multiplier_default_reads_from_options(self):
        # multiplier defaults to 1.0 when omitted from sep_rho_options
        ph = _make_ph()
        del ph.options["sep_rho_options"]["multiplier"]
        ext = _make_ext(ph)
        self.assertEqual(ext.multiplier, 1.0)


if __name__ == "__main__":
    unittest.main()
