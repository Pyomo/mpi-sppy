###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""SepRho and ReducedCostsRho are scheduled for deprecation per
https://github.com/Pyomo/mpi-sppy/issues/673. Both must emit a
DeprecationWarning the first time a user instantiates them so downstream
projects have lead time to migrate (likely to GradRho).

These tests assert only that the warning is issued; full instantiation
needs PH machinery the rho setters expect, but the deprecation warning
is the first thing each ``__init__`` does, so it fires regardless of
whether the rest of construction succeeds.
"""

import unittest
import warnings

from mpisppy.extensions.sep_rho import SepRho
from mpisppy.extensions.reduced_costs_rho import ReducedCostsRho


def _capture_first_warning(callable_, *, category):
    """Invoke ``callable_`` and return the first warning of ``category``
    captured during the call, or None. Swallows any exception raised after
    the warning fires; the deprecation warning is the first statement in
    each ``__init__`` so it is independent of downstream construction."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category)
        try:
            callable_()
        except Exception:
            pass
    matches = [w for w in caught if issubclass(w.category, category)]
    return matches[0] if matches else None


class _StubPH:
    """Just enough surface for the rho setters' first read of options.

    Construction will fail beyond the deprecation warning (Extension's
    base class needs more attributes) and that's fine — we only assert on
    the warning, which fires before any of those reads."""

    def __init__(self, key):
        self.options = {key: {"cfg": object()}}


class TestRhoDeprecations(unittest.TestCase):

    def test_sep_rho_emits_deprecation_warning(self):
        warning = _capture_first_warning(
            lambda: SepRho(_StubPH("sep_rho_options")),
            category=DeprecationWarning,
        )
        self.assertIsNotNone(
            warning,
            msg="SepRho.__init__ did not emit a DeprecationWarning",
        )
        self.assertIn("SepRho", str(warning.message))
        self.assertIn("673", str(warning.message))

    def test_reduced_costs_rho_emits_deprecation_warning(self):
        warning = _capture_first_warning(
            lambda: ReducedCostsRho(_StubPH("reduced_costs_rho_options")),
            category=DeprecationWarning,
        )
        self.assertIsNotNone(
            warning,
            msg="ReducedCostsRho.__init__ did not emit a DeprecationWarning",
        )
        self.assertIn("ReducedCostsRho", str(warning.message))
        self.assertIn("673", str(warning.message))


if __name__ == "__main__":
    unittest.main()
