###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Deprecation guards for the legacy rho setters (issue #673).

SepRho still ships but is scheduled for deprecation: it must emit a
DeprecationWarning the first time it is instantiated, so downstream projects
have lead time to migrate (likely to GradRho). That test asserts only that the
warning is issued; full instantiation needs PH machinery, but the warning is the
first statement in ``__init__`` so it fires regardless.

ReducedCostsRho has been removed entirely (reduced-cost rho was never shown to be
effective in practice and did not support flexible rank assignments). The option
survives in Config only so its selection fails loudly: Config.checker() raises a
dated deprecation error rather than silently doing nothing.
"""

import unittest
import warnings

from mpisppy.extensions.sep_rho import SepRho
from mpisppy.utils.config import Config


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

    def test_reduced_costs_rho_selection_raises_dated_error(self):
        # ReducedCostsRho was removed; selecting the surviving option must fail
        # at config-check time with a dated deprecation message, not silently do
        # nothing (the whole point is to not let a custom driver be burned).
        cfg = Config()
        cfg.reduced_costs_rho_args()
        cfg.reduced_costs_rho = True
        with self.assertRaises(ValueError) as ctx:
            cfg.checker()
        msg = str(ctx.exception)
        self.assertIn("reduced_costs_rho", msg)
        self.assertIn("2026-06-14", msg)  # deprecation date
        self.assertIn("673", msg)


if __name__ == "__main__":
    unittest.main()
