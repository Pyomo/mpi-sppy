###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""WheelSpinner reads per-cylinder rank_ratio dict keys and, when any is
non-default, apportions and reports the allocation then raises (the live
unequal-rank communicator path is not yet wired in). The uniform default
path is exercised by the cylinder test suite, not here.

The gate fires before any communicator is built, so a stub comm with just
Get_size/Get_rank is enough to drive it without MPI."""

import unittest

from mpisppy.spin_the_wheel import WheelSpinner


class _StubComm:
    """Minimal comm: the gate only needs size and rank."""
    def __init__(self, size, rank=0):
        self._size = size
        self._rank = rank

    def Get_size(self):
        return self._size

    def Get_rank(self):
        return self._rank


def _min_dict(role, **extra):
    # Dicts only need the keys WheelSpinner validates before the gate;
    # the classes are never instantiated on this path.
    d = {f"{role}_class": object, "opt_class": object}
    d.update(extra)
    return d


class TestFlexibleRankGate(unittest.TestCase):

    def test_nonuniform_ratio_raises_not_implemented(self):
        hub = _min_dict("hub", rank_ratio=1.0)
        spoke = _min_dict("spoke", rank_ratio=0.5)
        ws = WheelSpinner(hub, [spoke])
        with self.assertRaises(NotImplementedError):
            ws.run(comm_world=_StubComm(size=4))

    def test_nonuniform_with_too_few_ranks_raises_value_error(self):
        # When even the floor-of-one is infeasible, apportion_ranks raises
        # ValueError first (before the NotImplementedError gate).
        hub = _min_dict("hub", rank_ratio=1.0)
        spoke_a = _min_dict("spoke", rank_ratio=0.5)
        spoke_b = _min_dict("spoke", rank_ratio=0.5)
        ws = WheelSpinner(hub, [spoke_a, spoke_b])
        with self.assertRaises(ValueError):
            ws.run(comm_world=_StubComm(size=2))

    def test_explicit_uniform_ratios_pass_the_gate(self):
        # All ratios 1.0 -> gate is skipped; the next step (_make_comms) is
        # reached and fails on the stub (no Split). Reaching that point at
        # all proves the NotImplementedError gate did not fire.
        hub = _min_dict("hub", rank_ratio=1.0)
        spoke = _min_dict("spoke", rank_ratio=1.0)
        ws = WheelSpinner(hub, [spoke])
        with self.assertRaises(Exception) as ctx:
            ws.run(comm_world=_StubComm(size=2))
        self.assertNotIsInstance(ctx.exception, NotImplementedError)


if __name__ == "__main__":
    unittest.main()
