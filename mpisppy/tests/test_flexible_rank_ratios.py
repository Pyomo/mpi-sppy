###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""WheelSpinner's flexible-rank branch (the equal-vs-unequal decision in
``run``), exercised without MPI via a stub comm.

Phase 1 walled the unequal path off behind a NotImplementedError; the
communication-layer cut removes that wall and builds per-cylinder
communicators instead. These tests pin the surviving startup behavior that
needs no MPI:

  * an infeasible request (more cylinders than ranks) still errors early,
    from apportion_ranks, before any communicator is built; and
  * a feasible non-uniform request is no longer rejected -- it proceeds past
    the (now removed) gate into communicator construction, where the bare
    stub (which has no Split) fails with something *other* than
    NotImplementedError.

The full unequal path is exercised end to end by the mpiexec integration
test; here we only need Get_size/Get_rank to drive the branch decision."""

import unittest

from mpisppy.spin_the_wheel import WheelSpinner


class _StubComm:
    """Minimal comm: the branch decision only needs size and rank. It has no
    Split, so reaching communicator construction surfaces as AttributeError."""
    def __init__(self, size, rank=0):
        self._size = size
        self._rank = rank

    def Get_size(self):
        return self._size

    def Get_rank(self):
        return self._rank


def _min_dict(role, **extra):
    # Dicts only need the keys WheelSpinner validates before the branch;
    # the classes are never instantiated on this path.
    d = {f"{role}_class": object, "opt_class": object}
    d.update(extra)
    return d


class TestFlexibleRankBranch(unittest.TestCase):

    def test_nonuniform_with_too_few_ranks_raises_value_error(self):
        # Two 0.5-ratio spokes + a hub is three cylinders; with only two ranks
        # the floor-of-one is infeasible, so apportion_ranks raises ValueError
        # before any communicator construction is attempted.
        hub = _min_dict("hub", rank_ratio=1.0)
        spoke_a = _min_dict("spoke", rank_ratio=0.5)
        spoke_b = _min_dict("spoke", rank_ratio=0.5)
        ws = WheelSpinner(hub, [spoke_a, spoke_b])
        with self.assertRaises(ValueError):
            ws.run(comm_world=_StubComm(size=2))

    def test_nonuniform_no_longer_raises_not_implemented(self):
        # A feasible non-uniform request used to raise NotImplementedError.
        # Now it proceeds to build communicators; on the bare stub (no Split)
        # that surfaces as some other exception -- proving the gate is gone.
        hub = _min_dict("hub", rank_ratio=1.0)
        spoke = _min_dict("spoke", rank_ratio=0.5)
        ws = WheelSpinner(hub, [spoke])
        with self.assertRaises(Exception) as ctx:
            ws.run(comm_world=_StubComm(size=4))
        self.assertNotIsInstance(ctx.exception, NotImplementedError)

    def test_uniform_ratios_use_equal_path(self):
        # All ratios 1.0 -> equal path; reaches _make_comms and fails on the
        # stub (no Split). Reaching that point at all confirms the unequal
        # branch was not taken and nothing raised NotImplementedError.
        hub = _min_dict("hub", rank_ratio=1.0)
        spoke = _min_dict("spoke", rank_ratio=1.0)
        ws = WheelSpinner(hub, [spoke])
        with self.assertRaises(Exception) as ctx:
            ws.run(comm_world=_StubComm(size=2))
        self.assertNotIsInstance(ctx.exception, NotImplementedError)


if __name__ == "__main__":
    unittest.main()
