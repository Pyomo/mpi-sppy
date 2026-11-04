###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import math
import unittest
from types import SimpleNamespace

import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.cylinders.spcommunicator import SPCommunicator
from mpisppy.tests.examples import farmer
from mpisppy.utils.config import Config


def _communicator(is_minimizing, initial_outer_bound=None):
    comm = SimpleNamespace(
        opt=SimpleNamespace(is_minimizing=is_minimizing),
        options={"initial_outer_bound": initial_outer_bound},
        strata_rank=0,
        latest_ob_char=None,
        last_ob_idx=None,
    )
    comm.OuterBoundUpdate = SPCommunicator.OuterBoundUpdate.__get__(comm)
    SPCommunicator.initialize_bound_values(comm)
    return comm


class TestInitialOuterBound(unittest.TestCase):
    def test_default_minimization_bound_is_unchanged(self):
        self.assertEqual(_communicator(True).BestOuterBound, -math.inf)

    def test_default_maximization_bound_is_unchanged(self):
        self.assertEqual(_communicator(False).BestOuterBound, math.inf)

    def test_minimization_initial_outer_bound(self):
        comm = _communicator(True, 12.5)
        self.assertEqual(comm.BestOuterBound, 12.5)
        self.assertEqual(comm.latest_ob_char, "I")

    def test_maximization_initial_outer_bound(self):
        self.assertEqual(_communicator(False, 12.5).BestOuterBound, 12.5)

    def test_later_updates_keep_the_stronger_bound(self):
        comm = _communicator(True, 12.5)
        comm.BestOuterBound = comm.OuterBoundUpdate(10.0)
        self.assertEqual(comm.BestOuterBound, 12.5)
        comm.BestOuterBound = comm.OuterBoundUpdate(14.0)
        self.assertEqual(comm.BestOuterBound, 14.0)

    def test_nonfinite_bound_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be finite"):
            _communicator(True, math.inf)

    def test_ph_hub_forwards_option(self):
        cfg = Config()
        cfg.popular_args()
        cfg.two_sided_args()
        cfg.ph_args()
        cfg.solver_name = "not-used"
        cfg.initial_outer_bound = 12.5
        hub = vanilla.ph_hub(
            cfg,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            ["Scenario1"],
        )
        self.assertEqual(
            hub["hub_kwargs"]["options"]["initial_outer_bound"], 12.5)


if __name__ == "__main__":
    unittest.main()
