###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""certified_outer_bound on a persistent solver.

A file of its own because it needs gurobipy and exactly two ranks, and fails
rather than skips without them; the CI ipopt-tests job installs gurobipy and
runs it with `mpiexec -np 2`.

spopt loads only the primal values from a persistent solver, so unless the
spoke loads the duals itself every constraint is taken with multiplier zero.
That bound is valid, which is why nothing else catches it, and on farmer it is
about seven times looser than the same solver's non-persistent interface gives.
"""

import unittest

import pyomo.environ as pyo

import mpisppy.tests.examples.farmer as farmer
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils import config

from mpi4py import MPI

comm = MPI.COMM_WORLD

# The three-scenario farmer optimum, from an EF solve.
FARMER_EF_OPT = -108390.0


def _bound_with_spoke_solver(spoke_solver):
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.certified_outer_bound_args()
    cfg.num_scens = 3
    cfg.max_iterations = 5
    cfg.default_rho = 1.0
    cfg.solver_name = "ipopt"
    cfg.certified_outer_bound_solver_name = spoke_solver
    beans = (cfg, farmer.scenario_creator, farmer.scenario_denouement,
             farmer.scenario_names_creator(cfg.num_scens))
    kwargs = farmer.kw_creator(cfg)
    hub = vanilla.ph_hub(*beans, scenario_creator_kwargs=kwargs)
    spoke = vanilla.certified_outer_bound_spoke(*beans,
                                                scenario_creator_kwargs=kwargs)
    wheel = WheelSpinner(hub, [spoke])
    wheel.spin()
    return wheel


class TestPersistentSolver(unittest.TestCase):

    def test_setup(self):
        self.assertEqual(comm.size, 2, "run this file with mpiexec -np 2")
        for name in ("ipopt", "gurobi", "gurobi_persistent"):
            self.assertTrue(pyo.SolverFactory(name).available(exception_flag=False),
                            f"{name} is not available (pip install gurobipy)")

    def test_persistent_bound_matches_the_plain_interface(self):
        # Same hub, same W trajectory, same solver underneath: the two
        # interfaces must give the same certified bound.
        plain = _bound_with_spoke_solver("gurobi")
        persistent = _bound_with_spoke_solver("gurobi_persistent")
        if plain.global_rank != 1:
            return
        plain_bound = plain.spcomm.bound
        persistent_bound = persistent.spcomm.bound
        self.assertLessEqual(persistent_bound, FARMER_EF_OPT + 1e-6)
        self.assertAlmostEqual(persistent_bound, plain_bound,
                               delta=1e-6 * abs(plain_bound))


if __name__ == "__main__":
    unittest.main()
