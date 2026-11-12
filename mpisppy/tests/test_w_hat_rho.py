###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""WHatRho / nac_ef_w_hat on farmer."""

import unittest

import numpy as np

import mpisppy.tests.examples.farmer as farmer
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.extensions.w_hat_rho import WHatRho, nac_ef_w_hat
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver
from mpisppy.utils import config

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()


def _cfg():
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.dynamic_rho_args()
    cfg.w_hat_rho_args()
    cfg.solver_name = solver_name
    cfg.default_rho = 1
    cfg.max_solver_threads = 1
    cfg.num_scens = 3
    cfg.max_iterations = 0
    return cfg


@unittest.skipIf(not solver_available, "no solver is available")
class Test_w_hat_rho_farmer(unittest.TestCase):

    def setUp(self):
        self.cfg = _cfg()
        self.names = farmer.scenario_names_creator(3)
        self.kw = farmer.kw_creator(self.cfg)

    def test_w_hat_centered(self):
        w = nac_ef_w_hat(self.names, farmer.scenario_creator, self.kw,
                         solver_name=solver_name)
        W = np.array([w[n] for n in self.names])
        self.assertTrue(np.all(np.abs(W.mean(axis=0)) < 1e-6))
        self.assertGreater(np.abs(W).max(), 1e-3)

    def test_rho_installed(self):
        beans = (self.cfg, farmer.scenario_creator, farmer.scenario_denouement,
                 self.names)
        hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=self.kw)
        hub_dict["opt_kwargs"]["options"]["cfg"] = self.cfg
        wheel = WheelSpinner(hub_dict, [])
        wheel.spin()
        if wheel.strata_rank != 0:
            return
        ph = wheel.spcomm.opt
        ph.options["w_hat_rho_options"] = {"cfg": self.cfg,
                                           "scenario_creator_kwargs": self.kw}
        WHatRho(ph).post_iter0()
        rhos = [[r._value for r in s._mpisppy_model.rho.values()]
                for s in ph.local_scenarios.values()]
        self.assertTrue(np.all(np.array(rhos) > 0))
        for r in rhos[1:]:
            np.testing.assert_allclose(r, rhos[0])
        self.assertFalse(np.allclose(rhos[0], self.cfg.default_rho))


if __name__ == "__main__":
    unittest.main()
