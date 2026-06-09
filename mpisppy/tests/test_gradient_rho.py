###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Author: Ulysse Naepels and D.L. Woodruff
"""
IMPORTANT:
  Unless we run to convergence, the solver, and even solver
version matter a lot, so we often just do smoke tests.
"""

import unittest
import numpy as np
from mpisppy.utils import config

import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.tests.examples.farmer as farmer
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver
from mpisppy.extensions.grad_rho import GradRho

__version__ = 0.3

solver_available,solver_name, persistent_available, persistent_solver_name= get_solver()


class _NanXhatBuf:
    """Stand-in for the BEST_XHAT receive buffer that ``post_iter0`` reads via
    ``_eval_grad_exprs``. An all-NaN array keeps the gradient evaluation on the
    current (iter0) Var values regardless of the ``eval_at_xhat`` setting, so
    the test exercises ``post_iter0`` without needing a live xhat spoke."""

    def __init__(self, ph):
        n = max(len(s._mpisppy_data.nonant_indices)
                for s in ph.local_scenarios.values())
        self._arr = np.full(n, np.nan)

    def value_array(self):
        return self._arr

def _create_cfg():
    cfg = config.Config()
    cfg.gradient_args()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.dynamic_rho_args()
    cfg.solver_name = solver_name
    cfg.default_rho = 1
    cfg.grad_order_stat = 0.5
    cfg.max_solver_threads = 1
    return cfg

#*****************************************************************************

class Test_gradient_farmer(unittest.TestCase):
    """ Test the gradient code using farmer."""

    def _create_ph_farmer(self):
        # This causes iter zero to execute, which is overkill to just create ph for farmer
        self.cfg.num_scens = 3
        scenario_creator = farmer.scenario_creator
        scenario_denouement = farmer.scenario_denouement
        all_scenario_names = farmer.scenario_names_creator(self.cfg.num_scens)
        scenario_creator_kwargs = farmer.kw_creator(self.cfg)
        beans = (self.cfg, scenario_creator, scenario_denouement, all_scenario_names)
        hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
        hub_dict['opt_kwargs']['options']['cfg'] = self.cfg                            
        list_of_spoke_dict = list()
        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()
        if wheel.strata_rank == 0:
            ph_object = wheel.spcomm.opt
            return ph_object

    def setUp(self):
        self.cfg = _create_cfg()
        self.cfg.max_iterations = 0
        self.ph_object = self._create_ph_farmer()
        self.ph_object.options["grad_rho_options"] = {"cfg": self.cfg}
        

    def test_grad_rho_init(self):
        self.grad_object = GradRho(self.ph_object)

    @staticmethod
    def _reference_xbar(ph):
        """The probability-weighted mean of the current nonant values per
        (node, i) -- the quantity ``_Compute_Xbar`` produces (serial run: all
        scenarios are local, so the global mean is just the local sum)."""
        acc = {}
        for s in ph.local_scenarios.values():
            for node in s._mpisppy_node_list:
                ndn = node.name
                prob = s._mpisppy_data.prob_coeff[ndn]
                for i, v in enumerate(node.nonant_vardata_list):
                    acc[(ndn, i)] = acc.get((ndn, i), 0.0) + prob * v._value
        return acc

    def test_post_iter0_uses_iter0_xbar_not_zero(self):
        # Regression test: grad_rho must compute xbar from the iter0 solutions
        # before it sets rho. PHBase.Iter0 runs the iter0 solve loop but does
        # not call Compute_Xbar (that first happens in iterk_loop, i.e. at
        # iteration 1), so when the post_iter0 hook fires the xbars Param is
        # still at its 0.0 init value. Without computing xbar in post_iter0 the
        # rho denominator abs(x - xbar) collapses to abs(x) -- distance from
        # zero rather than from the iter0 consensus mean.
        ph = self.ph_object

        # Reproduce the state Iter0 leaves the models in right before it calls
        # the post_iter0 hook: iter0 solutions still on the Vars, xbars == 0.
        for s in ph.local_scenarios.values():
            for ndn_i in s._mpisppy_data.nonant_indices:
                s._mpisppy_model.xbars[ndn_i]._value = 0.0

        expected_xbar = self._reference_xbar(ph)
        # The test only discriminates the fix if the true mean is nonzero.
        self.assertTrue(
            any(abs(v) > 1e-6 for v in expected_xbar.values()),
            "iter0 xbar is all ~0; test cannot distinguish the fix",
        )

        grad = GradRho(ph)
        grad.best_xhat_buf = _NanXhatBuf(ph)
        grad.post_iter0()

        for s in ph.local_scenarios.values():
            for ndn_i in s._mpisppy_data.nonant_indices:
                self.assertAlmostEqual(
                    s._mpisppy_model.xbars[ndn_i]._value,
                    expected_xbar[ndn_i],
                    places=9,
                    msg=f"xbar at {ndn_i} not set to the iter0 mean; "
                        "post_iter0 did not Compute_Xbar before computing rho",
                )


if __name__ == '__main__':
    unittest.main()
