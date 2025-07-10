###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import mpisppy.extensions.dyn_rho_base
import numpy as np
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.calculus.derivatives import Modes
import pyomo.environ as pyo
import mpisppy.MPI as MPI
from mpisppy import global_toc
import mpisppy.utils.sputils as sputils
from mpisppy.utils.sputils import nonant_cost_coeffs
from mpisppy.cylinders.spwindow import Field

class GradRho(mpisppy.extensions.dyn_rho_base.Dyn_Rho_extension_base):
    """
    Gradient-based rho from
    Gradient-baed rho Parameter for Progressive Hedging
    U. Naepels, David L. Woodruff, 2023
    """

    def __init__(self, opt):
        cfg = opt.options["grad_rho_options"]["cfg"]
        super().__init__(opt, cfg)
        self.opt = opt
        self.alpha = cfg.grad_order_stat
        assert (self.alpha >= 0 and self.alpha <= 1), f"For grad_order_stat 0 is the min, 0.5 the average, 1 the max; {self.alpha=} is invalid."

    def pre_iter0(self):
        pass

    def iter0_post_solver_creation(self):
        pass

    def post_iter0(self):
        pass

    def miditer(self):
        pass

    def enditer(self):
        pass

    def post_everything(self):
        pass