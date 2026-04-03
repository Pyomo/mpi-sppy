###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# An extension to compute and output avg, min, max for
# a component (e.g., first stage cost).
# DLW, Feb 2019
# This extension uses options["avgminmax_name"]

import mpisppy.extensions.xhatbase

class MinMaxAvg(mpisppy.extensions.xhatbase.XhatBase):
    """
    Args:
        ph (PH object): the calling object
    """
    def __init__(self, ph):
        super().__init__(ph)
        self.compstr = self.opt.options["avgminmax_name"]

    def pre_iter0(self):
        return

    def post_iter0(self):
        avgv, minv, maxv = self.opt.avg_min_max(self.compstr)
        if (self.cylinder_rank == 0):
            print ("  ### ", self.compstr,": avg, min, max, max-min", avgv, minv, maxv, maxv-minv)
        
    def miditer(self):
        return

    def enditer(self):
        avgv, minv, maxv = self.opt.avg_min_max(self.compstr)
        if (self.cylinder_rank == 0):
            print ("  ### ", self.compstr,": avg, min, max, max-min", avgv, minv, maxv, maxv-minv)

    def post_everything(self):
        return


