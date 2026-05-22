###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import mpisppy.extension

class BatteryExtension(mpisppy.extension.Extension):

    def __init__(self, ph, rank, n_proc):
        self.cylinder_rank = rank
        self.ph = ph

    def pre_iter0(self):
        pass

    def post_iter0(self):
        pass

    def miditer(self, PHiter, conv):
        if (self.cylinder_rank == 0):
            print('{itr:3d} {conv:12.4e}'.format(itr=PHiter, conv=conv))

    def enditer(self, PHiter):
        pass

    def post_everything(self, PHiter, conv):
        pass
