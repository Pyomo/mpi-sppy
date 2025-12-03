###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import mpisppy.cylinders.spoke

class FrankWolfeOuterBound(mpisppy.cylinders.spoke.OuterBoundSpoke):

    converger_spoke_char = 'F'

    def main(self):
        self.opt.fwph_main()

    def is_converged(self):
        return self.got_kill_signal()

    def sync(self):
        # The FWPH spoke can call "sync" before it
        # even starts doing anything, so its possible
        # to get here without any bound information
        if not hasattr(self.opt, '_local_bound'):
            return
        # Tell the hub about the most recent bound
        self.bound = self.opt._local_bound

    def finalize(self):
        # The FWPH spoke can call "finalize" before it
        # even starts doing anything, so its possible
        # to get here without any bound information
        # if we terminated early
        if not hasattr(self.opt, '_local_bound'):
            return
        self.bound = self.opt._local_bound
        self.final_bound = self.opt._local_bound
        return self.final_bound
