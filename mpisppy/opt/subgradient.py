###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import mpisppy.phbase
import shutil
import mpisppy.MPI as mpi

class Subgradient(mpisppy.phbase.PHBase):
    """ Subgradient Algorithm """

    def subgradient_main(self, finalize=True):
        """ Execute the subgradient algorithm.

        Args:
            finalize (bool, optional, default=True):
                If True, call `PH.post_loops()`, if False, do not,
                and return None for Eobj

        Returns:
            tuple:
                Tuple containing

                conv (float):
                    The convergence value (not easily interpretable).
                Eobj (float or `None`):
                    If `finalize=True`, this is the expected, weighted
                    objective value. This value is not directly useful.
                    If `finalize=False`, this value is `None`.
                trivial_bound (float):
                    The "trivial bound", computed by solving the model with no
                    nonanticipativity constraints (immediately after iter 0).
        """
        verbose = self.options['verbose']
        smoothed = self.options['smoothed']
        if smoothed != 0:
            raise RuntimeError("Cannnot use smoothing with Subgradient algorithm")
        self.PH_Prep(attach_prox=False, attach_smooth=smoothed)

        if (verbose):
            print('Calling Subgradient Iter0 on global rank {}'.format(global_rank))
        trivial_bound = self.Iter0()
        if (verbose):
            print ('Completed Subgradient Iter0 on global rank {}'.format(global_rank))

        self.iterk_loop()

        if finalize:
            Eobj = self.post_loops(self.extensions)
        else:
            Eobj = None

        return self.conv, Eobj, trivial_bound

    def ph_main(self, finalize=True):
        # for working with a PHHub
        return self.subgradient_main(finalize=finalize)
