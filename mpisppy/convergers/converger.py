###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
''' Base class for converger objects

    DTM Dec 2019

    Replaces the old implementation in which
    convergers were modules rather than classes.

    DLW: as of  March 2023 note that user supplied convergers do not compute
    ph.conv (which is computed as a scaled norm difference)
    and both ph.conv and the user supplied converger, could trigger convergence
    (see phbase.py)
'''

import abc

class Converger:
    ''' Abstract base class for converger monitors.

        Args:
            opt (SPBase): The SPBase object for the current model
    '''
    def __init__(self, opt):
        self.conv = None  # intended to be the value used for comparison

    @abc.abstractmethod
    def is_converged(self):
        ''' Indicated whether the algorithm has converged.

            Must return a boolean. If True, the algorithm will terminate at the
            current iteration--no more solves will be performed by SPBase.
            Otherwise, the iterations will continue.
        '''
        pass

    def post_loops(self):
        '''Method called after the termination of the algorithm.
            This method is called after the post_loops of any extensions
        '''
        pass
