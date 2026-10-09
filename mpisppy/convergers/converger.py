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

    def checkpoint_state(self):
        '''Return this converger's own state as picklable plain data, or None.

            Same contract, and the same reason for existing, as
            ``Extension.checkpoint_state``: a converger that compares the
            current iterate against an earlier one keeps that earlier one on
            the converger object, where no model carries it and a resume
            would otherwise start it empty. A converger that recomputes
            everything from the current iterate has no state: override this
            to return None, and ``restore_state`` to do nothing.

            Optional, but not silently: the base raises, and a run with
            checkpointing or a resume refuses at startup a converger that
            does not override both. Convergers decide when the run *stops*,
            so getting this wrong is not just a divergence: a resumed run can
            terminate at a different iteration than the uninterrupted one
            would have.
        '''
        raise NotImplementedError(
            f"{type(self).__name__} does not implement checkpoint_state")

    def restore_state(self, state):
        '''Restore what checkpoint_state() returned, on a resumed run.

            Called once at the end of Iter0, after the converger has been
            constructed. Matched by class name, so a run resumed with a
            different converger is never handed the old one's state.

            Not collective, for the reason Extension.restore_state gives:
            the restore runs inside an agreement across the cylinder's ranks.
        '''
        raise NotImplementedError(
            f"{type(self).__name__} does not implement restore_state")
