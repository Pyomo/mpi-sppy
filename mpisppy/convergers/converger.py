# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' Base class for converger objects

    DTM Dec 2019

    Replaces the old implementation in which
    convergers were modules rather than classes.
'''

import abc

class Converger:
    ''' Abstract base class for converger monitors.

        Args:
            opt (SPBase): The SPBase object for the current model
            rank (int): MPI process rank in MPI_COMM_WORLD
            n_proc (int): Total number of processes in MPI_COMM_WORLD
    '''
    def __init__(self, opt, rank, n_proc):
        pass

    @abc.abstractmethod
    def convergence_value(self):
        ''' Compute a convergence value at this iteration.

            This method takes no arguments. The user can access the SPBase
            object by attaching the SPBase object to self in the Extension
            constructor. This method is not required to return any
            arguments--however, whatever it returns (possibly NONE) is attached
            to the SPBase object, and is passed to the miditer() function of
            any extensions, if they are present.
        '''
        pass

    @abc.abstractmethod
    def is_converged(self):
        ''' Indicated whether the algorithm has converged.
            
            Must return a boolean. If True, the algorithm will terminate at the
            current iteration--no more solves will be performed by SPBase.
            Otherwise, the iterations will continue.
        '''
        pass
