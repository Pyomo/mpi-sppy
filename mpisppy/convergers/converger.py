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
    '''
    def __init__(self, opt):
        pass

    @abc.abstractmethod
    def is_converged(self):
        ''' Indicated whether the algorithm has converged.
            
            Must return a boolean. If True, the algorithm will terminate at the
            current iteration--no more solves will be performed by SPBase.
            Otherwise, the iterations will continue.
        '''
        pass
