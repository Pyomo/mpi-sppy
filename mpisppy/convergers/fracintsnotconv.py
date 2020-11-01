# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
""" fraction of ints converger to illustrate a custom converger that does not
    directly use mpi reduce calls.
    DLW Jan 2019
"""

import pyomo.environ as pyo
import mpisppy.convergers.converger

class FractionalConverger(mpisppy.convergers.converger.Converger):
    """ Illustrate a class to contain data used by the converger 
    NOTE: unlike the extensions, we get only our object back so
          we need to keep references to everything we might want
          to look at.
    Args:
        PHoptions (dict): keys are option names
        local_scenarios (dict): keys are names, 
                                vals are concrete models with attachments
        comms (dict): key is node name; val is a comm object
        rank (int): mpi process rank
    """
    def __init__(self, phb, rank, n_proc):
        PHoptions = phb.PHoptions
        local_scenarios = phb.local_scenarios
        comms = phb.comms
        self.name = "fractintsnotconv"
        self.verbose = PHoptions["verbose"]
        self._PHoptions = PHoptions
        self._local_scenarios = local_scenarios
        self._comms = comms
        self._rank = rank
        if self.verbose:
            print ("Created converger=",self.name)
        
    def convergence_value(self):
        """ compute the fraction of *not* converged ints
        Args:
            self (object): create by prep

        Returns:
            number of converged ints divided by number of ints
        """
        numints = 0
        numconv = 0
        for k,s in self._local_scenarios.items():
            nlens = s._PySP_nlens        
            for node in s._PySPnode_list:
                ndn = node.name
                for i in range(nlens[ndn]):
                    xvar = node.nonant_vardata_list[i]
                    if xvar.is_integer() or xvar.is_binary():
                        numints += 1
                        xb = pyo.value(s._xbars[(ndn,i)])
                        #print ("dlw debug",xb*xb, pyo.value(s._xsqbars[(ndn,i)]))
                        if xb * xb == pyo.value(s._xsqbars[(ndn,i)]):
                            numconv += 1
        if self.verbose:
            print (self.name,": numints=",numints)
        if numints > 0:
            retval = 1.0 - numconv / numints
        else:
            retval = 0
        if self.verbose:
            print (self.name,": convergence value=",retval)
        return retval

    def is_converged(self):
        """ check for convergence
        Args:
            self (object): create by prep

        Returns:
           converged?: True if converged, False otherwise
        """
        return self.convval < self._PHoptions['convthresh']
