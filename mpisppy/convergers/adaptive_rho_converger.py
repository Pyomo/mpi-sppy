# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

import math
import mpisppy.convergers.converger

import mpi4py.MPI as MPI
import numpy as np

class AdaptiveRhoConverger(mpisppy.convergers.converger.Converger):

    def __init__(self, ph):
        ## TODO: this will never do anything unless the adaptive rho setter is also used
        ## TODO: the rho setter should leave a flag on the data block of the local subproblems to indicate
        ##       its use and this converger should check for it.
        if 'adaptive_rho_converger_options' in ph.PHoptions and \
                'verbose' in ph.PHoptions['adaptive_rho_converger_options'] and \
                ph.PHoptions['adaptive_rho_converger_options']['verbose']:
            self._verbose = True
        else:
            self._verbose = False
        self.ph = ph

    def _compute_rho_norm(self, ph):
        local_rho_norm = np.zeros(1)
        global_rho_norm = np.zeros(1)
        local_rho_norm[0] = sum(s.PySP_prob*sum( rho._value for rho in s._mpisppy_model.rho.values())\
                                for s in ph.local_scenarios.values() )
        ph.mpicomm.Allreduce(local_rho_norm, global_rho_norm, op=MPI.SUM)
        return float(global_rho_norm[0])

    def is_converged(self):
        """ check for convergence
        Args:
            self (object): create by prep

        Returns:
           converged?: True if converged, False otherwise
        """
        log_rho_norm = math.log(self._compute_rho_norm(self.ph))

        ret_val = log_rho_norm < self.ph.PHoptions['convthresh']
        if self._verbose and self.ph.cylinder_rank == 0:
            print(f"log(|rho|) = {log_rho_norm}")
            if ret_val:
                print("Adaptive Rho Convergence Check Passed")
            else:
                print("Adaptive Rho Convergence Check Failed "
                      f"(requires log(|rho|) < {self.ph.PHoptions['convthresh']}")
                print("Continuing PH with updated Rho")
        return ret_val 
