# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

# Adapted from the PySP extension adaptive_rho_converger, authored by Gabriel Hackebeil

import math
import mpisppy.convergers.converger

import mpi4py.MPI as MPI
import numpy as np

class NormRhoConverger(mpisppy.convergers.converger.Converger):

    def __init__(self, ph):
        ## TODO: this will never do anything unless the norm rho updater is also used
        ## TODO: the rho updater should leave a flag on the data block of the local subproblems to indicate
        ##       its use and this converger should check for it.
        if 'nrom_rho_converger_options' in ph.PHoptions and \
                'verbose' in ph.PHoptions['norm_rho_converger_options'] and \
                ph.PHoptions['norm_rho_converger_options']['verbose']:
            self._verbose = True
        else:
            self._verbose = False
        self.ph = ph

    def _compute_rho_norm(self, ph):
        local_rho_norm = np.zeros(1)
        global_rho_norm = np.zeros(1)
        local_rho_norm[0] = sum(s._mpisppy_probability*sum( rho._value for rho in s._mpisppy_model.rho.values())\
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
                print("Norm rho convergence check passed")
            else:
                print("Adaptive rho convergence check failed "
                      f"(requires log(|rho|) < {self.ph.PHoptions['convthresh']}")
                print("Continuing PH with updated rho")
        return ret_val 
