# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

import mpisppy.convergers.converger

import mpi4py.MPI as MPI

class AdaptiveRhoConverger(mpisppy.convergers.converger.Converger):

    def __init__(self, ph):
        ## TODO: this will never do anything unless the adaptive rho setter is also used
        self._log_rho_norm_convergence_tolerance = \
            ph.PHoptions['adaptive_rho_options']['log_rho_norm_convergence_tolerance']
        self.ph = ph

    def _compute_rho_norm(self, ph):
        local_rho_norm = np.zeros(1)
        global_rho_norm = np.zeros(1)
        local_rho_norm[0] = sum(s.PySP_prob*sum( rho._value for rho in s._mpisppy_model.rho)\
                                for s in ph.local_scenarios.values() )
        ph.mpicomm.Allreduce(local_rho_norm, global_rho_norm, op=MPI.SUM)
        return rho_norm

    def convergence_value(self):
        """ compute and set the rho convergence norm
        Args:
            self (object): create by prep

        Returns:
            None
        """
        self.rho_norm = self._compute_rho_norm(self.ph)

    def is_converged(self):
        """ check for convergence
        Args:
            self (object): create by prep

        Returns:
           converged?: True if converged, False otherwise
        """
        log_rho_norm = math.log(self.rho_norm)
        ph = self.ph

        ret_val = log_rho_norm < self._log_rho_norm_convergence_tolerance
        if ph.cylinder_rank == 0:
            print(f"log(|rho|) = {log_rho_norm}")
            if ret_val:
                print("Adaptive Rho Convergence Check Passed")
            else:
                print("Adaptive Rho Convergence Check Failed "
                      f"(requires log(|rho|) < {self._log_rho_norm_convergence_tolerance}")
                print("Continuing PH with updated Rho")
        return ret_val 
