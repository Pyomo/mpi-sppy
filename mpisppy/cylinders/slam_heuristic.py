# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import abc
import logging
import time
import random
import logging
import mpisppy.log
import mpisppy.utils.sputils as sputils
import mpisppy.cylinders.spoke as spoke
import mpisppy.MPI as mpi
import pyomo.environ as pyo
import numpy as np

from mpisppy.utils.xhat_eval import Xhat_Eval
from math import inf

# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger("mpisppy.cylinders.slam_heuristic",
                         "slamheur.log",
                         level=logging.CRITICAL)                         
logger = logging.getLogger("mpisppy.cylinders.slam_heuristic")

class _SlamHeuristic(spoke.InnerBoundNonantSpoke):

    @property
    @abc.abstractmethod
    def numpy_op(self):
        pass

    @property
    @abc.abstractmethod
    def mpi_op(self):
        pass

    def slam_heur_prep(self):
        if self.opt.multistage:
            raise RuntimeError(f'The {self.__class__.__name__} only supports '
                               'two-stage models at this time.')
        if not isinstance(self.opt, Xhat_Eval):
            raise RuntimeError(f"{self.__class__.__name__} must be used with Xhat_Eval.")
        verbose = self.opt.options['verbose']

        logger.debug(f"{self.__class__.__name__} spoke back from PH_Prep rank {self.global_rank}")

        self.tee = False
        if "tee-rank0-solves" in self.opt.options:
            self.tee = self.opt.options['tee-rank0-solves']

        self.verbose = verbose

        self.opt._update_E1()
        self.opt._lazy_create_solvers()

    def extract_local_candidate_soln(self):
        num_scen = len(self.opt.local_scenarios)
        num_vars = len(self.localnonants) // num_scen
        assert(num_scen * num_vars == len(self.localnonants))
        ## matrix with num_scen rows and num_vars columns
        nonant_matrix = np.reshape(self.localnonants, (num_scen, num_vars))

        ## maximize almong the local sceanrios
        local_candidate = self.numpy_op(nonant_matrix, axis=0)
        assert len(local_candidate) == num_vars
        return local_candidate

    def main(self):
        self.slam_heur_prep()

        slam_iter = 1
        while not self.got_kill_signal():
            if (slam_iter-1) % 10000 == 0:
                logger.debug(f'   {self.__class__.__name__} loop iter={slam_iter} on rank {self.global_rank}')
                logger.debug(f'   {self.__class__.__name__} got from opt on rank {self.global_rank}')

            if self.new_nonants:
                
                local_candidate = self.extract_local_candidate_soln()

                global_candidate = np.empty_like(local_candidate)

                self.cylinder_comm.Allreduce(local_candidate, global_candidate, op=self.mpi_op)

                '''
                ## round the candidate
                candidate = global_candidate.round()
                '''

                # Everyone has the candidate solution at this point
                for s in self.opt.local_scenarios.values():
                    is_pers = sputils.is_persistent(s._solver_plugin)
                    solver = s._solver_plugin if is_pers else None

                    for ix, var in enumerate(s._mpisppy_data.nonant_indices.values()):
                        var.fix(global_candidate[ix])
                        if (is_pers):
                            solver.update_var(var)

                obj = self.opt.calculate_incumbent(fix_nonants=False)

                self.update_if_improving(obj)
                
            slam_iter += 1

class SlamUpHeuristic(_SlamHeuristic):

    converger_spoke_char = 'U'

    @property
    def numpy_op(self):
        return np.amax

    @property
    def mpi_op(self):
        return mpi.MAX

class SlamDownHeuristic(_SlamHeuristic):

    converger_spoke_char = 'D'

    @property
    def numpy_op(self):
        return np.amin

    @property
    def mpi_op(self):
        return mpi.MIN
