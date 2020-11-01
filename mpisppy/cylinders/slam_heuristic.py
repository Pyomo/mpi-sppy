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
import mpi4py.MPI as mpi
import pyomo.environ as pyo
import numpy as np

from mpisppy.utils.xhat_tryer import XhatTryer
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
        if not isinstance(self.opt, XhatTryer):
            raise RuntimeError(f"{self.__class__.__name__} must be used with XhatTryer.")
        verbose = self.opt.options['verbose']

        self.opt.PH_Prep(attach_duals=False, attach_prox=False)  
        logger.debug(f"{self.__class__.__name__} spoke back from PH_Prep rank {self.rank_global}")

        self.opt.subproblem_creation(verbose)

        '''
        ## do some checks
        for sname, s in self.opt.local_scenarios.values():
            for var in s._nonant_indexes.values():
                if not var.is_integer():
                    raise Exception(f"{self.__class__.__name__} can only be used for problems "
                                    "with pure-integer first-stage variables")
        '''

        self.opt._update_E1()

        self.opt._create_solvers()

        self.tee = False
        if "tee-rank0-solves" in self.opt.options:
            self.tee = self.opt.options['tee-rank0-solves']

        self.verbose = verbose
        self.is_minimizing = self.opt.is_minimizing

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

        self.ib = inf if self.is_minimizing else -inf

        slam_iter = 1
        while not self.got_kill_signal():
            if (slam_iter-1) % 10000 == 0:
                logger.debug(f'   {self.__class__.__name__} loop iter={slam_iter} on rank {self.rank_global}')
                logger.debug(f'   {self.__class__.__name__} got from opt on rank {self.rank_global}')

            if self.new_nonants:
                
                local_candidate = self.extract_local_candidate_soln()

                global_candidate = np.empty_like(local_candidate)

                self.intracomm.Allreduce(local_candidate, global_candidate, op=self.mpi_op)

                '''
                ## round the candidate
                candidate = global_candidate.round()
                '''

                # Everyone has the candidate solution at this point
                # Moreover, we are guaranteed that it is feasible
                bundling = self.opt.bundling
                for (sname, s) in self.opt.local_subproblems.items():
                    is_pers = sputils.is_persistent(s._solver_plugin)
                    solver = s._solver_plugin if is_pers else None

                    nonant_source = s.ref_vars.values() if bundling else \
                            s._nonant_indexes.values()

                    for ix, var in enumerate(nonant_source):
                        var.fix(global_candidate[ix])
                        if (is_pers):
                            solver.update_var(var)

                obj = self.opt.calculate_incumbent(fix_nonants=False)

                update = (obj is not None) and \
                         ((obj < self.ib) if self.is_minimizing else (self.ib < obj))

                if update:
                    self.bound = obj
                    self.ib = obj
                
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
