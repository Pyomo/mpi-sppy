# This software is distributed under the 3-clause BSD License.
import logging
import time
import random
import mpisppy.log
import mpisppy.utils.sputils as sputils
import mpisppy.cylinders.spoke as spoke

from math import inf
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from mpisppy.phbase import PHBase
from mpisppy.extensions.xhatbase import XhatBase

# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger("mpisppy.cylinders.xhatshufflelooper_bounder",
                         "xhatclp.log",
                         level=logging.CRITICAL)                         
logger = logging.getLogger("mpisppy.cylinders.xhatshufflelooper_bounder")


# First, we'll write a custom PH class,
# which overwrites a lot of methods.
# Because the looper should never update
# the objective function, this should save
# some time, especially with persistent solvers

class XhatTryer(PHBase):

    def attach_Ws_and_prox(self):
        pass

    def attach_PH_to_objective(self, add_duals=False , add_prox=False):
        if add_duals:
            raise RuntimeError("XhatTryer has no notion of duals")
        if add_prox:
            raise RuntimeError("XhatTryer has no notion of prox")
        pass

    def solve_loop(self, solver_options=None,
                   use_scenarios_not_subproblems=False,
                   dtiming=False,
                   dis_W=True,
                   dis_prox=True,
                   gripe=False,
                   disable_pyomo_signal_handling=False,
                   tee=False,
                   verbose=False):
        """ Loop over self.local_subproblems and solve them in a manner 
            dicated by the arguments. In addition to changing the Var
            values in the scenarios, update _PySP_feas_indictor for each.

        ASSUMES:
            Every scenario already has a _solver_plugin attached.

        Args:
            solver_options (dict or None): the scenario solver options
            use_scenarios_not_subproblems (boolean): for use by bounds
            dtiming (boolean): indicates that timing should be reported
            dis_W (boolean): indicates that W should be disabled (and re-enabled)
            dis_prox (boolean): disable (and re-enable) prox term
            gripe (boolean): output a message if a solve fails
            disable_pyomo_signal_handling (boolean): set to true for asynch, 
                                                     ignored for persistent solvers.
            tee (boolean): show solver output to screen if possible
            verbose (boolean): indicates verbose output

        NOTE: I am not sure what happens with solver_options None for
              a persistent solver. Do options persist?

        NOTE: set_objective takes care of W and prox changes.
        """
        if not dis_W:
            raise RuntimeError("XhatTryer has no notion of W")
        if not dis_prox:
            raise RuntimeError("XhatTryer has no notion of prox")
        def _vb(msg): 
            if verbose and self.rank == self.rank0:
                print ("(rank0) " + msg)
        logger.debug("  early solve_loop for rank={}".format(self.rank))

        # note that when there is no bundling, scenarios are subproblems
        if use_scenarios_not_subproblems:
            s_source = self.local_scenarios
        else:
            s_source = self.local_subproblems
        for k,s in s_source.items():
            logger.debug("  in loop solve_loop k={}, rank={}".format(k, self.rank))

            ## no need to recompute objective
            solve_start_time = time.time()
            if (solver_options):
                _vb("Using sub-problem solver options="
                    + str(solver_options))
                for option_key,option_value in solver_options.items():
                    s._solver_plugin.options[option_key] = option_value

            solve_keyword_args = dict()
            if self.rank == self.rank0:
                if tee is not None and tee is True:
                    solve_keyword_args["tee"] = True
            if (sputils.is_persistent(s._solver_plugin)):
                solve_keyword_args["save_results"] = False
            elif disable_pyomo_signal_handling:
                solve_keyword_args["use_signal_handling"] = False

            try:
                results = s._solver_plugin.solve(s,
                                                 **solve_keyword_args,
                                                 load_solutions=False)
                solve_err = False
            except:
                solve_err = True
                
            pyomo_solve_time = time.time() - solve_start_time
            if solve_err or (results.solver.status != SolverStatus.ok) \
                  or (results.solver.termination_condition \
                        != TerminationCondition.optimal):
                 s._PySP_feas_indicator = False

                 if gripe:
                     print ("Solve failed for scenario", s.name)
                     if not solve_err:
                         print ("status=", results.solver.status)
                         print ("TerminationCondition=",
                                results.solver.termination_condition)
            else:
                 if sputils.is_persistent(s._solver_plugin):
                     s._solver_plugin.load_vars()
                 else:
                     s.solutions.load_from(results)
                 s._PySP_lb = results.Problem[0].Lower_bound
                 s._PySP_feas_indicator = True
            # TBD: get this ready for IPopt (e.g., check feas_prob every time)
            # propogate down
            if hasattr(s,"_PySP_subscen_names"): # must be a bundle
                for sname in s._PySP_subscen_names:
                     self.local_scenarios[sname]._PySP_feas_indicator\
                         = s._PySP_feas_indicator

        if dtiming:
            all_pyomo_solve_times = self.mpicomm.gather(pyomo_solve_time, root=0)
            if self.rank == self.rank0:
                print("Pyomo solve times (seconds):")
                print("\tmin=%4.2f mean=%4.2f max=%4.2f" %
                      (np.min(all_pyomo_solve_times),
                      np.mean(all_pyomo_solve_times),
                      np.max(all_pyomo_solve_times)))

    def Update_W(self, verbose):
        pass

    def _disable_prox(self):
        pass

    def _disable_W_and_prox(self):
        pass

    def _disable_W(self):
        pass

    def _reenable_prox(self):
        pass

    def _reenable_W_and_prox(self):
        pass

    def _reenable_W(self):
        pass


class XhatShuffleInnerBound(spoke.InnerBoundNonantSpoke):

    def xhatbase_prep(self):
        if self.opt.multistage:
            raise RuntimeError('The XhatShuffleInnerBound only supports '
                               'two-stage models at this time.')

        verbose = self.opt.options['verbose']
        if "bundles_per_rank" in self.opt.options\
           and self.opt.options["bundles_per_rank"] != 0:
            raise RuntimeError("xhat spokes cannot have bundles (yet)")

        if not isinstance(self.opt, XhatTryer):
            raise RuntimeError("XhatShuffleInnerBound must be used with XhatTryer.")
            
        xhatter = XhatBase(self.opt)

        self.opt.PH_Prep(attach_duals=False, attach_prox=False)  
        logger.debug(f"  xhatshuffle spoke back from PH_Prep rank {self.rank_global}")

        self.opt.subproblem_creation(verbose)

        ### begin iter0 stuff
        xhatter.pre_iter0()
        self.opt._save_original_nonants()
        self.opt._create_solvers()

        teeme = False
        if "tee-rank0-solves" in self.opt.options:
            teeme = self.opt.options['tee-rank0-solves']

        self.opt.solve_loop(
            solver_options=self.opt.current_solver_options,
            dtiming=False,
            gripe=True,
            tee=teeme,
            verbose=verbose
        )
        self.opt._update_E1()  # Apologies for doing this after the solves...
        if abs(1 - self.opt.E1) > self.opt.E1_tolerance:
            if self.rank_global == self.opt.rank0:
                print("ERROR")
                print("Total probability of scenarios was ", self.opt.E1)
                print("E1_tolerance = ", self.opt.E1_tolerance)
            quit()
        feasP = self.opt.feas_prob()
        if feasP != self.opt.E1:
            if self.rank_global == self.opt.rank0:
                print("ERROR")
                print("Infeasibility detected; E_feas, E1=", feasP, self.opt.E1)
            quit()
        ### end iter0 stuff

        xhatter.post_iter0()
        self.opt._save_nonants() # make the cache

        ## for later
        self.verbose = self.opt.options["verbose"] # typing aid  
        self.solver_options = self.opt.options["xhat_looper_options"]["xhat_solver_options"]
        self.is_minimizing = self.opt.is_minimizing
        self.xhatter = xhatter

        ## option drive this? (could be dangerous)
        self.random_seed = 42


    def try_scenario(self, scenario):
        obj = self.xhatter._try_one({"ROOT":scenario},
                            solver_options = self.solver_options,
                            verbose=False)
        def _vb(msg): 
            if self.verbose and self.opt.rank == self.opt.rank0:
                print ("(rank0) " + msg)

        if obj is None:
            _vb(f"    Infeasible {scenario}")
            return False
        _vb(f"    Feasible {scenario}, obj: {obj}")

        ib = self.ib

        ## update if we improve the current bound from this spoke
        update = (obj < ib) if self.is_minimizing else (ib < obj)
        # send a bound to the opt companion
        if update:
            self.bound = obj 
            self.ib = obj
            logger.debug(f'   send inner bound={obj} on rank {self.rank_global} (based on scenario {scenario})')
        logger.debug(f'   bottom of try_scenario on rank {self.rank_global}')
        return update

    def main(self):
        verbose = self.opt.options["verbose"] # typing aid  
        logger.debug(f"Entering main on xhatshuffle spoke rank {self.rank_global}")

        self.xhatbase_prep()
        self.ib = inf if self.is_minimizing else -inf

        # give all ranks the same seed
        random.seed(self.random_seed)
        # shuffle the scenarios (i.e., sample without replacement)
        shuffled_scenarios = random.sample(self.opt.all_scenario_names,
                                            len(self.opt.all_scenario_names))
        scenario_cycler = ScenarioCycler(shuffled_scenarios)

        def _vb(msg): 
            if self.verbose and self.opt.rank == self.opt.rank0:
                print ("(rank0) " + msg)

        xh_iter = 1
        while not self.got_kill_signal():
            if (xh_iter-1) % 10000 == 0:
                logger.debug(f'   Xhatshuffle loop iter={xh_iter} on rank {self.rank_global}')
                logger.debug(f'   Xhatshuffle got from opt on rank {self.rank_global}')

            if self.new_nonants:
                logger.debug(f'   *Xhatshuffle loop iter={xh_iter}')
                logger.debug(f'   *got a new one! on rank {self.rank_global}')
                logger.debug(f'   *localnonants={str(self.localnonants)}')

                # update the caches
                self.opt._put_nonant_cache(self.localnonants)
                self.opt._restore_nonants()

                # reset the scenarios we've tried
                # so far
                scenario_cycler.begin_epoch()

                # try the best so far when their are new nonant
                best_scenario = scenario_cycler.best
                _vb(f"   Trying best {best_scenario}")
                if best_scenario is not None:
                    self.try_scenario(best_scenario)

                ## we could continue here -- as is, this will try
                ## at least two scenarios per new_nonants, the best 
                ## and the next off the cycle.

            next_scenario = scenario_cycler.get_next()
            if next_scenario is not None:
                _vb(f"   Trying next {next_scenario}")
                update = self.try_scenario(next_scenario)
                if update:
                    _vb(f"   Updating best to {next_scenario}")
                    scenario_cycler.best = next_scenario

            #_vb(f"    scenario_cycler._scenarios_this_epoch {scenario_cycler._scenarios_this_epoch}")

            xh_iter += 1


class ScenarioCycler:

    def __init__(self, all_scenario_list):
        self._all_scenario_list = all_scenario_list
        self._num_scenarios = len(all_scenario_list)
        self._cycle_idx = 0
        self._cur_scen = all_scenario_list[0]
        self._scenarios_this_epoch = set()
        self._best = None

    @property
    def best(self):
        self._scenarios_this_epoch.add(self._best)
        return self._best

    @best.setter
    def best(self, value):
        self._best = value

    def begin_epoch(self):
        self._scenarios_this_epoch = set()

    def get_next(self):
        next_scen = self._cur_scen
        if next_scen in self._scenarios_this_epoch:
            self._iter_scen()
            return None
        self._scenarios_this_epoch.add(next_scen)
        self._iter_scen()
        return next_scen

    def _iter_scen(self):
        self._cycle_idx += 1
        ## wrap around
        self._cycle_idx %= self._num_scenarios
        self._cur_scen = self._all_scenario_list[self._cycle_idx]
