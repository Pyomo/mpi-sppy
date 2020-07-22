# This software is distributed under the 3-clause BSD License.
import logging
import random
import mpisppy.log
import mpisppy.utils.sputils as sputils
import mpisppy.cylinders.spoke as spoke

from math import inf
from mpisppy.utils.xhat_tryer import XhatTryer
from mpisppy.extensions.xhatbase import XhatBase

# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger("mpisppy.cylinders.xhatshufflelooper_bounder",
                         "xhatclp.log",
                         level=logging.CRITICAL)                         
logger = logging.getLogger("mpisppy.cylinders.xhatshufflelooper_bounder")

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
        # Have a separate stream for shuffling
        self.random_stream = random.Random()


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
        self.random_stream.seed(self.random_seed)
        # shuffle the scenarios (i.e., sample without replacement)
        shuffled_scenarios = self.random_stream.sample(
            self.opt.all_scenario_names,
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
