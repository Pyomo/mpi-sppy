# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import os
import time
import logging
import random
import mpisppy.log
import mpisppy.utils.sputils as sputils
import mpisppy.cylinders.spoke as spoke
import mpi4py.MPI as mpi

from math import inf, isclose
from mpisppy.utils.xhat_eval import Xhat_Eval
from mpisppy.extensions.xhatbase import XhatBase

# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger("mpisppy.cylinders.xhatshufflelooper_bounder",
                         "xhatclp.log",
                         level=logging.CRITICAL)                         
logger = logging.getLogger("mpisppy.cylinders.xhatshufflelooper_bounder")

class XhatShuffleInnerBound(spoke.InnerBoundNonantSpoke):

    converger_spoke_char = 'X'

    def xhatbase_prep(self):
        if self.opt.multistage:
            raise RuntimeError('The XhatShuffleInnerBound only supports '
                               'two-stage models at this time.')

        verbose = self.opt.options['verbose']
        if "bundles_per_rank" in self.opt.options\
           and self.opt.options["bundles_per_rank"] != 0:
            raise RuntimeError("xhat spokes cannot have bundles (yet)")

        # Start code to support running trace. TBD: factor this up?
        if self.cylinder_rank == 0 and \
                'suffle_running_trace_prefix' in self.opt.options and \
                self.opt.options['shuffle_running_trace_prefix'] is not None:
            running_trace_prefix =\
                            self.opt.options['shuffle_running_trace_prefix']

            filen = running_trace_prefix+self.__class__.__name__+'.csv'
            if os.path.exists(filen):
                raise RuntimeError(f"running trace file {filen} already exists!")
            with open(filen, 'w') as f:
                f.write("time,scen,value\n")
            self.running_trace_filen = filen
        else:
            self.running_trace_filen = None
        # end code to support running trace
        
        if not isinstance(self.opt, Xhat_Eval):
            raise RuntimeError("XhatShuffleInnerBound must be used with Xhat_Eval.")
            
        xhatter = XhatBase(self.opt)

        ### begin iter0 stuff
        xhatter.pre_iter0()
        self.opt._save_original_nonants()

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
            raise ValueError(f"Total probability of scenarios was {self.opt.E1}"+\
                                 f"E1_tolerance = {self.opt.E1_tolerance}")
        infeasP = self.opt.infeas_prob()
        if infeasP != 0.:
            raise ValueError(f"Infeasibility detected; E_infeas={infeasP}")

        ### end iter0 stuff

        xhatter.post_iter0()
        self.opt._save_nonants() # make the cache

        ## for later
        self.verbose = self.opt.options["verbose"] # typing aid  
        self.solver_options = self.opt.options["xhat_looper_options"]["xhat_solver_options"]
        self.xhatter = xhatter

        ## option drive this? (could be dangerous)
        self.random_seed = 42
        # Have a separate stream for shuffling
        self.random_stream = random.Random()


    def try_scenario(self, scenario):
        obj = self.xhatter._try_one({"ROOT":scenario},
                            solver_options = self.solver_options,
                            verbose=False,
                            restore_nonants=False)
        def _vb(msg): 
            if self.verbose and self.opt.cylinder_rank == 0:
                print ("(rank0) " + msg)

        if self.running_trace_filen is not None:
            with open(self.running_trace_filen, "a") as f:
                f.write(f"{time.time()},{scenario},{obj}\n")
        if obj is None:
            _vb(f"    Infeasible {scenario}")
            return False
        _vb(f"    Feasible {scenario}, obj: {obj}")

        update = self.update_if_improving(obj)
        logger.debug(f'   bottom of try_scenario on rank {self.global_rank}')
        return update

    def main(self):
        verbose = self.opt.options["verbose"] # typing aid  
        logger.debug(f"Entering main on xhatshuffle spoke rank {self.global_rank}")

        self.xhatbase_prep()

        # give all ranks the same seed
        self.random_stream.seed(self.random_seed)
        # shuffle the scenarios (i.e., sample without replacement)
        shuffled_scenarios = self.random_stream.sample(
            self.opt.all_scenario_names,
            len(self.opt.all_scenario_names))
        scenario_cycler = ScenarioCycler(shuffled_scenarios)

        def _vb(msg): 
            if self.verbose and self.opt.cylinder_rank == 0:
                print("(rank0) " + msg)

        xh_iter = 1
        while not self.got_kill_signal():

            if (xh_iter-1) % 100 == 0:
                logger.debug(f'   Xhatshuffle loop iter={xh_iter} on rank {self.global_rank}')
                logger.debug(f'   Xhatshuffle got from opt on rank {self.global_rank}')

            if self.new_nonants:
                # similar to above, not all ranks will agree on
                # when there are new_nonants (in the same loop)
                logger.debug(f'   *Xhatshuffle loop iter={xh_iter}')
                logger.debug(f'   *got a new one! on rank {self.global_rank}')
                logger.debug(f'   *localnonants={str(self.localnonants)}')

                # update the caches
                self.opt._put_nonant_cache(self.localnonants)
                self.opt._restore_nonants()

            next_scenario = scenario_cycler.get_next()
            if next_scenario is not None:
                _vb(f"   Trying next {next_scenario}")
                update = self.try_scenario(next_scenario)
                if update:
                    _vb(f"   Updating best to {next_scenario}")
                    scenario_cycler.best = next_scenario
            else:
                scenario_cycler.begin_epoch()

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
