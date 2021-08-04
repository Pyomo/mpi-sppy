# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import os
import time
import logging
import random
import numpy as np
import mpisppy.log
import mpisppy.utils.sputils as sputils
import mpisppy.cylinders.spoke as spoke
import mpi4py.MPI as mpi
fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

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


    def try_scenario_dict(self, xhat_scenario_dict):
        snamedict = xhat_scenario_dict
        obj = self.xhatter._try_one(snamedict,
                            solver_options = self.solver_options,
                            verbose=False,
                            restore_nonants=False)
        def _vb(msg): 
            if self.verbose and self.opt.cylinder_rank == 0:
                print ("(rank0) " + msg)

        if self.running_trace_filen is not None:
            with open(self.running_trace_filen, "a") as f:
                f.write(f"{time.time()},{snamedict},{obj}\n")
        if obj is None:
            _vb(f"    Infeasible {snamedict}")
            return False
        _vb(f"    Feasible {snamedict}, obj: {obj}")

        update = self.update_if_improving(obj)
        logger.debug(f'   bottom of try_scenario_dict on rank {self.global_rank}')
        return update

    def main(self):
        verbose = self.opt.options["verbose"] # typing aid  
        logger.debug(f"Entering main on xhatshuffle spoke rank {self.global_rank}")

        self.xhatbase_prep()

        # give all ranks the same seed
        self.random_stream.seed(self.random_seed)
        
        # shuffle the scenarios associated (i.e., sample without replacement)
        shuffled_scenarios = self.random_stream.sample(self.opt.all_scenario_names, 
                                                       len(self.opt.all_scenario_names))
        scenario_cycler = ScenarioCycler(shuffled_scenarios,
                                         self.opt.nonleaves)

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

            next_scendict = scenario_cycler.get_next()
            if next_scendict is not None:
                _vb(f"   Trying next {next_scendict}")
                update = self.try_scenario_dict(next_scendict)
                if update:
                    _vb(f"   Updating best to {next_scendict}")
                    scenario_cycler.best = next_scendict
            else:
                scenario_cycler.begin_epoch()

            #_vb(f"    scenario_cycler._scenarios_this_epoch {scenario_cycler._scenarios_this_epoch}")

            xh_iter += 1


class ScenarioCycler:

    def __init__(self, shuffled_snames,nonleaves):
        root_kids = nonleaves['ROOT'].kids
        if root_kids[0].is_leaf:
            self._multi = False
            self._iter_shift = 1
        else:
            self._multi = True
            self.BF0 = len(root_kids)
            self._nonleaves = nonleaves
            
            self._iter_shift = self.BF0
            self._reversed = False #Do we iter in reverse mode ?
        
        self._shuffled_snames = shuffled_snames
        self._num_scenarios = len(shuffled_snames)
        self._cycle_idx = 0
        self._cur_ROOTscen = shuffled_snames[0]
        self.create_nodescen_dict()
        
        self._scenarios_this_epoch = set()
        self._best = None

    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, value):
        self._best = value
        
    def _fill_nodescen_dict(self,empty_nodes):   
        filling_idx = self._cycle_idx
        while len(empty_nodes) >0:
            #Sanity check to make no infinite loop.
            if filling_idx == self._cycle_idx and 'ROOT' in self.nodescen_dict:
                raise RuntimeError("_fill_nodescen_dict looped over every scenario but was not able to find a scen for every nonleaf node.")
            sname = self._shuffled_snames[filling_idx]
            
            def _add_sname_to_node(ndn):
                first = self._nonleaves[ndn].scenfirst
                last = self._nonleaves[ndn].scenlast
                snum = sputils.extract_num(sname)
                if snum>=first and snum<=last:
                    self.nodescen_dict[ndn] = sname
                    return False
                else:
                    return True
            #Adding sname to every nodes it goes by, and removing the nodes from empty_nodes
            empty_nodes = list(filter(_add_sname_to_node,empty_nodes))
            filling_idx +=1
            filling_idx %= self._num_scenarios
        
    def create_nodescen_dict(self):
        '''
        Creates an attribute nodescen_dict. 
        Keys are nonleaf names, values are local scenario names 
        (a value can be None if the associated scenario is not in our rank)
        '''
        if not self._multi:
            raise RuntimeWarning("Using create_nodescen_dict for 2stage problems is deprecated. Use directly self._cycle_idx instead")
        self.nodescen_dict = dict()
        self._fill_nodescen_dict(self._nonleaves.keys())
    
    def update_nodescen_dict(self,snames_to_remove):
        raise RuntimeError("Do not work yet")
        

    def begin_epoch(self):
        self._scenarios_this_epoch = set()

    def get_next(self):
        next_scen = self._cur_ROOTscen
        next_scendict = self.nodescen_dict
        if next_scen in self._scenarios_this_epoch:
            self._iter_scen()
            if self._reversed or not self._multi:
                #For 2stage problem, a scen can be used for 'ROOT' only once
                #For a multi stage problem, it can be used twice (including reverse)
                return None
        self._scenarios_this_epoch.add(next_scen)
        self._iter_scen()
        return next_scendict

    def _iter_scen(self):
        if len(self._scenarios_this_epoch) == self._num_scenarios:
            if self._multi and not self._reversed:
                self.reverse()
        self._cycle_idx += self._iter_shift
        ## wrap around
        self._cycle_idx %= self._num_scenarios
        
        #do not reuse a previously visited scenario for 'ROOT'
        tmp_cycle_idx = self._cycle_idx
        while tmp_cycle_idx in self._scenarios_this_epoch and (
                (tmp_cycle_idx+1)%self._num_scenarios != self._cycle_idx):
            tmp_cycle_idx +=1
            tmp_cycle_idx %= self._num_scenarios
        self._cycle_idx = tmp_cycle_idx
        
        #Updating scenarios
        self._cur_ROOTscen = self._shuffled_snames[self._cycle_idx]
        self.create_nodescen_dict()
        #TODO Make a better update system for multistage
        # if self._multi:
        #     self.nodescen_dict = {'ROOT': self._cur_ROOTscen}
        # else:
        #     self.create_nodescen_dict()
        
    
    def reverse(self):
        self._shuffled_snames = list(reversed(self._shuffled_snames))
        self._reversed = True
        self._scenarios_this_epoch = set() #Resetting the set of visited scenarios
        self._cycle_idx = 0
