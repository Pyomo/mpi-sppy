###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import logging
import random
import mpisppy.log

from mpisppy.extensions.xhatbase import XhatBase
from mpisppy.cylinders.xhatbase import XhatInnerBoundBase

# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger("mpisppy.cylinders.xhatshufflelooper_bounder",
                         "xhatclp.log",
                         level=logging.CRITICAL)                         
logger = logging.getLogger("mpisppy.cylinders.xhatshufflelooper_bounder")

class XhatShuffleInnerBound(XhatInnerBoundBase):

    converger_spoke_char = 'X'

    def xhat_extension(self):
        return XhatBase(self.opt)

    def xhat_prep(self):
        self.xhatter = super().xhat_prep()

        ## option drive this? (could be dangerous)
        self.random_seed = 42
        # Have a separate stream for shuffling
        self.random_stream = random.Random()


    def try_scenario_dict(self, xhat_scenario_dict):
        """ wrapper for _try_one"""
        snamedict = xhat_scenario_dict

        stage2EFsolvern = self.opt.options.get("stage2EFsolvern", None)
        branching_factors = self.opt.options.get("branching_factors", None)  # for stage2ef
        obj = self.xhatter._try_one(snamedict,
                                    solver_options = self.solver_options,
                                    verbose=False,
                                    restore_nonants=True,
                                    stage2EFsolvern=stage2EFsolvern,
                                    branching_factors=branching_factors)
        def _vb(msg): 
            if self.verbose and self.opt.cylinder_rank == 0:
                print ("(rank0) " + msg)

        if obj is None:
            _vb(f"    Infeasible {snamedict}")
            return False
        _vb(f"    Feasible {snamedict}, obj: {obj}")

        # XhatBase._try_one updates the solution cache in the opt object for us
        update = self.update_if_improving(obj, update_best_solution_cache=False)
        logger.debug(f'   bottom of try_scenario_dict on rank {self.global_rank}')
        return update

    def main(self):
        logger.debug(f"Entering main on xhatshuffle spoke rank {self.global_rank}")

        self.xhat_prep()
        if "reverse" in self.opt.options["xhat_looper_options"]:
            self.reverse = self.opt.options["xhat_looper_options"]["reverse"]
        else:
            self.reverse = True
        if "iter_step" in self.opt.options["xhat_looper_options"]:
            self.iter_step = self.opt.options["xhat_looper_options"]["iter_step"]
        else:
            self.iter_step = None
        self.solver_options = self.opt.options["xhat_looper_options"]["xhat_solver_options"]

        # give all ranks the same seed
        self.random_stream.seed(self.random_seed)
        
        #We need to keep track of the way scenario_names were sorted
        scen_names = list(enumerate(self.opt.all_scenario_names))
        
        # shuffle the scenarios associated (i.e., sample without replacement)
        shuffled_scenarios = self.random_stream.sample(scen_names, 
                                                       len(scen_names))
        
        scenario_cycler = ScenarioCycler(shuffled_scenarios,
                                         self.opt.nonleaves,
                                         self.reverse,
                                         self.iter_step)

        def _vb(msg): 
            if self.verbose and self.opt.cylinder_rank == 0:
                print("(rank0) " + msg)

        xh_iter = 1
        while not self.got_kill_signal():
            # When there is no iter0, the serial number must be checked.
            # (unrelated: uncomment the next line to see the source of delay getting an xhat)
            if self.get_serial_number() == 0:
                continue

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
                # just for sending the values to other scenarios
                # so we don't need to tell persistent solvers
                self.opt._restore_nonants(update_persistent=False)
                
                _vb("   Begin epoch")
                scenario_cycler.begin_epoch()

                # always try at least two for each set of nonants
                # so we continue to explore the scenarios and
                # do not stall out on a single scenario because
                # the hub is moving very fast
                next_scendict = scenario_cycler.get_next()
                if next_scendict is not None:
                    _vb(f"   Trying next {next_scendict}")
                    update = self.try_scenario_dict(next_scendict)
                    if update:
                        _vb(f"   Updating best to {next_scendict}")
                        scenario_cycler.best = next_scendict["ROOT"]

            next_scendict = scenario_cycler.get_next()
            if next_scendict is not None:
                _vb(f"   Trying next {next_scendict}")
                update = self.try_scenario_dict(next_scendict)
                if update:
                    _vb(f"   Updating best to {next_scendict}")
                    scenario_cycler.best = next_scendict["ROOT"]

            #_vb(f"    scenario_cycler._scenarios_this_epoch {scenario_cycler._scenarios_this_epoch}")

            xh_iter += 1


class ScenarioCycler:

    def __init__(self, shuffled_scenarios,nonleaves,reverse,iter_step):
        root_kids = nonleaves['ROOT'].kids if 'ROOT' in nonleaves else None
        if root_kids is None or len(root_kids)==0 or root_kids[0].is_leaf:
            self._multi = False
            self._iter_shift = 0 if iter_step is None else iter_step
            self._use_reverse = False #It is useless to reverse for 2stage SP
        else:
            self._multi = True
            self.BF0 = len(root_kids)
            self._nonleaves = nonleaves
            # TODO: is this right for multistage, or should the default be
            #       0 like in the two-stage case?
            self._iter_shift = self.BF0 if iter_step is None else iter_step
            self._use_reverse = True if reverse is None else reverse
            self._reversed = False #Do we iter in reverse mode ?
        self._shuffled_scenarios = shuffled_scenarios
        self._num_scenarios = len(shuffled_scenarios)
        
        self._cycle_idx = 0
        self._best = None
        self._begin_normal_epoch()
        

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
            if filling_idx == self._cycle_idx and 'ROOT' in self.nodescen_dict and self.nodescen_dict['ROOT'] is not None:
                print(self.nodescen_dict)
                raise RuntimeError("_fill_nodescen_dict looped over every scenario but was not able to find a scen for every nonleaf node.")
            sname = self._shuffled_snames[filling_idx]
            snum = self._original_order[filling_idx]
            
            def _add_sname_to_node(ndn):
                first = self._nonleaves[ndn].scenfirst
                last = self._nonleaves[ndn].scenlast
                if snum>=first and snum<=last:
                    self.nodescen_dict[ndn] = sname
                    return False
                else:
                    return True
            #Adding sname to every nodes it goes by, and removing the nodes from empty_nodes
            empty_nodes = list(filter(_add_sname_to_node,empty_nodes))
            filling_idx +=1
            filling_idx %= self._num_scenarios
        
    def _create_nodescen_dict(self):
        '''
        Creates an attribute nodescen_dict. 
        Keys are nonleaf names, values are local scenario names 
        (a value can be None if the associated scenario is not in our rank)
        
        WARNING: _cur_ROOTscen must be up to date when calling this method
        '''
        if not self._multi:
            self.nodescen_dict = {'ROOT':self._cur_ROOTscen}
        else:
            self.nodescen_dict = dict()
            self._fill_nodescen_dict(self._nonleaves.keys())
    
    def _update_nodescen_dict(self,snames_to_remove):
        '''
        WARNING: _cur_ROOTscen must be up to date when calling this method
        '''
        if not self._multi:
            self.nodescen_dict = {'ROOT':self._cur_ROOTscen}
        else:
            empty_nodes = []
            for ndn in self._nonleaves.keys():
                if self.nodescen_dict[ndn] in snames_to_remove:
                    self.nodescen_dict[ndn] = None
                    empty_nodes.append(ndn)
            self._fill_nodescen_dict(empty_nodes)
        

    def begin_epoch(self):
        if self._multi and self._use_reverse and not self._reversed:
            self._begin_reverse_epoch()
        else:
            self._begin_normal_epoch()
        
    def _begin_normal_epoch(self):
        if self._multi:
            self._reversed = False
        self._shuffled_snames = [s[1] for s in self._shuffled_scenarios]
        self._original_order = [s[0] for s in self._shuffled_scenarios]
        self._cur_ROOTscen = self._shuffled_snames[0] if self.best is None else self.best
        self._create_nodescen_dict()
        
        self._scenarios_this_epoch = set()
    
    def _begin_reverse_epoch(self):
        self._reversed = True
        self._shuffled_snames = [s[1] for s in reversed(self._shuffled_scenarios)]
        self._original_order = [s[0] for s in reversed(self._shuffled_scenarios)]
        self._cur_ROOTscen = self._shuffled_snames[0] if self.best is None else self.best
        self._create_nodescen_dict()
        
        self._scenarios_this_epoch = set()

    def get_next(self):
        next_scen = self._cur_ROOTscen
        next_scendict = self.nodescen_dict
        if next_scen in self._scenarios_this_epoch:
            return None
        self._scenarios_this_epoch.add(next_scen)
        self._iter_scen()
        return next_scendict

    def _iter_scen(self):
        old_idx = self._cycle_idx
        self._cycle_idx += self._iter_shift
        ## wrap around
        self._cycle_idx %= self._num_scenarios
        
        #do not reuse a previously visited scenario for 'ROOT'
        tmp_cycle_idx = self._cycle_idx
        while self._shuffled_snames[tmp_cycle_idx] in self._scenarios_this_epoch and (
                (tmp_cycle_idx+1)%self._num_scenarios != self._cycle_idx):
            tmp_cycle_idx +=1
            tmp_cycle_idx %= self._num_scenarios
        
        self._cycle_idx = tmp_cycle_idx
        
        #Updating scenarios
        self._cur_ROOTscen = self._shuffled_snames[self._cycle_idx]
        if old_idx<self._cycle_idx:
            scens_to_remove = set(self._shuffled_snames[old_idx:self._cycle_idx])
        else:
            scens_to_remove = set(self._shuffled_snames[old_idx:]+self._shuffled_snames[:self._cycle_idx])
        self._update_nodescen_dict(scens_to_remove)
        
    
