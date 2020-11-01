# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
from pyomo.repn.standard_repn import generate_standard_repn
from mpi4py import MPI
from mpisppy.utils.lshaped_cuts import LShapedCutGenerator

import numpy as np
import pyomo.environ as pyo
import mpisppy.cylinders.spoke as spoke

from mpisppy import tt_timer

class CrossScenarioCutSpoke(spoke.Spoke):
    def __init__(self, spbase_object, fullcomm, intercomm, intracomm):
        super().__init__(spbase_object, fullcomm, intercomm, intracomm)

    def make_windows(self):
        nscen = len(self.opt.all_scenario_names)
        if nscen == 0:
            raise RuntimeError(f"(rank: {self.rank}), no local_scenarios")

        self.nscen = nscen
        vbuflen = 0
        self.nonant_per_scen = 0
        for s in self.opt.local_scenarios.values():
            vbuflen += len(s._nonant_indexes)
        local_scen_count = len(self.opt.local_scenario_names)
        self.nonant_per_scen = int(vbuflen / local_scen_count)

        ## the _locals will also have the kill signal
        self.all_nonant_len = vbuflen
        self.all_eta_len = nscen*local_scen_count
        self._locals = np.zeros(nscen*local_scen_count + vbuflen + 1)
        self._coefs = np.zeros(nscen*(nscen + self.nonant_per_scen) + 1 + 1)
        self._new_locals = False

        # local, remote
        # send, receive
        self._make_windows(nscen*(self.nonant_per_scen + 1 + 1), nscen*local_scen_count + vbuflen)

    def _got_kill_signal(self):
        ''' returns True if a kill signal was received,
            and refreshes the array and _locals'''
        self._new_locals = self.spoke_from_hub(self._locals)
        kill = (self._locals[-1] == -1)
        return kill

    def prep_cs_cuts(self):
        # create a map scenario -> index, this index is used for various lists containing scenario dependent info.
        self.scenario_to_index = { scen : indx for indx, scen in enumerate(self.opt.all_scenario_names) }

        # create concrete model to use as pseudo-master
        self.opt.master = pyo.ConcreteModel()

        ##get the nonants off an arbitrary scenario
        arb_scen = self.opt.local_scenarios[self.opt.local_scenario_names[0]]
        non_ants = arb_scen._PySPnode_list[0].nonant_vardata_list

        # add copies of the nonanticipatory variables to the master problem
        # NOTE: the LShaped code expects the nonant vars to be in a particular
        #       order and with a particular *name*.
        #       We're also creating an index for reference against later 
        nonant_vid_to_copy_map = dict()
        master_vars = list()
        for v in non_ants:
            non_ant_copy = pyo.Var(name=v.name)
            self.opt.master.add_component(v.name, non_ant_copy)
            master_vars.append(non_ant_copy)
            nonant_vid_to_copy_map[id(v)] = non_ant_copy

        self.opt.master_vars = master_vars

        # create an index of these non_ant_copies to be in the same
        # order as PH, used below
        nonants = dict()
        for ndn_i, nonant in arb_scen._nonant_indexes.items():
            vid = id(nonant)
            nonants[ndn_i] = nonant_vid_to_copy_map[vid]

        self.master_nonants = nonants
        self.opt.master.eta = pyo.Var(self.opt.all_scenario_names)

        self.opt.master.bender = LShapedCutGenerator()
        self.opt.master.bender.set_input(master_vars=self.opt.master_vars, 
                                            tol=1e-4, comm=self.intracomm)
        self.opt.master.bender.set_ls(self.opt)

        ## the below for loop can take some time,
        ## so return early if we get a kill signal,
        ## but only after a barrier
        self.intracomm.Barrier()
        if self.got_kill_signal():
            return

        # add the subproblems for all
        for scen in self.opt.all_scenario_names:
            subproblem_fn_kwargs = dict()

            # need to modify this to accept in user kwargs as well
            subproblem_fn_kwargs['scenario_name'] = scen
            self.opt.master.bender.add_subproblem(subproblem_fn=self.opt.create_subproblem,
                                                 subproblem_fn_kwargs=subproblem_fn_kwargs,
                                                 master_eta=self.opt.master.eta[scen],
                                                 subproblem_solver=self.opt.options["sp_solver"],
                                                 subproblem_solver_options=self.opt.options["sp_solver_options"])

        ## the above for loop can take some time,
        ## so return early if we get a kill signal,
        ## but only after a barrier
        self.intracomm.Barrier()
        if self.got_kill_signal():
            return

        ## This call is blocking, depending on the
        ## configuration. This necessitates the barrier
        ## above.
        self.opt.set_eta_bounds()
        self._eta_lb_array = np.fromiter(
                (self.opt.valid_eta_lb[s] for s in self.opt.all_scenario_names),
                dtype='d', count=len(self.opt.all_scenario_names))
        self.make_eta_lb_cut()

    def make_eta_lb_cut(self):
        ## we'll be storing a matrix as an array
        ## row_len is the length of each row
        row_len = 1+1+len(self.master_nonants)
        all_coefs = np.zeros( self.nscen*row_len+1, dtype='d')
        for idx, k in enumerate(self.opt.all_scenario_names):
            ## cut_array -- [ constant, eta_coef, *nonant_coefs ]
            ## this cut  -- [ LB, -1, *0s ], i.e., -1*\eta + LB <= 0
            all_coefs[row_len*idx] = self._eta_lb_array[idx]
            all_coefs[row_len*idx+1] = -1
        self.spoke_to_hub(all_coefs)

    def make_cut(self):

        ## cache opt
        opt = self.opt

        ## unpack these the way they were packed:
        all_nonants_and_etas = self._locals
        nonants = dict()
        etas = dict()
        ci = 0
        for k, s in opt.local_scenarios.items():
            for ndn, i in s._nonant_indexes:
                nonants[k, ndn, i] = all_nonants_and_etas[ci]
                ci += 1

        # get all the etas
        for k, s in opt.local_scenarios.items():
            for sn in opt.all_scenario_names:
                etas[k, sn] = all_nonants_and_etas[ci]
                ci += 1

        ## self.nscen == len(opt.all_scenario_names)
        # compute local min etas
        min_eta_vals = np.fromiter(( min(etas[k,sn] for k in opt.local_scenarios) \
                                       for sn in opt.all_scenario_names ),
                                    dtype='d', count=self.nscen)
        # Allreduce the etas to take the minimum
        global_eta_vals = np.empty(self.nscen, dtype='d')
        self.intracomm.Allreduce(min_eta_vals, global_eta_vals, op=MPI.MIN)

        eta_lb_viol = (global_eta_vals + np.full_like(global_eta_vals, 1e-3) \
                        < self._eta_lb_array).any()
        if eta_lb_viol:
            self.make_eta_lb_cut()
            return

        # set the master etas to be the minimum from every scenario
        master_etas = opt.master.eta
        for idx, scen_name in enumerate(opt.all_scenario_names):
            master_etas[scen_name].set_value(global_eta_vals[idx])

        # sum the local nonants for average computation
        master_nonants = self.master_nonants

        local_nonant_sum = np.fromiter( ( sum(nonants[k, nname, ix] for k in opt.local_scenarios)
                                          for nname, ix in master_nonants),
                                          dtype='d', count=len(master_nonants) )


        # Allreduce the xhats to get averages
        global_nonant_sum = np.empty(len(local_nonant_sum), dtype='d')
        self.intracomm.Allreduce(local_nonant_sum, global_nonant_sum, op = MPI.SUM)
        # need to divide through by the number of different spoke processes
        global_xbar = global_nonant_sum / self.nscen

        local_dist = np.array([0],dtype='d')
        local_winner = None
        # iterate through the ranks xhats to get the ranks maximum dist
        for i, k in enumerate(opt.local_scenarios):
            scenario_xhat = np.fromiter( (nonants[k, nname, ix] for nname, ix in master_nonants),
                                         dtype='d', count=len(master_nonants) )
            scenario_dist = np.linalg.norm(scenario_xhat - global_xbar)
            local_dist[0] = max(local_dist[0], scenario_dist)
            if local_winner is None:
                local_winner = k
            elif scenario_dist >= local_dist[0]:
                local_winner = k

        # Allreduce to find the biggest distance
        global_dist = np.empty(1, dtype='d')
        self.intracomm.Allreduce(local_dist, global_dist, op=MPI.MAX)
        vote = np.array([-1], dtype='i')
        if local_dist[0] >= global_dist[0]:
            vote[0] = self.intracomm.Get_rank()

        global_rank = np.empty(1, dtype='i')
        self.intracomm.Allreduce(vote, global_rank, op=MPI.MAX)

        # if we are the winner, grab the xhat and bcast it to the other ranks
        if self.intracomm.Get_rank() == global_rank[0]:
            farthest_xhat = np.fromiter( (nonants[local_winner, nname, ix] 
                                            for nname, ix in master_nonants),
                                         dtype='d', count=len(master_nonants) )
        else:
            farthest_xhat = np.zeros(len(master_nonants), dtype='d')

        self.intracomm.Bcast(farthest_xhat, root=global_rank)

        # set the first stage in the lshape object to correspond to farthest_xhat
        for ci, k in enumerate(master_nonants):
            master_nonants[k].set_value(farthest_xhat[ci])

        # generate cuts
        cuts = opt.master.bender.generate_cut()

        # eta var_id map:
        eta_id_map = { id(var) : k for k,var in master_etas.items()}
        coef_dict = dict()
        feas_cuts = list()
        # package cuts, slightly silly in that we reconstruct the coefficients from the cuts
        # TODO: modify the lshaped_cuts method to have a separate generate_coeffs function
        for cut in cuts:
            repn = generate_standard_repn(cut.body)
            if len(repn.nonlinear_vars) > 0:
                raise RuntimeError("BendersCutGenerator returned nonlinear cut")

            ## create a map from id(var) to index in repn
            id_var_to_idx = { id(var) : i for i,var in enumerate(repn.linear_vars) }

            ## find the eta index
            for vid in eta_id_map:
                if vid in id_var_to_idx:
                    scen_name = eta_id_map[vid]
                    ## cut_array -- [ constant, eta_coef, *nonant_coefs ]
                    cut_array = [repn.constant, repn.linear_coefs[id_var_to_idx[vid]]]
                    # each eta_s should only appear at most once per set of cuts
                    del eta_id_map[vid]
                    # each variable should only appear once in repn.linear_vars
                    del id_var_to_idx[vid]
                    break
            else: # no break,
                # so no scenario recourse cost variables appear in the cut
                cut_array = [repn.constant, 0.]
                # we don't know what scenario,
                # but since eta_s is 0, it doesn't
                # matter
                scen_name = None

            ## be intentional about how these are loaded
            ## unloaded the same way
            ## master_vars is in the order PH expects
            ## (per above)
            for var in master_nonants.values():
                # each variable should only appear at most once in repn.linear_vars
                idx = id_var_to_idx.pop(id(var), None)
                if idx is not None:
                    cut_array.append(repn.linear_coefs[idx])
                else:
                    cut_array.append(0)

            if scen_name is not None:
                coef_dict[scen_name] = np.array(cut_array, dtype='d')
            else:
                feas_cuts.append( np.array(cut_array, dtype='d') )

        ## we'll be storing a matrix as an array
        ## row_len is the length of each row
        row_len = 1+1+len(master_nonants)
        all_coefs = np.zeros( self.nscen*row_len +1, dtype='d')
        for idx, k in enumerate(opt.all_scenario_names):
            if k in coef_dict:
                all_coefs[row_len*idx:row_len*(idx+1)] = coef_dict[k]
            elif feas_cuts:
                all_coefs[row_len*idx:row_len*(idx+1)] = feas_cuts.pop()
        self.spoke_to_hub(all_coefs)

    def main(self):
        # call main cut generation routine

        # prep cut generation
        self.prep_cs_cuts()

        # main loop
        while not (self.got_kill_signal()):
            if self._new_locals:
                self.make_cut()
