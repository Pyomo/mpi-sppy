# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap, ComponentSet
from mpisppy.extensions.extension import Extension
from mpisppy.utils.sputils import is_persistent

import egret.common.lazy_ptdf_utils as lpu
from egret.models.unit_commitment import (_lazy_ptdf_check_violations,
                                          _lazy_ptdf_log_terminate_on_violations,
                                          _lazy_ptdf_warmstart_copy_violations,
                                          _lazy_ptdf_solve,
                                          _lazy_ptdf_normal_terminatation,
                                          _lazy_ptdf_violation_adder,
                                         )
import logging
from egret.common.log import logger

logger.setLevel(logging.ERROR)

_egret_ptdf_options = ('rel_ptdf_tol', 'abs_ptdf_tol', 'abs_flow_tol', 'rel_flow_tol',
                       'lazy_rel_flow_tol', 'max_violations_per_iteration', 'lazy',
                       'branch_kv_threshold', 'kv_threshold_type', 'active_flow_tol',)

class PTDFExtension(Extension):
    ''' Abstract base class for extensions to general SPBase objects.

        Args:
            ph (PHBase): The PHBase object for the current model
    '''
    def __init__(self, spopt_object, **kwargs):
        # attach self.opt
        super().__init__(spopt_object)

        self.pre_lp_iteration_limit = kwargs.get('pre_lp_iteration_limit', 100)
        self.lp_iteration_limit = kwargs.get('lp_iteration_limit', 100)
        self.lp_cleanup_phase = kwargs.get('lp_cleanup_phase', True)
        self.iteration_limit = kwargs.get('iteration_limit', 100000)
        self.verbose = kwargs.get('verbose',False)

        # Egret PTDF options
        self.egret_ptdf_options = {}
        for option in _egret_ptdf_options:
            if option in kwargs:
                self.egret_ptdf_options[option] = kwargs[option]

        if self.verbose:
            logger.setLevel(logging.INFO)

        self.initial_pass_complete = ComponentSet()
        self.bundling = self.opt.bundling

        self.vars_to_load = ComponentMap()
        self.time_periods = ComponentMap()
        self.bundle_conditional_probability = ComponentMap()

    def pre_solve(self, subproblem):
        if subproblem not in self.initial_pass_complete:
            self.spoke_name = None
            if self.opt.spcomm is not None:
                self.spoke_name = self.opt.spcomm.__class__.__name__
            self._initial_pass(subproblem)
            self.initial_pass_complete.add(subproblem)

    def post_solve(self, subproblem, results):
        scenario_blocks = self._get_scenario_blocks(subproblem)
        termination_cond, results, iterations = \
            self._mip_pass(subproblem, scenario_blocks, results)
        return results

    def _get_scenario_blocks(self, subproblem):
        if self.bundling:
            return tuple( subproblem.component(sname) \
                          for sname in subproblem._ef_scenario_names )
        else:
            return ( subproblem, )

    def _initial_pass(self, subproblem):
        # get vars_to_load for later
        scenario_blocks = self._get_scenario_blocks(subproblem)
        if is_persistent(subproblem._solver_plugin):
            subproblem_vars_to_load = []
            for s in scenario_blocks:
                for t in s.TimePeriods:
                    b = s.TransmissionBlock[t]
                    assert isinstance(b.p_nw, pyo.Var)
                    subproblem_vars_to_load.extend(b.p_nw.values())

            self.vars_to_load[subproblem] = subproblem_vars_to_load
        else:
            self.vars_to_load[subproblem] = None

        for s in scenario_blocks:
            self.time_periods[s] = s.TimePeriods
            if self.bundling:
                self.bundle_conditional_probability[s] = \
                        s._mpisppy_data.bundle_conditional_probability
            else:
                self.bundle_conditional_probability[s] = 1.

            # load in user-specified PTDF options
            for k,v in self.egret_ptdf_options:
                s._ptdf_options[k] = v

        self.tee = ("tee-rank0-solves" in self.opt.options
                    and self.opt.options['tee-rank0-solves']
                    and self.opt.cylinder_rank == 0
                    )

        if (self.pre_lp_iteration_limit + self.lp_iteration_limit) == 0:
            return

        # relax the initial subproblems
        for s in scenario_blocks:
            lpu.uc_instance_binary_relaxer(s, subproblem._solver_plugin)

        # solve the model
        for k,val in self.opt.current_solver_options.items():
            subproblem._solver_plugin.options[k] = val

        if is_persistent(subproblem._solver_plugin):
            results = subproblem._solver_plugin.solve(subproblem, tee=self.tee, save_results=False, load_solutions=False)
            subproblem._solver_plugin.load_vars(self.vars_to_load[subproblem])
        else:
            results = subproblem._solver_plugin.solve(subproblem, tee=self.tee, load_solutions=False)
            subproblem.solutions.load_from(results)

        if self.pre_lp_iteration_limit > 0:
            lp_warmstart_termination_cond, results, lp_warmstart_iterations = \
                    self._pre_lp_pass(subproblem, scenario_blocks)

        if self.lp_iteration_limit > 0:
            lp_termination_cond, results, lp_iterations = \
                    self._lp_pass(subproblem, scenario_blocks)

        if self.lp_cleanup_phase:
            tot_removed = 0
            for s in scenario_blocks:
                for t,b in s.TransmissionBlock.items():
                    tot_removed += lpu.remove_inactive(b, subproblem._solver_plugin, 
                                                        t, prepend_str=f"[LP cleanup phase on rank {self.opt.global_rank}] ")
            logger.info(f"[LP cleanup phase on rank {self.opt.global_rank} for {self.spoke_name}] removed {tot_removed} inactive flow constraint(s)")
        # enforce binaries in subproblems
        for s in scenario_blocks:
            lpu.uc_instance_binary_enforcer(s, subproblem._solver_plugin)

        # mpi-sppy will solve the MIP
        return

    def _do_pass(self, subproblem, scenario_blocks, time_periods, vars_to_load, 
            prepend_str, iteration_limit, add_all_lazy_violations=False, 
            results=None, pre_lp_cleanup=False):

        persistent_solver = is_persistent(subproblem._solver_plugin)
        for i in range(iteration_limit):
            flows, viol_lazy = ComponentMap(), ComponentMap()
            terminate_this_iter, all_viol_in_model = ComponentMap(), ComponentMap()
            for s in scenario_blocks:
                flows[s], viol_num, mon_viol_num, viol_lazy[s] = \
                        _lazy_ptdf_check_violations(s, s.model_data, time_periods[s],
                                s._ptdf_options, prepend_str)

                terminate_this_iter[s], all_viol_in_model[s] = \
                        _lazy_ptdf_log_terminate_on_violations(viol_num, mon_viol_num,
                                                                i, prepend_str)

            all_viol_in_model = all(all_viol_in_model.values())
            terminate_this_iter = all(terminate_this_iter.values())
            if terminate_this_iter and not add_all_lazy_violations:
                if pre_lp_cleanup:
                    results = self._pre_lp_cleanup(subproblem, scenario_blocks,
                                                    persistent_solver, time_periods, prepend_str)
                return _lazy_ptdf_normal_terminatation(all_viol_in_model, results, i, prepend_str)

            for s in scenario_blocks:
                _lazy_ptdf_violation_adder(s, s.model_data, flows[s], viol_lazy[s], time_periods[s],
                                subproblem._solver_plugin, s._ptdf_options, prepend_str, i,
                                obj_multi=self.bundle_conditional_probability[s])

            if terminate_this_iter and add_all_lazy_violations:
                if pre_lp_cleanup:
                    results = self._pre_lp_cleanup(subproblem, scenario_blocks,
                                                    persistent_solver, time_periods, prepend_str)
                return _lazy_ptdf_normal_terminatation(all_viol_in_model, results, i, prepend_str)

            results = _lazy_ptdf_solve(subproblem, subproblem._solver_plugin, persistent_solver,
                                       symbolic_solver_labels=False, solver_tee=self.tee,
                                       vars_to_load=vars_to_load, solve_method_options=None)
        else:
            if pre_lp_cleanup:
                results = self._pre_lp_cleanup(subproblem, scenario_blocks,
                                        persistent_solver, time_periods, prepend_str)
            return lpu.LazyPTDFTerminationCondition.ITERATION_LIMIT, results, i

    def _pre_lp_cleanup(self, subproblem, scenario_blocks, persistent_solver, time_periods, prepend_str):
        if persistent_solver:
            # unpack lpu._load_pf_slacks into a single call to load_vars
            vars_to_load = []
            for s in scenario_blocks:
                for t in time_periods[s]:
                    b = s.TransmissionBlock[t]
                    vars_to_load.extend(b.pf_slack_pos.values())
                    vars_to_load.extend(b.pf_slack_neg.values())
                    vars_to_load.extend(b.pfi_slack_pos.values())
                    vars_to_load.extend(b.pfi_slack_neg.values())
                    vars_to_load.extend(b.pfc_slack_pos.values())
                    vars_to_load.extend(b.pfc_slack_neg.values())
            if vars_to_load:
                subproblem._solver_plugin.load_vars(vars_to_load)

        for s in scenario_blocks:
            _lazy_ptdf_warmstart_copy_violations(s, s.model_data, time_periods[s],
                    subproblem._solver_plugin, s._ptdf_options, prepend_str,
                    obj_multi=self.bundle_conditional_probability[s])

        # the basis is usually not a good start now
        # so reset the solver if we can
        subproblem._solver_plugin.reset()

        results = _lazy_ptdf_solve(subproblem, subproblem._solver_plugin, persistent_solver,
                                   symbolic_solver_labels=False, solver_tee=self.tee,
                                   vars_to_load=self.vars_to_load[subproblem],
                                   solve_method_options=None)
        return results

    def _pre_lp_pass(self, subproblem, scenario_blocks):
        vars_to_load_t_subset = ComponentMap()
        t_subset = ComponentMap()
        persistent_solver = is_persistent(subproblem._solver_plugin)

        if persistent_solver:
            vars_to_load = []
        else:
            vars_to_load = None
        for s in scenario_blocks:
            max_demand_time = max(s.TotalDemand, key=s.TotalDemand.__getitem__)
            t_subset[s] = [max_demand_time,]
            if persistent_solver:
                transmission_block = s.TransmissionBlock[max_demand_time]
                assert isinstance(transmission_block.p_nw, pyo.Var)
                vars_to_load.extend(transmission_block.p_nw.values())

        return self._do_pass(subproblem, scenario_blocks, t_subset, vars_to_load,
                             prepend_str=f"[LP warmstart phase on rank {self.opt.global_rank} for {self.spoke_name}] ",
                             iteration_limit=self.pre_lp_iteration_limit,
                             add_all_lazy_violations=False,
                             results=None, pre_lp_cleanup=True) 

    def _lp_pass(self, subproblem, scenario_blocks):
        return self._do_pass(subproblem, scenario_blocks,
                             self.time_periods, self.vars_to_load[subproblem],
                             prepend_str=f"[LP phase on rank {self.opt.global_rank} for {self.spoke_name}] ",
                             iteration_limit=self.lp_iteration_limit,
                             add_all_lazy_violations=True,
                             results=None, pre_lp_cleanup=False) 

    def _mip_pass(self, subproblem, scenario_blocks, results):
        return self._do_pass(subproblem, scenario_blocks,
                             self.time_periods, self.vars_to_load[subproblem],
                             prepend_str=f"[MIP phase on rank {self.opt.global_rank} for {self.spoke_name}] ",
                             iteration_limit=self.iteration_limit,
                             add_all_lazy_violations=False,
                             results=results, pre_lp_cleanup=False) 
