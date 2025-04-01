###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
''' Implementation of the Frank-Wolfe Progressive Hedging (FW-PH) algorithm
    described in the paper:

    N. Boland et al., "Combining Progressive Hedging with a Frank-Wolfe method
    to compute Lagrangian dual bounds in stochastic mixed-integer programming".
    SIAM J. Optim. 28(2):1312--1336, 2018.

    Current implementation supports parallelism and bundling.
'''

import mpisppy.phbase
import mpisppy.utils.sputils as sputils
import numpy as np
import pyomo.environ as pyo
import time
import re # For manipulating scenario names
import random

from mpisppy import MPI
from mpisppy import global_toc
from pyomo.repn.standard_repn import generate_standard_repn
from mpisppy.utils.sputils import find_active_objective
from pyomo.core.expr.visitor import replace_expressions
from pyomo.core.expr.numeric_expr import LinearExpression

from mpisppy.cylinders.xhatshufflelooper_bounder import ScenarioCycler
from mpisppy.extensions.xhatbase import XhatBase

class FWPH(mpisppy.phbase.PHBase):
    
    def __init__(
        self,
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement=None,
        all_nodenames=None,
        mpicomm=None,
        scenario_creator_kwargs=None,
        extensions=None,
        extension_kwargs=None,
        ph_converger=None,
        rho_setter=None,
        variable_probability=None,
    ):
        super().__init__(
            options,
            all_scenario_names,
            scenario_creator,
            scenario_denouement,
            all_nodenames,
            mpicomm,
            scenario_creator_kwargs,
            extensions,
            extension_kwargs,
            ph_converger,
            rho_setter,
            variable_probability,
        )      
        assert (variable_probability is None), "variable probability is not allowed with fwph"
        self._init(options)

    def _init(self, FW_options):
        self.FW_options = FW_options
        self._options_checks_fw()
        self.vb = True
        if ('FW_verbose' in self.FW_options):
            self.vb = self.FW_options['FW_verbose']

    def fw_prep(self):
        self.PH_Prep(attach_duals=True, attach_prox=False)
        self._output_header()
        self._attach_MIP_vars()
        self._cache_nonant_var_swap_mip()

        trivial_bound = self.Iter0()
        secs = time.time() - self.t0
        self._output(trivial_bound, trivial_bound, np.nan, secs)
        best_bound = trivial_bound

        # Lines 2 and 3 of Algorithm 3 in Boland
        # Now done a the beginning of the first iteration
        # self.Compute_Xbar(self.options['verbose'])
        # self.Update_W(self.options['verbose'])

        # Necessary pre-processing steps
        # We disable_W so they don't appear
        # in the MIP objective when _set_QP_objective
        # snarfs it for the QP
        self._disable_W()
        self._initialize_QP_subproblems()
        self._attach_indices()
        self._attach_MIP_QP_maps()
        self._set_QP_objective()
        self._initialize_QP_var_values()
        self._cache_nonant_var_swap_qp()
        self._setup_shared_column_generation()

        number_initial_column_tries = self.options.get("FW_initialization_attempts", 10)
        if self.FW_options["FW_iter_limit"] == 1 and number_initial_column_tries < 1:
            global_toc(f"{self.__class__.__name__}: Warning: FWPH needs an initial shared column if FW_iter_limit == 1. Increasing FW_iter_limit to 2 to ensure convergence", self.cylinder_rank == 0)
            self.FW_options["FW_iter_limit"] = 2
        if self.FW_options["FW_iter_limit"] == 1 or number_initial_column_tries > 0:
            number_points = self._generate_shared_column(number_initial_column_tries)
            if number_points == 0 and self.FW_options["FW_iter_limit"] == 1:
                global_toc(f"{self.__class__.__name__}: Warning: FWPH failed to find an initial feasible solution. Increasing FW_iter_limit to 2 to ensure convergence", self.cylinder_rank == 0)
                self.FW_options["FW_iter_limit"] = 2

        self._reenable_W()

        if (self.ph_converger):
            self.convobject = self.ph_converger(self, self.cylinder_rank, self.n_proc)

        return best_bound

    def fwph_main(self, finalize=True):
        self.t0 = time.time()
        best_bound = self.fw_prep()

        # FWPH takes some time to initialize
        # If run as a spoke, check for convergence here
        if self.spcomm and self.spcomm.is_converged():
            return None, None, None

        # The body of the algorithm
        for self._PHIter in range(1, self.options['PHIterLimit']+1):

            # tbphloop = time.time()
            # TODO: should implement our own Xbar / W computation
            #       which just considers the QP subproblems
            self._swap_nonant_vars()
            self.Compute_Xbar(self.options['verbose'])
            self.Update_W(self.options['verbose'])
            self._swap_nonant_vars_back()

            if (self.extensions): 
                self.extobject.miditer()

            if hasattr(self.spcomm, "sync_Ws"):
                self.spcomm.sync_Ws()
            if (self._is_timed_out()):
                global_toc("FWPH Timed Out", self.cylinder_rank == 0)
                break

            if (self.ph_converger):
                diff = self.convobject.convergence_value()
                if (self.convobject.is_converged()):
                    secs = time.time() - self.t0
                    self._output(self._local_bound, best_bound, diff, secs)
                    global_toc('FWPH converged to user-specified criteria', self.cylinder_rank == 0)
                    break
            else: # Convergence check from Boland
                diff = self._conv_diff()
                if (diff < self.options['convthresh']):
                    secs = time.time() - self.t0
                    self._output(self._local_bound, best_bound, diff, secs)
                    global_toc(f'FWPH converged based on standard criteria, convergence diff: {diff}',
                               self.cylinder_rank == 0,
                    )
                    break

            self._swap_nonant_vars()
            self._local_bound = 0
            # tbsdm = time.time()
            for name in self.local_subproblems:
                dual_bound = self.SDM(name)
                if dual_bound is None:
                    dual_bound = np.nan
                self._local_bound += self.local_subproblems[name]._mpisppy_probability * \
                                     dual_bound
            # tsdm = time.time() - tbsdm
            # print(f"PH iter {self._PHIter}, total SDM time: {tsdm}")
            self._compute_dual_bound()
            if (self.is_minimizing):
                best_bound = np.fmax(best_bound, self._local_bound)
            else:
                best_bound = np.fmin(best_bound, self._local_bound)
            if self._can_update_best_bound():
                self.best_bound_obj_val = best_bound
            self._swap_nonant_vars_back()

            if (self.extensions): 
                self.extobject.enditer()

            secs = time.time() - self.t0
            self._output(self._local_bound, best_bound, diff, secs)


            # add a shared column
            shared_columns = self.options.get("FWPH_shared_columns_per_iteration", 0)
            if shared_columns > 0:
                self.mpicomm.Barrier()
                self._disable_W()
                for s in self.local_subproblems.values():
                    if sputils.is_persistent(s._solver_plugin):
                        active_objective_datas = list(s.component_data_objects(
                             pyo.Objective, active=True, descend_into=True))
                        s._solver_plugin.set_objective(active_objective_datas[0])
                self._generate_shared_column(shared_columns)
                self._reenable_W()

            ## Hubs/spokes take precedence over convergers
            if hasattr(self.spcomm, "sync_nonants"):
                self.spcomm.sync_nonants()
                self.spcomm.sync_bounds()
                self.spcomm.sync_extensions()
            elif hasattr(self.spcomm, "sync"):
                self.spcomm.sync()
            if self.spcomm and self.spcomm.is_converged():
                secs = time.time() - self.t0
                self._output(self._local_bound, best_bound, np.nan, secs)
                global_toc('FWPH stopped due to cylinder convergence')
                break

            if (self.extensions): 
                self.extobject.enditer_after_sync()
            # tphloop = time.time() - tbphloop
            # print(f"PH iter {self._PHIter}, total time: {tphloop}")

        if finalize:
            weight_dict = self._gather_weight_dict() # None if rank != 0
            xbars_dict  = self._get_xbars() # None if rank != 0
            return self._PHIter, weight_dict, xbars_dict
        return self._PHIter

    def SDM(self, model_name):
        '''  Algorithm 2 in Boland et al. (with small tweaks)
        '''
        mip = self.local_subproblems[model_name]
        qp  = self.local_QP_subproblems[model_name]
    
        verbose = self.options["verbose"]
        dtiming = True # self.options["display_timing"]
        teeme = ("tee-rank0-solves" in self.options
                 and self.options['tee-rank0-solves']
                 and self.cylinder_rank == 0
                 )
        # Set the QP dual weights to the correct values If we are bundling, we
        # initialize the QP dual weights to be the dual weights associated with
        # the first scenario in the bundle (this should be okay, because each
        # scenario in the bundle has the same dual weights, analytically--maybe
        # a numerical problem).
        arb_scen_mip = self.local_scenarios[mip.scen_list[0]] \
                       if self.bundling else mip
        for (node_name, ix) in arb_scen_mip._mpisppy_data.nonant_indices:
            qp._mpisppy_model.W[node_name, ix]._value = \
                arb_scen_mip._mpisppy_model.W[node_name, ix].value

        alpha = self.FW_options['FW_weight']
        # Algorithm 3 line 6
        xt = {ndn_i:
            (1 - alpha) * arb_scen_mip._mpisppy_model.xbars[ndn_i]._value
            + alpha * xvar._value
            for ndn_i, xvar in arb_scen_mip._mpisppy_data.nonant_indices.items()
            }

        mip_source = mip.scen_list if self.bundling else [model_name]

        for itr in range(self.FW_options['FW_iter_limit']):
            # loop_start = time.time()
            # Algorithm 2 line 4
            for scenario_name in mip_source:
                scen_mip = self.local_scenarios[scenario_name]
                for ndn_i in scen_mip._mpisppy_data.nonant_indices:
                    scen_mip._mpisppy_model.W[ndn_i]._value = (
                        qp._mpisppy_model.W[ndn_i]._value
                        + scen_mip._mpisppy_model.rho[ndn_i]._value
                        * (xt[ndn_i]
                        -  scen_mip._mpisppy_model.xbars[ndn_i]._value))

            cutoff = pyo.value(qp._mpisppy_model.mip_obj_in_qp) + pyo.value(qp.recourse_cost)
            # tbmipsolve = time.time()
            active_objective = sputils.find_active_objective(mip)
            if self.is_minimizing:
                # obj <= cutoff
                obj_cutoff_constraint = (None, active_objective, cutoff)
            else:
                # obj >= cutoff
                obj_cutoff_constraint = (cutoff, active_objective, None)
            mip._mpisppy_model.obj_cutoff_constraint = pyo.Constraint(expr=obj_cutoff_constraint)
            if sputils.is_persistent(mip._solver_plugin):
                mip._solver_plugin.add_constraint(mip._mpisppy_model.obj_cutoff_constraint)
            # Algorithm 2 line 5
            self.solve_one(
                self.options["iterk_solver_options"],
                model_name,
                mip,
                dtiming=dtiming,
                tee=teeme,
                verbose=verbose,
            )
            if sputils.is_persistent(mip._solver_plugin):
                mip._solver_plugin.remove_constraint(mip._mpisppy_model.obj_cutoff_constraint)
            mip.del_component(mip._mpisppy_model.obj_cutoff_constraint)
            # tmipsolve = time.time() - tbmipsolve
            if mip._mpisppy_data.scenario_feasible:

                # Algorithm 2 lines 6--8
                if (itr == 0):
                    dual_bound = mip._mpisppy_data.outer_bound

                # Algorithm 2 line 9 (compute \Gamma^t)
                inner_bound = mip._mpisppy_data.inner_bound
                if abs(inner_bound) > 1e-9:
                    stop_check = (cutoff - inner_bound) / abs(inner_bound) # \Gamma^t in Boland, but normalized
                else:
                    stop_check = cutoff - inner_bound # \Gamma^t in Boland
                # print(f"{model_name}, Gamma^t = {stop_check}")
                stop_check_tol = self.FW_options["stop_check_tol"]\
                                 if "stop_check_tol" in self.FW_options else 1e-4
                if (self.is_minimizing and stop_check < -stop_check_tol):
                    print('Warning (fwph): convergence quantity Gamma^t = '
                         '{sc:.2e} (should be non-negative)'.format(sc=stop_check))
                    print('Try decreasing the MIP gap tolerance and re-solving')
                elif (not self.is_minimizing and stop_check > stop_check_tol):
                    print('Warning (fwph): convergence quantity Gamma^t = '
                         '{sc:.2e} (should be non-positive)'.format(sc=stop_check))
                    print('Try decreasing the MIP gap tolerance and re-solving')

                # tbcol = time.time()
                self._add_QP_column(model_name)
                # tcol = time.time() - tbcol
                # print(f"{model_name} QP add_column time: {tcol}")

            else:
                dual_bound = None
                global_toc(f"{self.__class__.__name__}: Could not find an improving column for {model_name}!", True)
                # couldn't find an improving direction, the column would not become active

            # tbqpsol = time.time()
            # QPs are weird if bundled
            _bundling = self.bundling
            self.solve_one(
                self.options["qp_solver_options"],
                model_name,
                qp,
                dtiming=dtiming,
                tee=teeme,
                verbose=verbose,
            )
            self.bundling = _bundling
            # tqpsol = time.time() - tbqpsol

            # print(f"{model_name}, solve + add_col time: {tmipsolve + tcol + tqpsol}")
            # fwloop = time.time() - loop_start
            # print(f"{model_name}, total loop time: {fwloop}")

            if dual_bound is None or (stop_check < self.FW_options['FW_conv_thresh']):
                break

            # reset for next loop
            xt = {ndn_i : xvar._value for ndn_i, xvar in arb_scen_mip._mpisppy_data.nonant_indices.items()}

        # Re-set the mip._mpisppy_model.W so that the QP objective 
        # is correct in the next major iteration
        for scenario_name in mip_source:
            scen_mip = self.local_scenarios[scenario_name]
            for (node_name, ix) in scen_mip._mpisppy_data.nonant_indices:
                scen_mip._mpisppy_model.W[node_name, ix]._value = \
                    qp._mpisppy_model.W[node_name, ix]._value

        return dual_bound

    def _add_QP_column(self, model_name):
        ''' Add a column to the QP, with values taken from the most recent MIP
            solve. Assumes the inner_bound is up-to-date in the MIP model.
        '''
        mip = self.local_subproblems[model_name]
        qp  = self.local_QP_subproblems[model_name]
        solver = qp._solver_plugin
        persistent = sputils.is_persistent(solver)

        total_recourse_cost = mip._mpisppy_data.inner_bound - pyo.value(mip._mpisppy_model.nonant_obj_part)

        if hasattr(solver, 'add_column'):
            new_var = qp.a.add()
            coef_list = [1.]
            constr_list = [qp.sum_one]
            target = mip.ref_vars if self.bundling else mip._mpisppy_data.nonant_vars
            for (node, ix) in qp.eqx.index_set():
                coef_list.append(target[node, ix].value)
                constr_list.append(qp.eqx[node, ix])
            coef_list.append(total_recourse_cost)
            constr_list.append(qp.eq_recourse_cost)
            solver.add_column(qp, new_var, 0, constr_list, coef_list)
            return

        # Add new variable and update \sum a_i = 1 constraint
        new_var = qp.a.add() # Add the new convex comb. variable
        lb, body, ub = qp.sum_one.to_bounded_expression()
        body += new_var
        qp.sum_one.set_value((lb, body, ub))
        if (persistent):
            solver.add_var(new_var)
            solver.remove_constraint(qp.sum_one)
            solver.add_constraint(qp.sum_one)

        target = mip.ref_vars if self.bundling else mip._mpisppy_data.nonant_vars
        for (node, ix) in qp.eqx.index_set():
            lb, body, ub = qp.eqx[node, ix].to_bounded_expression()
            body += new_var * target[node, ix].value
            qp.eqx[node, ix].set_value((lb, body, ub))
            if (persistent):
                solver.remove_constraint(qp.eqx[node, ix])
                solver.add_constraint(qp.eqx[node, ix])
        lb, body, ub = qp.eq_recourse_cost.to_bounded_expression()
        body += new_var * total_recourse_cost
        qp.eq_recourse_cost.set_value((lb, body, ub))
        if (persistent):
            solver.remove_constraint(qp.eq_recourse_cost)
            solver.add_constraint(qp.eq_recourse_cost)

    def _attach_indices(self):
        ''' Attach the fields x_indices to the model objects in
            self.local_subproblems (not self.local_scenarios, nor
            self.local_QP_subproblems).

            x_indices is a list of tuples of the form...
                (scenario name, node name, variable index) <-- bundling
                (node name, variable index)                <-- no bundling
            
            Must be called after the subproblems (MIPs AND QPs) are created.
        '''
        for mip in self.local_subproblems.values():
            if (self.bundling):
                x_indices = [(scenario_name, node_name, ix)
                    for scenario_name in mip.scen_list
                    for (node_name, ix) in 
                        self.local_scenarios[scenario_name]._mpisppy_data.nonant_indices]
            else:
                x_indices = mip._mpisppy_data.nonant_indices.keys()

            x_indices = pyo.Set(initialize=x_indices)
            x_indices.construct()
            mip._mpisppy_model.x_indices = x_indices

    def _attach_MIP_QP_maps(self):
        ''' Create dictionaries that map MIP variable ids to their QP
            counterparts, and vice versa.
        '''
        for name in self.local_subproblems.keys():
            mip = self.local_subproblems[name]
            qp  = self.local_QP_subproblems[name]

            mip._mpisppy_data.mip_to_qp = {id(mip._mpisppy_data.nonant_vars[key]): qp.x[key]
                                for key in mip._mpisppy_model.x_indices}
            qp._mpisppy_data.qp_to_mip = {id(qp.x[key]): mip._mpisppy_data.nonant_vars[key]
                                for key in mip._mpisppy_model.x_indices}

    def _attach_MIP_vars(self):
        ''' Create a list indexed (node_name, ix) for all the MIP
            non-anticipative and leaf variables, so that they can be easily
            accessed when adding columns to the QP.
        '''
        if (self.bundling):
            for (bundle_name, EF) in self.local_subproblems.items():
                EF._mpisppy_data.nonant_vars = dict()
                for scenario_name in EF.scen_list:
                    mip = self.local_scenarios[scenario_name]
                    # Non-anticipative variables
                    nonant_dict = {(scenario_name, ndn, ix): nonant
                        for (ndn,ix), nonant in mip._mpisppy_data.nonant_indices.items()}
                    EF._mpisppy_data.nonant_vars.update(nonant_dict)
                    # Reference variables are already attached: EF.ref_vars
                    # indexed by (node_name, index)
                self._attach_nonant_objective(mip)
        else:
            for (name, mip) in self.local_scenarios.items():
                mip._mpisppy_data.nonant_vars = mip._mpisppy_data.nonant_indices
                self._attach_nonant_objective(mip)

    def _compute_dual_bound(self):
        ''' Compute the FWPH dual bound using self._local_bound from each rank
        '''
        send = np.array(self._local_bound)
        recv = np.array(0.)
        self.comms['ROOT'].Allreduce(
            [send, MPI.DOUBLE], [recv, MPI.DOUBLE], op=MPI.SUM)
        self._local_bound = recv

    def _conv_diff(self):
        ''' Perform the convergence check of Algorithm 3 in Boland et al. '''
        diff = 0.
        for name in self.local_subproblems.keys():
            mip = self.local_subproblems[name]
            arb_mip = self.local_scenarios[mip.scen_list[0]] \
                        if self.bundling else mip
            qp  = self.local_QP_subproblems[name]
            qpx = qp.xr if self.bundling else qp.x
            xbars = arb_mip._mpisppy_model.xbars
            diff_s = mip._mpisppy_probability * sum((qpx[idx]._value - xbars[idx]._value)**2
                                                    for idx in arb_mip._mpisppy_data.nonant_indices)
            diff += diff_s
        diff = np.array(diff)
        recv = np.array(0.)
        self.comms['ROOT'].Allreduce(
            [diff, MPI.DOUBLE], [recv, MPI.DOUBLE], op=MPI.SUM)
        return recv

    def _attach_nonant_objective(self, mip, _print_warning=[True]):
        """ Extract the parts of the objective function which involve nonants.
            Adds to mip._mpisppy_model.nonant_obj_part

            Args:
                mip (Pyomo ConcreteModel): MIP model for a scenario or bundle.
        """
        obj = find_active_objective(mip)
        repn = generate_standard_repn(obj.expr, compute_values=False, quadratic=False)
        nonant_var_ids = mip._mpisppy_data.varid_to_nonant_index
        if repn.nonlinear_vars:
            for v in repn.nonlinear_vars:
                if id(v) in nonant_var_ids:
                    raise RuntimeError("FWPH does not support models where the nonants "
                                       "participate nonlinearly in the objective function")
            global_toc("Using FWPH with nonlinear recourse cost. "
                       "Simplicial decomposition iterates may not return vertices!",
                       (self.cylinder_rank==0 and _print_warning[0])
            )
            _print_warning[0] = False
        linear_coefs = []
        linear_vars = []
        for coef, var in zip(repn.linear_coefs, repn.linear_vars):
            if id(var) in nonant_var_ids:
                linear_coefs.append(coef)
                linear_vars.append(var)
        mip._mpisppy_model.nonant_obj_part = LinearExpression(linear_coefs=linear_coefs, linear_vars=linear_vars)

    def _extract_nonant_objective(self, mip):
        ''' Extract the original part of the provided MIP's objective function
            (no dual or prox terms), and create a copy containing the QP
            variables in place of the MIP variables.

            Args:
                mip (Pyomo ConcreteModel): MIP model for a scenario or bundle.

            Returns:
                obj (Pyomo Objective): objective function extracted
                    from the MIP
                new (Pyomo Expression): expression from the MIP model
                    objective with the MIP variables replaced by QP variables.
                    Does not inculde dual or prox terms.

            Notes:
                Acts on either a single-scenario model or a bundle
        '''
        obj = mip._mpisppy_model.nonant_obj_part
        mip_to_qp = mip._mpisppy_data.mip_to_qp
        linear_vars = [mip_to_qp[id(var)] for var in obj.linear_vars]
        linear_coefs = [pyo.value(coef) for coef in obj.linear_coefs]
        new = LinearExpression(
            linear_coefs=linear_coefs, linear_vars=linear_vars
        )
        return obj, new

    def _gather_weight_dict(self, strip_bundle_names=False):
        ''' Compute a double nested dictionary of the form

                weights[scenario name][variable name] = weight value

            for FWPH to return to the user.
        
            Notes:
                Must be called after the variables are swapped back.
        '''
        local_weights = dict()
        for (name, scenario) in self.local_scenarios.items():
            if (self.bundling and strip_bundle_names):
                scenario_weights = dict()
                for ndn_ix, var in scenario._mpisppy_data.nonant_indices.items():
                    rexp = r'^' + scenario.name + r'\.'
                    var_name = re.sub(rexp, '', var.name)
                    scenario_weights[var_name] = \
                                        scenario._mpisppy_model.W[ndn_ix].value
            else:
                scenario_weights = {nonant.name: scenario._mpisppy_model.W[ndn_ix].value
                        for ndn_ix, nonant in scenario._mpisppy_data.nonant_indices.items()}
            local_weights[name] = scenario_weights

        weights = self.comms['ROOT'].gather(local_weights, root=0)
        return weights

    def _get_xbars(self, strip_bundle_names=False):
        ''' Return the xbar vector if rank = 0 and None, otherwise
            (Consistent with _gather_weight_dict).

            Args:
                TODO
                
            Notes:
                Paralellism is not necessary since each rank already has an
                identical copy of xbar, provided by Compute_Xbar().

                Returned dictionary is indexed by variable name 
                (as provided by the user).

                Must be called after variables are swapped back (I think).
        '''
        if (self.cylinder_rank != 0):
            return None
        else:
            random_scenario_name = list(self.local_scenarios.keys())[0]
            scenario = self.local_scenarios[random_scenario_name]
            xbar_dict = {}
            for node in scenario._mpisppy_node_list:
                for (ix, var) in enumerate(node.nonant_vardata_list):
                    var_name = var.name
                    if (self.bundling and strip_bundle_names):
                        rexp = r'^' + random_scenario_name + r'\.'
                        var_name = re.sub(rexp, '', var_name)
                    xbar_dict[var_name] = scenario._mpisppy_model.xbars[node.name, ix].value
            return xbar_dict

    def _initialize_QP_subproblems(self):
        ''' Instantiates the (convex) QP subproblems (eqn. (13) in the Boland
            paper) for each scenario. Does not create/attach an objective.

            Attachs a local_QP_subproblems dict to self. Keys are scenario
            names (or bundle names), values are Pyomo ConcreteModel objects
            corresponding to the QP subproblems. 

            QP subproblems are in their original form, without the x and y
            variables eliminated. Rationale: pre-solve will get this, easier
            bookkeeping (objective does not need to be changed at each inner
            iteration this way).
        '''
        self.local_QP_subproblems = dict()
        for (name, model) in self.local_subproblems.items():
            if (self.bundling):
                xr_indices = model.ref_vars.keys()
                nonant_indices = model._mpisppy_data.nonant_vars.keys()
            else:
                nonant_indices = model._mpisppy_data.nonant_indices.keys()

            ''' Convex comb. coefficients '''
            QP = pyo.ConcreteModel()
            QP.a = pyo.VarList(domain=pyo.NonNegativeReals)
            QP.a.add() # Just one variable (1-based index!) to start

            ''' Other variables '''
            QP.x = pyo.Var(nonant_indices, within=pyo.Reals)
            mip_recourse_cost = model._mpisppy_data.inner_bound - pyo.value(model._mpisppy_model.nonant_obj_part)
            QP.recourse_cost = pyo.Var(within=pyo.Reals, initialize=mip_recourse_cost)

            if (self.bundling):
                QP.xr = pyo.Var(xr_indices, within=pyo.Reals)

            ''' Non-anticipativity constraint '''
            if (self.bundling):
                def nonant_rule(m, scenario_name, node_name, ix):
                    return m.x[scenario_name, node_name, ix] == \
                            m.xr[node_name, ix]
                QP.na = pyo.Constraint(nonant_indices, rule=nonant_rule)
            
            ''' (x,y) constraints '''
            if (self.bundling):
                def x_rule(m, node_name, ix):
                    return -m.xr[node_name, ix] + m.a[1] * \
                            model.ref_vars[node_name, ix].value == 0
                def rc_rule(m):
                    return -m.recourse_cost + m.a[1] * mip_recourse_cost == 0
                QP.eqx = pyo.Constraint(xr_indices, rule=x_rule)
            else:
                def x_rule(m, node_name, ix):
                    return -m.x[node_name, ix] + m.a[1] * \
                            model._mpisppy_data.nonant_vars[node_name, ix].value == 0
                def rc_rule(m):
                    return -m.recourse_cost + m.a[1] * mip_recourse_cost == 0
                QP.eqx = pyo.Constraint(nonant_indices, rule=x_rule)

            QP.eq_recourse_cost = pyo.Constraint(rule=rc_rule)
            QP.sum_one = pyo.Constraint(expr=pyo.quicksum(QP.a.values())==1)

            QP._mpisppy_data = pyo.Block(name="For non-Pyomo mpi-sppy data")
            QP._mpisppy_model = pyo.Block(name="For mpi-sppy Pyomo additions to the scenario model")
            QP._mpisppy_data.nonant_indices = pyo.Reference(QP.x)

            self.local_QP_subproblems[name] = QP
                
    def _initialize_QP_var_values(self):
        ''' Set the value of the QP variables to be equal to the values of the
            corresponding MIP variables.

            Notes:
                Must be called before _swap_nonant_vars()

                Must be called after Iter0().
        '''
        for name in self.local_subproblems.keys():
            mip = self.local_subproblems[name]
            qp  = self.local_QP_subproblems[name]

            for key in mip._mpisppy_model.x_indices:
                qp.x[key]._value = mip._mpisppy_data.nonant_vars[key].value
            qp.recourse_cost._value = mip._mpisppy_data.inner_bound - pyo.value(mip._mpisppy_model.nonant_obj_part)

            # Set the non-anticipative reference variables if we're bundling
            if (self.bundling):
                arb_scenario = mip.scen_list[0]
                naix = self.local_scenarios[arb_scenario]._mpisppy_data.nonant_indices
                for (node_name, ix) in naix:
                    # Check that non-anticipativity is satisfied
                    # within the bundle (for debugging)
                    vals = [mip._mpisppy_data.nonant_vars[scenario_name, node_name, ix].value
                            for scenario_name in mip.scen_list]
                    assert(max(vals) - min(vals) < 1e-7)
                    qp.xr[node_name, ix].set_value(
                        mip._mpisppy_data.nonant_vars[arb_scenario, node_name, ix].value)

    def _setup_shared_column_generation(self):
        """ helper for shared column generation """
        #We need to keep track of the way scenario_names were sorted
        scen_names = list(enumerate(self.all_scenario_names))

        self._random_seed = 42
        # Have a separate stream for shuffling
        random_stream = random.Random()
        random_stream.seed(self._random_seed)

        # shuffle the scenarios associated (i.e., sample without replacement)
        shuffled_scenarios = random_stream.sample(scen_names, len(scen_names))

        self._scenario_cycler = ScenarioCycler(shuffled_scenarios,
                                         self.nonleaves,
                                         False,
                                         None)

        self._xhatter = XhatBase(self)
        self._xhatter.post_iter0()

    def _generate_shared_column(self, tries=1):
        """ Called after iter 0 to satisfy the condition of equation (17)
            in Boland et al., if t_max / FW_iter_limit == 1
        """

        stage2EFsolvern = self.options.get("stage2EFsolvern", None)
        branching_factors = self.options.get("branching_factors", None)  # for stage2ef

        number_points = 0
        for t in range(min(tries, len(self.all_scenario_names))):
            # will save in best solution
            snamedict = self._scenario_cycler.get_next()
            if snamedict is None:
                self._scenario_cycler.begin_epoch()
                snamedict = self._scenario_cycler.get_next()
            obj = self._xhatter._try_one(snamedict,
                                   solver_options = self.options["iterk_solver_options"],
                                   verbose=False,
                                   restore_nonants=False,
                                   stage2EFsolvern=stage2EFsolvern,
                                   branching_factors=branching_factors)
            if obj is not None:
                for model_name in self.local_subproblems:
                    self._add_QP_column(model_name)
                # self._restore_nonants()
                number_points += 1
            self._restore_nonants()
        return number_points

    def _is_timed_out(self):
        if (self.cylinder_rank == 0):
            time_elapsed = time.time() - self.t0
            status = 1 if (time_elapsed > self.FW_options['time_limit']) \
                       else 0
        else:
            status = None
        status = self.comms['ROOT'].bcast(status, root=0)
        return status != 0

    def _options_checks_fw(self):
        ''' Name                Boland notation (Algorithm 2)
            -------------------------------------------------
            FW_iter_limit       t_max
            FW_weight           alpha
            FW_conv_thresh      tau
        '''
        # 1. Check for required options
        reqd_options = ['FW_iter_limit', 'FW_weight', 'FW_conv_thresh',
                        'solver_name']
        losers = [opt for opt in reqd_options if opt not in self.FW_options]
        if (len(losers) > 0):
            msg = "FW_options is missing the following key(s): " + \
                  ", ".join(losers)
            raise RuntimeError(msg)

        # 3a. Check that the user did not specify the linearization of binary
        #    proximal terms (no binary variables allowed in FWPH QPs)
        if ('linearize_binary_proximal_terms' in self.options
            and self.options['linearize_binary_proximal_terms']):
            print('Warning: linearize_binary_proximal_terms cannot be used '
                  'with the FWPH algorithm. Ignoring...')
            self.options['linearize_binary_proximal_terms'] = False

        # 3b. Check that the user did not specify the linearization of all
        #    proximal terms (FWPH QPs should be QPs)
        if ('linearize_proximal_terms' in self.options
            and self.options['linearize_proximal_terms']):
            print('Warning: linearize_proximal_terms cannot be used '
                  'with the FWPH algorithm. Ignoring...')
            self.options['linearize_proximal_terms'] = False

        # 4. Provide a time limit of inf if the user did not specify
        if ('time_limit' not in self.FW_options or self.FW_options['time_limit'] is None):
            self.FW_options['time_limit'] = np.inf

    def _output(self, bound, best_bound, diff, secs):
        if (self.cylinder_rank == 0 and self.vb):
            print('{itr:3d} {bound:12.4f} {best_bound:12.4f} {diff:12.4e} {secs:11.1f}s'.format(
                    itr=self._PHIter, bound=bound, best_bound=best_bound, 
                    diff=diff, secs=secs))
        if (self.cylinder_rank == 0 and 'save_file' in self.FW_options.keys()):
            fname = self.FW_options['save_file']
            with open(fname, 'a') as f:
                f.write('{itr:d},{bound:.16f},{best_bound:.16f},{diff:.16f},{secs:.16f}\n'.format(
                    itr=self._PHIter, bound=bound, best_bound=best_bound,
                    diff=diff, secs=secs))

    def _output_header(self):
        if (self.cylinder_rank == 0 and self.vb):
            print('itr {bound:>12s} {bb:>12s} {cd:>12s} {tm:>12s}'.format(
                    bound="bound", bb="best bound", cd="conv diff", tm="time"))
        if (self.cylinder_rank == 0 and 'save_file' in self.FW_options.keys()):
            fname = self.FW_options['save_file']
            with open(fname, 'a') as f:
                f.write('{itr:s},{bound:s},{bb:s},{diff:s},{secs:s}\n'.format(
                    itr="Iteration", bound="Bound", bb="Best bound",
                    diff="Error", secs="Time(s)"))

    def save_weights(self, fname):
        ''' Save the computed weights to the specified file.

            Notes:
                Handles parallelism--only writes one copy of the file.

                Rather "fast-and-loose", in that it doesn't enforce _when_ this
                function can be called.
        '''
        weights = self._gather_weight_dict(strip_bundle_names=self.bundling) # None if rank != 0
        if (self.cylinder_rank != 0):
            return
        with open(fname, 'w') as f:
            for block in weights:
                for (scenario_name, wts) in block.items():
                    for (var_name, weight_val) in wts.items():
                        row = '{sn},{vn},{wv:.16f}\n'.format(
                            sn=scenario_name, vn=var_name, wv=weight_val)
                        f.write(row)

    def save_xbars(self, fname):
        ''' Save the computed xbar to the specified file.

            Notes:
                Handles parallelism--only writes one copy of the file.

                Rather "fast-and-loose", in that it doesn't enforce _when_ this
                function can be called.
        '''
        if (self.cylinder_rank != 0):
            return
        xbars = self._get_xbars(strip_bundle_names=self.bundling) # None if rank != 0
        with open(fname, 'w') as f:
            for (var_name, xbs) in xbars.items():
                row = '{vn},{vv:.16f}\n'.format(vn=var_name, vv=xbs)
                f.write(row)

    def _set_QP_objective(self):
        ''' Attach dual weights, objective function and solver to each QP.
        
            QP dual weights are initialized to the MIP dual weights.
        '''

        for name, mip in self.local_subproblems.items():
            QP = self.local_QP_subproblems[name]

            obj, new = self._extract_nonant_objective(mip)

            new += QP.recourse_cost
            ## Finish setting up objective for QP
            if self.bundling:
                m_source = self.local_scenarios[mip.scen_list[0]]
                x_source = QP.xr
            else:
                m_source = mip
                x_source = QP.x

            QP._mpisppy_model.W = pyo.Param(
                m_source._mpisppy_data.nonant_indices.keys(), mutable=True, initialize=m_source._mpisppy_model.W
            )
            # rhos are attached to each scenario, not each bundle (should they be?)
            ph_term = pyo.quicksum((
                QP._mpisppy_model.W[nni] * x_source[nni] +
                (m_source._mpisppy_model.rho[nni] / 2.) * (x_source[nni] - m_source._mpisppy_model.xbars[nni]) * (x_source[nni] - m_source._mpisppy_model.xbars[nni])
                for nni in m_source._mpisppy_data.nonant_indices
            ))

            if self.is_minimizing:
                QP.obj = pyo.Objective(expr=new+ph_term, sense=pyo.minimize)
            else:
                QP.obj = pyo.Objective(expr=-new+ph_term, sense=pyo.minimize)

            mip_obj_in_qp  = replace_expressions(obj, mip._mpisppy_data.mip_to_qp)
            QP._mpisppy_model.mip_obj_in_qp = mip_obj_in_qp
            ''' Attach a solver with various options '''
            solver = pyo.SolverFactory(self.FW_options['solver_name'])
            if sputils.is_persistent(solver):
                solver.set_instance(QP)
            if 'qp_solver_options' in self.FW_options:
                qp_opts = self.FW_options['qp_solver_options']
                if qp_opts:
                    for (key, option) in qp_opts.items():
                        solver.options[key] = option

            self.local_QP_subproblems[name]._solver_plugin = solver

    def _cache_nonant_var_swap_mip(self):
        """ cache the lists used for the nonant var swap """
        self._MIP_nonants = {}

        # MIP nonants
        for k, s in self.local_scenarios.items():
            nonant_vardata_lists = {}
            for node in s._mpisppy_node_list:
                nonant_vardata_lists[node.name] = node.nonant_vardata_list
            # this cache should have anything changed by _attach_nonant_indices
            self._MIP_nonants[s] = {
                "nonant_vardata_lists" : nonant_vardata_lists,
                "nonant_indices" : s._mpisppy_data.nonant_indices,
                "all_surrogate_nonants" : s._mpisppy_data.all_surrogate_nonants,
            }

    def _cache_nonant_var_swap_qp(self):
        """ cache the lists used for the nonant var swap """

        for (name, model) in self.local_subproblems.items():
            scens = model.scen_list if self.bundling else [name]
            for scenario_name in scens:
                scenario = self.local_scenarios[scenario_name]
                num_nonant_vars = scenario._mpisppy_data.nlens
                node_list = scenario._mpisppy_node_list
                for node in node_list:
                    node.nonant_vardata_list = [
                        self.local_QP_subproblems[name].xr[node.name,i]
                        if self.bundling else
                        self.local_QP_subproblems[name].x[node.name,i]
                        for i in range(num_nonant_vars[node.name])]
        self._attach_nonant_indices()

        self._QP_nonants = {}

        # QP nonants
        for k, s in self.local_scenarios.items():
            nonant_vardata_lists = {}
            for node in s._mpisppy_node_list:
                nonant_vardata_lists[node.name] = node.nonant_vardata_list
            # this cache should have anything changed by _attach_nonant_indices
            self._QP_nonants[s] = {
                "nonant_vardata_lists" : nonant_vardata_lists,
                "nonant_indices" : s._mpisppy_data.nonant_indices,
                "all_surrogate_nonants" : s._mpisppy_data.all_surrogate_nonants,
            }

        self._swap_nonant_vars_back()

    def _swap_nonant_vars(self):
        ''' Change the pointers in
            scenario._mpisppy_node_list[i].nonant_vardata_list
            to point to the QP variables, rather than the MIP variables.

            Notes:
                When computing xBar and updating the weights in the outer
                iteration, the values of the x variables are pulled from
                scenario._mpisppy_node_list[i].nonant_vardata_list. In the FWPH
                algorithm, xBar should be computed using the QP values, not the
                MIP values (like in normal PH).

                Reruns SPBase._attach_nonant_indices so that the scenario 
                _nonant_indices dictionary has the correct variable pointers
                
                Updates nonant_vardata_list but NOT nonant_list.
        '''
        for s, nonant_data in self._QP_nonants.items():
            for node in s._mpisppy_node_list:
                node.nonant_vardata_list = nonant_data["nonant_vardata_lists"][node.name]
            s._mpisppy_data.nonant_indices = nonant_data["nonant_indices"]
            s._mpisppy_data.all_surrogate_nonants = nonant_data["all_surrogate_nonants"]

    def _swap_nonant_vars_back(self):
        ''' Swap variables back, in case they're needed somewhere else.
        '''
        for s, nonant_data in self._MIP_nonants.items():
            for node in s._mpisppy_node_list:
                node.nonant_vardata_list = nonant_data["nonant_vardata_lists"][node.name]
            s._mpisppy_data.nonant_indices = nonant_data["nonant_indices"]
            s._mpisppy_data.all_surrogate_nonants = nonant_data["all_surrogate_nonants"]

    # need to overwrite a few methods due to how fwph manages things
    def _can_update_best_bound(self):
        for s in self.local_scenarios.values():
            for v in s._mpisppy_data.nonant_vars.values():
                if v.fixed:
                    if v not in self._initial_fixed_varibles:
                        return False
        return True

if __name__=='__main__':
    print('fwph.py has no main()')
