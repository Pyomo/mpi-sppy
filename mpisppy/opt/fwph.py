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

    Current implementation supports parallelism.
'''

import mpisppy.phbase
import mpisppy.utils.sputils as sputils
import numpy as np
import pyomo.environ as pyo
import time
import random
import math

from mpisppy import MPI
from mpisppy import global_toc
from pyomo.repn.standard_repn import generate_standard_repn
from mpisppy.utils import nice_join
from mpisppy.utils.sputils import find_active_objective
from pyomo.core.expr.visitor import replace_expressions
from pyomo.core.expr.numeric_expr import LinearExpression

from mpisppy.cylinders.xhatshufflelooper_bounder import ScenarioCycler
from mpisppy.cylinders.spwindow import Field
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

    def fwph_main(self, finalize=True):
        self.PH_Prep(attach_duals=True, attach_prox=False)
        self._output_header()
        self._attach_MIP_vars()
        self._cache_nonant_var_swap_mip()

        trivial_bound = self.Iter0()
        secs = time.perf_counter() - self.start_time
        self._output(trivial_bound, trivial_bound, np.nan, secs)
        self._fwph_best_bound = trivial_bound

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

        self._generate_initial_columns_if_needed()

        self._reenable_W()

        if (self.ph_converger):
            self.convobject = self.ph_converger(self)

        if self.options.get("FW_LP_start_iterations", 0) > 0:
            global_toc("Starting LP PH...")
            lp_iterations = self.options["FW_LP_start_iterations"]
            total_iterations = self.options["PHIterLimit"]
            self.options["PHIterLimit"] = lp_iterations
            integer_relaxer = pyo.TransformationFactory('core.relax_integer_vars')
            for s in self.local_subproblems.values():
                integer_relaxer.apply_to(s)
                if sputils.is_persistent(s._solver_plugin):
                    for v,_ in s._relaxed_integer_vars[None].values():
                        s._solver_plugin.update_var(v)
            self.attach_PH_to_objective(add_duals=False, add_prox=True)
            self._reenable_prox()
            super().iterk_loop()
            self._disable_prox()
            for s in self.local_subproblems.values():
                for v, d in s._relaxed_integer_vars[None].values():
                    v.domain = d
                    if sputils.is_persistent(s._solver_plugin):
                        s._solver_plugin.update_var(v)
                        # s._solver_plugin.update_var(v)
                s.del_component("_relaxed_integer_vars")
            self.options["PHIterLimit"] = total_iterations
            self._PHIter -= 1

            global_toc("Finished LP PH; Starting FW PH crossover")
            teeme = (
                self.options.get("tee-rank0-solves", False)
                and self.cylinder_rank == 0
            )
            # teeme = True
            self.fwph_solve_loop(
                mip_solver_options=self.current_solver_options,
                dtiming=self.options["display_timing"],
                tee=teeme,
                verbose=self.options["verbose"],
                # sdm_iter_limit=20,
                # don't cut off integer solutions
                # for this pass
                FW_conv_thresh=1e+10,
            )

        # sometimes we take a while to initialize,
        # sometimes LP may prove optimizality
        # check before entering the main loop
        if self.spcomm and self.spcomm.is_converged():
            if finalize:
                weight_dict = self._gather_weight_dict() # None if rank != 0
                xbars_dict  = self._get_xbars() # None if rank != 0
                return 0, weight_dict, xbars_dict
            return 0

        global_toc("Starting FW PH")

        self.iterk_loop()

        if finalize:
            weight_dict = self._gather_weight_dict() # None if rank != 0
            xbars_dict  = self._get_xbars() # None if rank != 0
            return self._PHIter, weight_dict, xbars_dict
        return self._PHIter

    def iterk_loop(self):

        verbose = self.options["verbose"]
        dprogress = self.options["display_progress"]
        dtiming = self.options["display_timing"]
        dconvergence_detail = self.options["display_convergence_detail"]
        teeme = (
            "tee-rank0-solves" in self.options
             and self.options["tee-rank0-solves"]
            and self.cylinder_rank == 0
        )
        # teeme = True

        self.conv = None

        max_iterations = int(self.options["PHIterLimit"])

        # The body of the algorithm
        while (self._PHIter < max_iterations):
            iteration_start_time = time.perf_counter()
            if dprogress:
                global_toc(f"Initiating FWPH Major Iteration {self._PHIter+1}\n", self.cylinder_rank == 0)

            # tbphloop = time.perf_counter()
            # TODO: should implement our own Xbar / W computation
            #       which just considers the QP subproblems
            self._swap_nonant_vars()
            self.Compute_Xbar(verbose)
            self.Update_W(verbose)
            self._swap_nonant_vars_back()

            if hasattr(self.spcomm, "sync_Ws"):
                self.spcomm.sync_Ws()

            self.conv = self.fwph_convergence_diff()

            if (self.extensions): 
                self.extobject.miditer()

            if (self.ph_converger):
                self._swap_nonant_vars()
                if (self.convobject.is_converged()):
                    secs = time.perf_counter() - self.start_time
                    self._output(self._local_bound, self._fwph_best_bound, self.conv, secs)
                    global_toc('FWPH converged to user-specified criteria', self.cylinder_rank == 0)
                    self._swap_nonant_vars_back()
                    break
                self._swap_nonant_vars_back()
            if self.conv is not None: # Convergence check from Boland
                if (self.conv < self.options['convthresh']):
                    secs = time.perf_counter() - self.start_time
                    self._output(self._local_bound, self._fwph_best_bound, self.conv, secs)
                    global_toc(
                        "FWPH convergence metric=%f dropped below user-supplied threshold=%f" % (self.conv, self.options["convthresh"]),
                        self.cylinder_rank == 0,
                    )
                    break

            if (self._is_timed_out()):
                global_toc(f"Time limit {self.options['time_limit']} seconds reached.", self.cylinder_rank == 0)
                break

            self.fwph_solve_loop(
                mip_solver_options=self.current_solver_options,
                dtiming=dtiming,
                tee=teeme,
                verbose=verbose
            )

            if (self.extensions): 
                self.extobject.enditer()

            secs = time.perf_counter() - self.start_time
            self._output(self._local_bound, self._fwph_best_bound, self.conv, secs)

            ## Hubs/spokes take precedence over convergers
            if self.spcomm and self.spcomm.is_converged(screen_trace=False):
                secs = time.perf_counter() - self.start_time
                self._output(self._local_bound, self._fwph_best_bound, np.nan, secs)
                global_toc("Cylinder convergence", self.cylinder_rank == 0)
                break

            if (self.extensions):
                self.extobject.enditer_after_sync()

            if dprogress and self.cylinder_rank == 0:
                print("")
                print("After FWPH Iteration",self._PHIter)
                print("FWPH Convergence Metric=",self.conv)
                print("Iteration time: %6.2f" % (time.perf_counter() - iteration_start_time))
                print("Elapsed time:   %6.2f" % (time.perf_counter() - self.start_time))

            if dconvergence_detail:
                self.report_var_values_at_rank0(header="Convergence detail:", fixed_vars=False)

            # tphloop = time.perf_counter() - tbphloop
            # print(f"PH iter {self._PHIter}, total time: {tphloop}")
        else: # no break, (self._PHIter == max_iterations)
            # NOTE: If we return for any other reason things are reasonably in-sync.
            #       due to the convergence check. However, here we return we'll be
            #       out-of-sync because of the solve_loop could take vasty different
            #       times on different threads. This can especially mess up finalization.
            #       As a guard, we'll put a barrier here.
            self.mpicomm.Barrier()
            global_toc("Reached user-specified limit=%d on number of FWPH iterations" % max_iterations, self.cylinder_rank == 0)

    def fwph_solve_loop(
            self,
            mip_solver_options=None,
            dtiming=False,
            tee=False,
            verbose=False,
            sdm_iter_limit=None,
            FW_conv_thresh=None,
        ):
            if sdm_iter_limit is None:
                sdm_iter_limit = self.FW_options["FW_iter_limit"]
            if FW_conv_thresh is None:
                FW_conv_thresh = self.FW_options["FW_conv_thresh"]
            max_iterations = int(self.options["PHIterLimit"])
            # print(f"{sdm_iter_limit=}")
            self._swap_nonant_vars()
            self._local_bound = 0
            # tbsdm = time.perf_counter()
            _sdm_generators = {}
            stop = False
            best_bound_update = self._can_update_best_bound()
            for name in self.local_subproblems:
                _sdm_generators[name] = self.SDM(name, mip_solver_options, dtiming, tee, verbose, sdm_iter_limit, FW_conv_thresh, best_bound_update)
                try:
                    dual_bound = next(_sdm_generators[name])
                except StopIteration as e:
                    dual_bound = e.value
                    stop = True
                self._local_bound += self.local_subproblems[name]._mpisppy_probability * \
                                     dual_bound
            self._update_dual_bounds()
            self._PHIter += 1
            if self._PHIter == max_iterations:
                stop = True
            if self._sync_after_mip_solve():
                stop = True
            stop = self.allreduce_or(stop)
            while not stop:
                stop = False
                for col_generator in _sdm_generators.values():
                    try:
                        next(col_generator)
                    except StopIteration:
                       stop = True
                self._PHIter += 1
                if self._PHIter == max_iterations:
                    stop = True
                if self._sync_after_mip_solve():
                    stop = True
                stop = self.allreduce_or(stop)
            # tsdm = time.perf_counter() - tbsdm
            # print(f"PH iter {self._PHIter}, total SDM time: {tsdm}")

            # Re-set the mip._mpisppy_model.W so that the QP objective
            # is correct in the next major iteration
            for model_name, mip in self.local_subproblems.items():
                qp  = self.local_QP_subproblems[model_name]
                scen_mip = self.local_scenarios[model_name]
                for (node_name, ix) in scen_mip._mpisppy_data.nonant_indices:
                    scen_mip._mpisppy_model.W[node_name, ix]._value = \
                        qp._mpisppy_model.W[node_name, ix]._value

            self._swap_nonant_vars_back()

    def _sync_after_mip_solve(self):
        # add columns from cylinder(s)
        self._swap_nonant_vars_back()
        # spoke or hub (FWPH_Cylinder)
        if hasattr(self.spcomm, "add_cylinder_columns"):
            self.spcomm.add_cylinder_columns()
        # hub
        if hasattr(self.spcomm, "sync_bounds"):
            self.spcomm.sync_nonants()
            self.spcomm.sync_bounds()
            self.spcomm.sync_extensions()
        # spoke
        elif hasattr(self.spcomm, "sync"):
            self.spcomm.sync()
        self._swap_nonant_vars()
        if self.spcomm and self.spcomm.is_converged():
            return True
        return False

    def SDM(self, model_name, mip_solver_options, dtiming, tee, verbose, sdm_iter_limit, FW_conv_thresh, best_bound_update):
        '''  Algorithm 2 in Boland et al. (with small tweaks)
        '''
        mip = self.local_subproblems[model_name]
        qp  = self.local_QP_subproblems[model_name]
    
        # Set the QP dual weights to the correct values.
        arb_scen_mip = self.local_scenarios[model_name]

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

        for itr in range(sdm_iter_limit):
            # loop_start = time.perf_counter()
            # Algorithm 2 line 4
            scen_mip = self.local_scenarios[model_name]
            for ndn_i in scen_mip._mpisppy_data.nonant_indices:
                scen_mip._mpisppy_model.W[ndn_i]._value = (
                    qp._mpisppy_model.W[ndn_i]._value
                    + scen_mip._mpisppy_model.rho[ndn_i]._value
                    * (xt[ndn_i]
                    -  scen_mip._mpisppy_model.xbars[ndn_i]._value))

            self._fix_fixings(model_name, mip, qp)
            cutoff = self._add_objective_cutoff(mip, qp, model_name, best_bound_update, FW_conv_thresh)
            # print(f"{model_name=}, {cutoff=}")
            # Algorithm 2 line 5
            self.solve_one(
                mip_solver_options,
                model_name,
                mip,
                dtiming=dtiming,
                tee=tee,
                verbose=verbose,
                # if the problem isn't feasible,
                # becuase of the cutoff, we should
                # probably keep going ??
                need_solution=False,
                warmstart=sputils.WarmstartStatus.PRIOR_SOLUTION,
            )
            self._remove_objective_cutoff(mip)
            # tmipsolve = time.perf_counter() - tbmipsolve

            if mip._mpisppy_data.solution_available:

                # Algorithm 2 line 9 (compute \Gamma^t)
                inner_bound = mip._mpisppy_data.inner_bound
                # print(f"{model_name=}, {inner_bound=}")
                gamma_t = self._compute_gamma_t(cutoff, inner_bound)
                # print(f"{itr=}, {model_name=}, {gamma_t=}")

                # tbcol = time.perf_counter()
                self._add_QP_column(model_name)
                # tcol = time.perf_counter() - tbcol
                # print(f"{model_name} QP add_column time: {tcol}")

            else:
                global_toc(f"{self.__class__.__name__}: Could not find an improving column for {model_name}!", True)
                # couldn't find an improving direction, the column would not become active

            # add a shared column(s)
            shared_columns = self.options.get("FWPH_shared_columns_per_iteration", 0)
            if shared_columns > 0:
                self._swap_nonant_vars_back()
                self._add_shared_columns(shared_columns)
                self._swap_nonant_vars()

            # tbqpsol = time.perf_counter()
            self.solve_one(
                self.options["qp_solver_options"],
                model_name,
                qp,
                dtiming=dtiming,
                tee=tee,
                verbose=verbose,
            )
            # tqpsol = time.perf_counter() - tbqpsol

            # print(f"{model_name}, solve + add_col time: {tmipsolve + tcol + tqpsol}")
            # fwloop = time.perf_counter() - loop_start
            # print(f"{model_name}, total loop time: {fwloop}")
            # Algorithm 2 lines 6--8
            # Stopping after the MIP solve will give a point
            # to synchronize with spokes
            dual_bound = None
            if (itr == 0):
                dual_bound = mip._mpisppy_data.outer_bound

            if itr + 1 == sdm_iter_limit or not mip._mpisppy_data.solution_available or gamma_t < FW_conv_thresh:
                return dual_bound
            else:
                yield dual_bound

            # reset for next loop
            for ndn_i, xvar in arb_scen_mip._mpisppy_data.nonant_indices.items():
                xt[ndn_i] = xvar._value


    def _add_shared_columns(self, shared_columns):
        self.mpicomm.Barrier()
        self._disable_W()
        for s in self.local_subproblems.values():
            if sputils.is_persistent(s._solver_plugin):
                active_objective_datas = list(s.component_data_objects(
                     pyo.Objective, active=True, descend_into=True))
                s._solver_plugin.set_objective(active_objective_datas[0])
        self._generate_shared_column(shared_columns)
        self._reenable_W()

    def _fix_fixings(self, model_name, mip, qp):
        """ If some variable is fixed in the mip, but its value in the QP does
            not agree with that fixed value, we will have a bad time. This method
            removes such fixings.
        """
        for var in qp.x.values():
            if var.fixed:
                raise RuntimeError(f"var {var.name} is fixed in QP!!")
        solver = mip._solver_plugin
        unfixed = 0
        target = mip._mpisppy_data.nonant_vars
        mip_to_qp = mip._mpisppy_data.mip_to_qp
        for ndn_i, var in target.items():
            if var.fixed:
                if not math.isclose(mip_to_qp[id(var)].value, var.value, abs_tol=1e-5):
                    var.unfix()
                    if sputils.is_persistent(solver):
                        solver.update_var(var)
                    unfixed += 1
        if unfixed > 0:
            global_toc(f"{self.__class__.__name__}: unfixed {unfixed} nonant variables in {model_name}", True)

    def _add_QP_column(self, model_name, disable_W=False):
        ''' Add a column to the QP, with values taken from the most recent MIP
            solve. Assumes the inner_bound is up-to-date in the MIP model.
        '''
        mip = self.local_subproblems[model_name]
        qp  = self.local_QP_subproblems[model_name]
        solver = qp._solver_plugin
        persistent = sputils.is_persistent(solver)

        if disable_W:
            self._disable_W()
        total_recourse_cost = mip._mpisppy_data.inner_bound - pyo.value(mip._mpisppy_model.nonant_obj_part)
        if disable_W:
            self._reenable_W()

        if hasattr(solver, 'add_column'):
            new_var = qp.a.add()
            coef_list = [1.]
            constr_list = [qp.sum_one]
            for (node, ix) in qp.eqx.index_set():
                coef_list.append(mip._mpisppy_data.nonant_vars[node, ix].value)
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

        for (node, ix) in qp.eqx.index_set():
            lb, body, ub = qp.eqx[node, ix].to_bounded_expression()
            body += new_var * mip._mpisppy_data.nonant_vars[node, ix].value
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

    def _compute_gamma_t(self, cutoff, inner_bound):
        if abs(cutoff) > 1e-9:
            stop_check = (cutoff - inner_bound) / abs(cutoff) # \Gamma^t in Boland, but normalized
        else:
            stop_check = cutoff - inner_bound # \Gamma^t in Boland
        # print(f"{model_name}, Gamma^t = {stop_check}")
        stop_check_tol = self.FW_options.get("stop_check_tol", 1e-4)
        if (self.is_minimizing and stop_check < -stop_check_tol):
            print('Warning (fwph): convergence quantity Gamma^t = '
                 '{sc:.2e} (should be non-negative)'.format(sc=stop_check))
            print('Try decreasing the MIP gap tolerance and re-solving')
        elif (not self.is_minimizing and stop_check > stop_check_tol):
            print('Warning (fwph): convergence quantity Gamma^t = '
                 '{sc:.2e} (should be non-positive)'.format(sc=stop_check))
            print('Try decreasing the MIP gap tolerance and re-solving')
        return stop_check

    def _add_objective_cutoff(self, mip, qp, model_name, best_bound_update, FW_conv_thresh):
        """ Add a constraint to the MIP objective ensuring
            an improving direction in the QP subproblem is generated
        """
        assert not hasattr(mip._mpisppy_model, "obj_cutoff_constraint")
        # print(f"\tnonants part: {pyo.value(qp._mpisppy_model.mip_obj_in_qp)}")
        # print(f"\trecoursepart: {pyo.value(qp.recourse_cost)}")
        cutoff = pyo.value(qp._mpisppy_model.mip_obj_in_qp) + pyo.value(qp.recourse_cost)
        epsilon = FW_conv_thresh
        # normalized Gamma^t
        epsilon = max(epsilon, abs(epsilon*cutoff))
        # tbmipsolve = time.perf_counter()
        active_objective = sputils.find_active_objective(mip)
        if best_bound_update:
            # If we're providing the best bound, we better make sure the current solution
            # is feasible. Otherwise we can't prove its the best.
            current_obj_value = pyo.value(active_objective)
            # global_toc(f"{model_name}, {current_obj_value=}, {cutoff=}", True)
            if self.is_minimizing:
                cutoff = max(current_obj_value, cutoff)
            else:
                cutoff = min(current_obj_value, cutoff)
        if self.is_minimizing:
            # obj <= cutoff
            obj_cutoff_constraint = (None, active_objective.expr, cutoff+epsilon)
        else:
            # obj >= cutoff
            obj_cutoff_constraint = (cutoff-epsilon, active_objective.expr, None)
        mip._mpisppy_model.obj_cutoff_constraint = pyo.Constraint(expr=obj_cutoff_constraint)
        if sputils.is_persistent(mip._solver_plugin):
            mip._solver_plugin.add_constraint(mip._mpisppy_model.obj_cutoff_constraint)
        return cutoff

    def _remove_objective_cutoff(self, mip):
        """ Remove the constraint to the MIP objective added by
            _add_objective_cutoff
        """
        assert hasattr(mip._mpisppy_model, "obj_cutoff_constraint")
        if sputils.is_persistent(mip._solver_plugin):
            mip._solver_plugin.remove_constraint(mip._mpisppy_model.obj_cutoff_constraint)
        mip.del_component(mip._mpisppy_model.obj_cutoff_constraint)
        delattr(mip._mpisppy_model, "obj_cutoff_constraint")

    def _attach_indices(self):
        ''' Attach the fields x_indices to the model objects in
            self.local_subproblems (not self.local_scenarios, nor
            self.local_QP_subproblems).

            x_indices is a list of tuples of the form...
                (node name, variable index)
            
            Must be called after the subproblems (MIPs AND QPs) are created.
        '''
        for mip in self.local_subproblems.values():
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
        for (name, mip) in self.local_scenarios.items():
            mip._mpisppy_data.nonant_vars = mip._mpisppy_data.nonant_indices
            self._attach_nonant_objective(mip)

    def _update_dual_bounds(self):
        ''' Compute the FWPH dual bound using self._local_bound from each rank
        '''
        send = np.array(self._local_bound)
        recv = np.array(0.)
        self.comms['ROOT'].Allreduce(
            [send, MPI.DOUBLE], [recv, MPI.DOUBLE], op=MPI.SUM)
        self._local_bound = recv

        if (self.is_minimizing):
            self._fwph_best_bound = np.fmax(self._fwph_best_bound, self._local_bound)
        else:
            self._fwph_best_bound = np.fmin(self._fwph_best_bound, self._local_bound)
        if self._can_update_best_bound():
            self.best_bound_obj_val = self._fwph_best_bound
        # if self.cylinder_rank == 0:
        #     print(f"{self._local_bound=}")

    def fwph_convergence_diff(self):
        ''' Perform the convergence check of Algorithm 3 in Boland et al. '''
        diff = 0.
        for name in self.local_subproblems.keys():
            mip = self.local_subproblems[name]
            qp  = self.local_QP_subproblems[name]
            xbars = mip._mpisppy_model.xbars
            diff_s = mip._mpisppy_probability * sum((qp.x[idx]._value - xbars[idx]._value)**2
                                                    for idx in mip._mpisppy_data.nonant_indices)
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
                mip (Pyomo ConcreteModel): MIP model for a scenario.
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
                mip (Pyomo ConcreteModel): MIP model for a scenario.

            Returns:
                obj (Pyomo Objective): objective function extracted
                    from the MIP
                new (Pyomo Expression): expression from the MIP model
                    objective with the MIP variables replaced by QP variables.
                    Does not inculde dual or prox terms.
        '''
        obj = mip._mpisppy_model.nonant_obj_part
        mip_to_qp = mip._mpisppy_data.mip_to_qp
        linear_vars = [mip_to_qp[id(var)] for var in obj.linear_vars]
        linear_coefs = [pyo.value(coef) for coef in obj.linear_coefs]
        new = LinearExpression(
            linear_coefs=linear_coefs, linear_vars=linear_vars
        )
        return obj, new

    def _gather_weight_dict(self):
        ''' Compute a double nested dictionary of the form

                weights[scenario name][variable name] = weight value

            for FWPH to return to the user.

            Notes:
                Must be called after the variables are swapped back.
        '''
        local_weights = dict()
        for (name, scenario) in self.local_scenarios.items():
            scenario_weights = {nonant.name: scenario._mpisppy_model.W[ndn_ix].value
                    for ndn_ix, nonant in scenario._mpisppy_data.nonant_indices.items()}
            local_weights[name] = scenario_weights

        weights = self.comms['ROOT'].gather(local_weights, root=0)
        return weights

    def _get_xbars(self):
        ''' Return the xbar vector if rank = 0 and None, otherwise
            (Consistent with _gather_weight_dict).

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
                    xbar_dict[var_name] = scenario._mpisppy_model.xbars[node.name, ix].value
            return xbar_dict

    def _initialize_QP_subproblems(self):
        ''' Instantiates the (convex) QP subproblems (eqn. (13) in the Boland
            paper) for each scenario. Does not create/attach an objective.

            Attachs a local_QP_subproblems dict to self. Keys are scenario
            names, values are Pyomo ConcreteModel objects
            corresponding to the QP subproblems. 

            QP subproblems are in their original form, without the x and y
            variables eliminated. Rationale: pre-solve will get this, easier
            bookkeeping (objective does not need to be changed at each inner
            iteration this way).
        '''
        self.local_QP_subproblems = dict()
        for (name, model) in self.local_subproblems.items():
            nonant_indices = model._mpisppy_data.nonant_indices.keys()

            ''' Convex comb. coefficients '''
            QP = pyo.ConcreteModel()
            QP.a = pyo.VarList(domain=pyo.NonNegativeReals)
            QP.a.add() # Just one variable (1-based index!) to start

            ''' Other variables '''
            QP.x = pyo.Var(nonant_indices, within=pyo.Reals)
            mip_recourse_cost = model._mpisppy_data.inner_bound - pyo.value(model._mpisppy_model.nonant_obj_part)
            QP.recourse_cost = pyo.Var(within=pyo.Reals, initialize=mip_recourse_cost)

            ''' (x,y) constraints '''
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

    def _generate_initial_columns_if_needed(self):
        if self.spcomm is not None:
            if self.spcomm.receive_field_spcomms[Field.BEST_XHAT]:
                # we'll get the initial columns from the incumbent finder spoke
                return
        number_initial_column_tries = self.options.get("FW_initialization_attempts", 10)
        if self.FW_options["FW_iter_limit"] == 1 and number_initial_column_tries < 1:
            global_toc(f"{self.__class__.__name__}: Warning: FWPH needs an initial shared column if FW_iter_limit == 1. Increasing FW_iter_limit to 2 to ensure convergence", self.cylinder_rank == 0)
            self.FW_options["FW_iter_limit"] = 2
        if self.FW_options["FW_iter_limit"] == 1 or number_initial_column_tries > 0:
            number_points = self._generate_shared_column(number_initial_column_tries)
            if number_points == 0 and self.FW_options["FW_iter_limit"] == 1:
                global_toc(f"{self.__class__.__name__}: Warning: FWPH failed to find an initial feasible solution. Increasing FW_iter_limit to 2 to ensure convergence", self.cylinder_rank == 0)
                self.FW_options["FW_iter_limit"] = 2

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
                                   solver_options = self.options["iter0_solver_options"],
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
        return self.allreduce_or( (time.perf_counter() - self.start_time) >= self.options["time_limit"] )

    def _options_checks_fw(self):
        ''' Name                Boland notation (Algorithm 2)
            -------------------------------------------------
            FW_iter_limit       t_max
            FW_weight           alpha
            FW_conv_thresh      tau
        '''
        # 1. Check for required options
        reqd_options = ['FW_iter_limit', 'FW_weight', 'FW_conv_thresh', 'solver_name']
        missing = [opt for opt in reqd_options if opt not in self.FW_options]
        if missing:
            raise RuntimeError(
                f"FW_options misses the following key(s): {nice_join(missing, conjunction='and')}."
            )

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
                    diff="conv diff", secs="Time(s)"))

    def save_weights(self, fname):
        ''' Save the computed weights to the specified file.

            Notes:
                Handles parallelism--only writes one copy of the file.

                Rather "fast-and-loose", in that it doesn't enforce _when_ this
                function can be called.
        '''
        weights = self._gather_weight_dict() # None if rank != 0
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
        xbars = self._get_xbars() # None if rank != 0
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
            QP._mpisppy_model.W = pyo.Param(
                mip._mpisppy_data.nonant_indices.keys(), mutable=True, initialize=mip._mpisppy_model.W
            )
            ph_term = pyo.quicksum((
                QP._mpisppy_model.W[nni] * QP.x[nni] +
                (mip._mpisppy_model.rho[nni] / 2.) * (QP.x[nni] - mip._mpisppy_model.xbars[nni]) * (QP.x[nni] - mip._mpisppy_model.xbars[nni])
                for nni in mip._mpisppy_data.nonant_indices
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
            scenario = self.local_scenarios[name]
            num_nonant_vars = scenario._mpisppy_data.nlens
            node_list = scenario._mpisppy_node_list
            for node in node_list:
                node.nonant_vardata_list = [
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
