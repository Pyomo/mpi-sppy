###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
from mpisppy.extensions.extension import Extension
from mpisppy.utils.sputils import find_active_objective
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import LinearExpression
from mpisppy.cylinders.cross_scen_spoke import CrossScenarioCutSpoke

import pyomo.environ as pyo
import sys
import math
import mpisppy.utils.sputils as sputils
import numpy as np
import mpisppy.MPI as mpi

class CrossScenarioExtension(Extension):
    def __init__(self, spbase_object):
        super().__init__(spbase_object)
        if self.opt.multistage:
            raise RuntimeError('CrossScenarioExtension only supports '
                                'two-stage models at this time')

        opt = self.opt
        if 'cross_scen_options' in opt.options and \
                'check_bound_improve_iterations' in opt.options['cross_scen_options']:
            self.check_bound_iterations = opt.options['cross_scen_options']['check_bound_improve_iterations']
        else:
            self.check_bound_iterations = None

        self.cur_ib = None
        self.iter_at_cur_ib = 1

        self.cur_ob = None

        self.reenable_W = None
        self.reenable_prox = None

        self.any_cuts = False
        self.iter_since_last_check = 0

    def _disable_W_and_prox(self):
        assert self.reenable_W is None
        assert self.reenable_prox is None
        ## hold the PH object harmless
        opt = self.opt
        self.reenable_W = False
        self.reenable_prox = False
        if not opt.W_disabled and not opt.prox_disabled:
            opt.disable_W_and_prox()
            self.reenable_W = True
            self.reenable_prox = True 
        elif not opt.W_disabled:
            opt._disable_W()
            self.eenable_W = True
        elif not opt.prox_disabled:
            opt._disable_prox()
            self.reenable_prox = True 

    def _enable_W_and_prox(self):
        assert self.reenable_W is not None
        assert self.reenable_prox is not None
        
        opt = self.opt
        if self.reenable_W and self.reenable_prox:
            opt.reenable_W_and_prox()
        elif self.reenable_W:
            opt._reenable_W()
        elif self.reenable_prox:
            opt._reenable_prox()

        self.reenable_W = None
        self.reenable_prox = None

    def _check_bound(self):
        opt = self.opt

        cached_ph_obj = dict()

        for k,s in opt.local_subproblems.items():
            phobj = find_active_objective(s)
            phobj.deactivate()
            cached_ph_obj[k] = phobj
            s._mpisppy_model.EF_Obj.activate()

        teeme = (
            "tee-rank0-solves" in opt.options
             and opt.options["tee-rank0-solves"]
        )
        opt.solve_loop(
                solver_options=opt.current_solver_options,
                dtiming=opt.options["display_timing"],
                gripe=True,
                disable_pyomo_signal_handling=False,
                tee=teeme,
                verbose=opt.options["verbose"],
        )

        local_obs = np.fromiter((s._mpisppy_data.outer_bound for s in opt.local_subproblems.values()),
                                dtype="d", count=len(opt.local_subproblems))

        local_ob = np.empty(1)
        if opt.is_minimizing:
            local_ob[0] = local_obs.max()
        else:
            local_ob[0] = local_obs.min()

        global_ob = np.empty(1)

        if opt.is_minimizing:
            opt.mpicomm.Allreduce(local_ob, global_ob, op=mpi.MAX)
        else: 
            opt.mpicomm.Allreduce(local_ob, global_ob, op=mpi.MIN)

        #print(f"CrossScenarioExtension OB: {global_ob[0]}")

        opt.spcomm.BestOuterBound = opt.spcomm.OuterBoundUpdate(global_ob[0], char='C')

        for k,s in opt.local_subproblems.items():
            s._mpisppy_model.EF_Obj.deactivate()
            cached_ph_obj[k].activate()

    def get_from_cross_cuts(self):
        spcomm = self.opt.spcomm
        idx = self.cut_gen_spoke_index
        receive_buffer = np.empty(spcomm.remote_lengths[idx - 1] + 1, dtype="d") # Must be doubles
        is_new = spcomm.hub_from_spoke(receive_buffer, idx)
        if is_new:
            self.make_cuts(receive_buffer)

    def send_to_cross_cuts(self):
        idx = self.cut_gen_spoke_index

        # get the stuff we want to send
        self.opt._save_nonants()
        ci = 0  ## index to self.nonant_send_buffer

        # get all the nonants
        all_nonants_and_etas = self.all_nonants_and_etas
        for k, s in self.opt.local_scenarios.items():
            for xvar in s._mpisppy_data.nonant_indices.values():
                all_nonants_and_etas[ci] = xvar._value
                ci += 1

        # get all the etas
        for k, s in self.opt.local_scenarios.items():
            for sn in self.opt.all_scenario_names:
                all_nonants_and_etas[ci] = s._mpisppy_model.eta[sn]._value
                ci += 1
        self.opt.spcomm.hub_to_spoke(all_nonants_and_etas, idx)

    def make_cuts(self, coefs):
        # take the coefficient array and assemble cuts accordingly

        # this should have already been set in the extension !
        opt = self.opt

        # rows are:
        # [ const, eta_coeff, *nonant_coeffs ]
        row_len = 1+1+self.nonant_len
        outer_iter = int(coefs[-1])

        if opt.bundling:
            for bn,b in opt.local_subproblems.items():
                persistent_solver = sputils.is_persistent(b._solver_plugin)
                ## get an arbitrary scenario
                s = opt.local_scenarios[b.scen_list[0]]
                for idx, k in enumerate(opt.all_scenario_names):
                    row = coefs[row_len*idx:row_len*(idx+1)]
                    # the row could be all zeros,
                    # which doesn't do anything
                    if (row == 0.).all():
                        continue
                    # rows are:
                    # [ const, eta_coeff, *nonant_coeffs ]
                    linear_const = row[0]
                    linear_coefs = list(row[1:])
                    linear_vars = [b._mpisppy_model.eta[k]]

                    for ndn_i in s._mpisppy_data.nonant_indices:
                        ## for bundles, we add the constrains only
                        ## to the reference first stage variables
                        linear_vars.append(b.ref_vars[ndn_i])

                    cut_expr = LinearExpression(constant=linear_const, linear_coefs=linear_coefs,
                                                linear_vars=linear_vars)
                    b._mpisppy_model.benders_cuts[outer_iter, k] = (None, cut_expr, 0)
                    if persistent_solver:
                        b._solver_plugin.add_constraint(b._mpisppy_model.benders_cuts[outer_iter, k])

        else:
            for sn,s in opt.local_subproblems.items():
                persistent_solver = sputils.is_persistent(s._solver_plugin)
                for idx, k in enumerate(opt.all_scenario_names):
                    row = coefs[row_len*idx:row_len*(idx+1)]
                    # the row could be all zeros,
                    # which doesn't do anything
                    if (row == 0.).all():
                        continue
                    # rows are:
                    # [ const, eta_coeff, *nonant_coeffs ]
                    linear_const = row[0]
                    linear_coefs = list(row[1:])
                    linear_vars = [s._mpisppy_model.eta[k]]
                    linear_vars.extend(s._mpisppy_data.nonant_indices.values())

                    cut_expr = LinearExpression(constant=linear_const, linear_coefs=linear_coefs,
                                                linear_vars=linear_vars)
                    s._mpisppy_model.benders_cuts[outer_iter, k] = (None, cut_expr, 0.)
                    if persistent_solver:
                        s._solver_plugin.add_constraint(s._mpisppy_model.benders_cuts[outer_iter, k])

        # NOTE: the LShaped code negates the objective, so
        #       we do the same here for consistency
        ib = self.opt.spcomm.BestInnerBound
        ob = self.opt.spcomm.BestOuterBound
        if not opt.is_minimizing:
            ib = -ib
            ob = -ob
        add_cut = (math.isfinite(ib) or math.isfinite(ob)) and \
                ((ib < self.best_inner_bound) or (ob > self.best_outer_bound))
        if add_cut:
            self.best_inner_bound = ib
            self.best_outer_bound = ob
            for sn,s in opt.local_subproblems.items():
                persistent_solver = sputils.is_persistent(s._solver_plugin)
                prior_outer_iter = list(s._mpisppy_model.inner_bound_constr.keys())
                s._mpisppy_model.inner_bound_constr[outer_iter] = (ob, s._mpisppy_model.EF_obj, ib)
                if persistent_solver:
                    s._solver_plugin.add_constraint(s._mpisppy_model.inner_bound_constr[outer_iter])
                # remove other ib constraints (we only need the tightest)
                for it in prior_outer_iter:
                    if persistent_solver:
                        s._solver_plugin.remove_constraint(s._mpisppy_model.inner_bound_constr[it])
                    del s._mpisppy_model.inner_bound_constr[it]

        ## helping the extention track cuts
        self.new_cuts = True

    def setup_hub(self):
        idx = self.cut_gen_spoke_index
        self.all_nonants_and_etas = np.zeros(self.opt.spcomm.local_lengths[idx - 1] + 1)

        self.nonant_len = self.opt.nonant_length

        # save the best bounds so far
        self.best_inner_bound = math.inf
        self.best_outer_bound = -math.inf

        # helping the extension track cuts
        self.new_cuts = False

    def initialize_spoke_indices(self):
        for (i, spoke) in enumerate(self.opt.spcomm.spokes):
            if spoke["spoke_class"] == CrossScenarioCutSpoke:
                self.cut_gen_spoke_index = i + 1

    def sync_with_spokes(self):
        self.send_to_cross_cuts()
        self.get_from_cross_cuts()

    def pre_iter0(self):
        if self.opt.multistage:
            raise RuntimeError("CrossScenarioExtension does not support "
                               "multi-stage problems at this time")
        ## hack as this provides logs for outer bounds
        self.opt.spcomm.has_outerbound_spokes = True

    def post_iter0(self):
        opt = self.opt
        # NOTE: the LShaped code negates the objective, so
        #       we do the same here for consistency
        if 'cross_scen_options' in opt.options and \
                'valid_eta_bound' in opt.options['cross_scen_options']:
            valid_eta_bound = opt.options['cross_scen_options']['valid_eta_bound']
            if not opt.is_minimizing:
                _eta_init = { k: -v for k,v in valid_eta_bound.items() }
            else:
                _eta_init = valid_eta_bound
            def _eta_bounds(m, k):
                return _eta_init[k], None
        else:
            lb = (-sys.maxsize - 1) * 1. / len(opt.all_scenario_names)
            def _eta_init(m, k):
                return lb
            def _eta_bounds(m, k):
                return lb, None

        # eta is attached to each subproblem, regardless of bundles
        bundling = opt.bundling
        for k,s in opt.local_subproblems.items():
            s._mpisppy_model.eta = pyo.Var(opt.all_scenario_names, initialize=_eta_init, bounds=_eta_bounds)
            if sputils.is_persistent(s._solver_plugin):
                for var in s._mpisppy_model.eta.values():
                    s._solver_plugin.add_var(var)
            if bundling: ## create a refence to eta on each subproblem
                for sn in s.scen_list:
                    scenario = opt.local_scenarios[sn]
                    scenario._mpisppy_model.eta = { k : s._mpisppy_model.eta[k] for k in opt.all_scenario_names }

        ## hold the PH object harmless
        self._disable_W_and_prox()
        
        for k,s in opt.local_subproblems.items():

            obj = find_active_objective(s)

            repn = generate_standard_repn(obj.expr, quadratic=True)
            if len(repn.nonlinear_vars) > 0:
                raise ValueError("CrossScenario does not support models with nonlinear objective functions")

            if bundling:
                ## NOTE: this is slighly wasteful, in that for a bundle
                ##       the first-stage cost appears len(s.scen_list) times
                ##       If this really made a difference, we could use s.ref_vars
                ##       to do the substitution
                nonant_vardata_list = list()
                for sn in s.scen_list:
                    nonant_vardata_list.extend( \
                            opt.local_scenarios[sn]._mpisppy_node_list[0].nonant_vardata_list)
            else:
                nonant_vardata_list = s._mpisppy_node_list[0].nonant_vardata_list

            nonant_ids = set((id(var) for var in nonant_vardata_list))

            linear_coefs = list(repn.linear_coefs)
            linear_vars = list(repn.linear_vars)

            quadratic_coefs = list(repn.quadratic_coefs)

            # adjust coefficients by scenario/bundle probability
            scen_prob = s._mpisppy_probability
            for i,var in enumerate(repn.linear_vars):
                if id(var) not in nonant_ids:
                    linear_coefs[i] *= scen_prob

            for i,(x,y) in enumerate(repn.quadratic_vars):
                # only multiply through once
                if id(x) not in nonant_ids:
                    quadratic_coefs[i] *= scen_prob
                elif id(y) not in nonant_ids:
                    quadratic_coefs[i] *= scen_prob

            # NOTE: the LShaped code negates the objective, so
            #       we do the same here for consistency
            if not opt.is_minimizing:
                for i,coef in enumerate(linear_coefs):
                    linear_coefs[i] = -coef
                for i,coef in enumerate(quadratic_coefs):
                    quadratic_coefs[i] = -coef

            # add the other etas
            if bundling:
                these_scenarios = set(s.scen_list)
            else:
                these_scenarios = [k]

            eta_scenarios = list()
            for sn in opt.all_scenario_names:
                if sn not in these_scenarios:
                    linear_coefs.append(1)
                    linear_vars.append(s._mpisppy_model.eta[sn])
                    eta_scenarios.append(sn)

            expr = LinearExpression(constant=repn.constant, linear_coefs=linear_coefs,
                                    linear_vars=linear_vars)

            if repn.quadratic_vars:
                expr += pyo.quicksum(
                    (coef*x*y for coef,(x,y) in zip(quadratic_coefs, repn.quadratic_vars))
                )

            s._mpisppy_model.EF_obj = pyo.Expression(expr=expr)

            if opt.is_minimizing:
                s._mpisppy_model.EF_Obj = pyo.Objective(expr=s._mpisppy_model.EF_obj, sense=pyo.minimize)
            else:
                s._mpisppy_model.EF_Obj = pyo.Objective(expr=-s._mpisppy_model.EF_obj, sense=pyo.maximize)
            s._mpisppy_model.EF_Obj.deactivate()

            # add cut constraint dicts
            s._mpisppy_model.benders_cuts = pyo.Constraint(pyo.Any)
            s._mpisppy_model.inner_bound_constr = pyo.Constraint(pyo.Any)

        self._enable_W_and_prox()

        # try to get the initial eta LB cuts
        # (may not be available)
        self.get_from_cross_cuts()

    def miditer(self):
        self.iter_since_last_check += 1

        ib = self.opt.spcomm.BestInnerBound
        if ib != self.cur_ib:
            self.cur_ib = ib
            self.iter_at_cur_ib = 1
        elif self.cur_ib is not None and math.isfinite(self.cur_ib):
            self.iter_at_cur_ib += 1

        ob = self.opt.spcomm.BestOuterBound
        if self.cur_ob is not None and math.isclose(ob, self.cur_ob):
            ob_new = False
        else:
            self.cur_ob = ob
            ob_new = True
        
        if not self.any_cuts:
            if self.new_cuts:
                self.any_cuts = True

        ## if its the second time or more with this IB, we'll only check
        ## if the last improved the OB, or if the OB is new itself (from somewhere else)
        check = (self.check_bound_iterations is not None) and self.any_cuts and ( \
                (self.iter_at_cur_ib == self.check_bound_iterations) or \
                (self.iter_at_cur_ib > self.check_bound_iterations and ob_new) or \
                ((self.iter_since_last_check%self.check_bound_iterations == 0) and self.new_cuts))
                # if there hasn't been OB movement, check every so often if we have new cuts
        if check:
            self._check_bound()
            self.new_cuts = False
            self.iter_since_last_check = 0

    def enditer(self):
        pass

    def post_everything(self):
        pass
