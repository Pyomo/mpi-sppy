# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
from mpisppy.extensions.extension import PHExtension
from pyomo.pysp.phutils import find_active_objective
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import LinearExpression
from mpisppy import tt_timer

import pyomo.environ as pyo
import sys
import math
import mpisppy.utils.sputils as sputils
import numpy as np
import mpi4py.MPI as mpi

class CrossScenarioExtension(PHExtension):
    def __init__(self, spbase_object):
        super().__init__(spbase_object)

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
            opt._disable_W_and_prox()
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
            opt._reenable_W_and_prox()
        elif self.reenable_W:
            opt._reenable_W()
        elif self.reenable_prox:
            opt._reenable_prox()

        self.reenable_W = None
        self.reenable_prox = None

    def _check_bound(self):
        opt = self.opt

        chached_ph_obj = dict()

        for k,s in opt.local_subproblems.items():
            phobj = find_active_objective(s,True)
            phobj.deactivate()
            chached_ph_obj[k] = phobj
            s._EF_Obj.activate()

        teeme = (
            "tee-rank0-solves" in opt.PHoptions
             and opt.PHoptions["tee-rank0-solves"]
        )
        opt.solve_loop(
                solver_options=opt.current_solver_options,
                dtiming=opt.PHoptions["display_timing"],
                gripe=True,
                disable_pyomo_signal_handling=False,
                tee=teeme,
                verbose=opt.PHoptions["verbose"],
        )

        local_obs = np.fromiter((s._PySP_ob for s in opt.local_subproblems.values()),
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
            s._EF_Obj.deactivate()
            chached_ph_obj[k].activate()

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
            _eta_bounds = lambda m,k : (_eta_init[k], None)
        else:
            lb = (-sys.maxsize - 1) * 1. / len(opt.all_scenario_names)
            _eta_init = lambda m,k : lb
            _eta_bounds = lambda m,k : (lb, None)

        # eta is attached to each subproblem, regardless of bundles
        bundling = opt.bundling
        for k,s in opt.local_subproblems.items():
            s.eta = pyo.Var(opt.all_scenario_names, initialize=_eta_init, bounds=_eta_bounds)
            if sputils.is_persistent(s._solver_plugin):
                for var in s.eta.values():
                    s._solver_plugin.add_var(var)
            if bundling: ## create a refence to eta on each subproblem
                for sn in s.scen_list:
                    scenario = opt.local_scenarios[sn]
                    scenario.eta = { k : s.eta[k] for k in opt.all_scenario_names }

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
                            opt.local_scenarios[sn]._PySPnode_list[0].nonant_vardata_list)
            else:
                nonant_vardata_list = s._PySPnode_list[0].nonant_vardata_list

            nonant_ids = set((id(var) for var in nonant_vardata_list))

            linear_coefs = list(repn.linear_coefs)
            linear_vars = list(repn.linear_vars)

            quadratic_coefs = list(repn.quadratic_coefs)

            # adjust coefficients by scenario/bundle probability
            scen_prob = s.PySP_prob
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
                    linear_vars.append(s.eta[sn])
                    eta_scenarios.append(sn)

            expr = LinearExpression(constant=repn.constant, linear_coefs=linear_coefs,
                                    linear_vars=linear_vars)

            if repn.quadratic_vars:
                expr += pyo.quicksum(
                    (coef*x*y for coef,(x,y) in zip(quadratic_coefs, repn.quadratic_vars))
                )

            s._EF_obj = pyo.Expression(expr=expr)

            if opt.is_minimizing:
                s._EF_Obj = pyo.Objective(expr=s._EF_obj, sense=pyo.minimize)
            else:
                s._EF_Obj = pyo.Objective(expr=-s._EF_obj, sense=pyo.maximize)
            s._EF_Obj.deactivate()

            # add cut constraint dicts
            s._benders_cuts = pyo.Constraint(pyo.Any)
            s._ib_constr = pyo.Constraint(pyo.Any)

        self._enable_W_and_prox()

        # try to get the initial eta LB cuts
        # (may not be available)
        opt.spcomm.get_from_cross_cuts()

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
            if self.opt.spcomm.new_cuts:
                self.any_cuts = True

        ## if its the second time or more with this IB, we'll only check
        ## if the last improved the OB, or if the OB is new itself (from somewhere else)
        check = (self.check_bound_iterations is not None) and self.any_cuts and ( \
                (self.iter_at_cur_ib == self.check_bound_iterations) or \
                (self.iter_at_cur_ib > self.check_bound_iterations and ob_new) or \
                ((self.iter_since_last_check%self.check_bound_iterations == 0) and self.opt.spcomm.new_cuts))
                # if there hasn't been OB movement, check every so often if we have new cuts
        if check:
            if self.opt.spcomm.rank_global == 0:
                tt_timer.toc(f"Attempting to update Best Bound with CrossScenarioExtension", delta=False)
            self._check_bound()
            self.opt.spcomm.new_cuts = False
            self.iter_since_last_check = 0

    def enditer(self):
        pass

    def post_everything(self):
        pass
