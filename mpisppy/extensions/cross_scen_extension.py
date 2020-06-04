# This software is distributed under the 3-clause BSD License.
from mpisppy.extensions.extension import PHExtension
from pyomo.pysp.phutils import find_active_objective
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import LinearExpression

import pyomo.environ as pyo
import sys
import mpisppy.utils.sputils as sputils

class CrossScenarioExtension(PHExtension):
    def __init__(self, spbase_object):
        super().__init__(spbase_object)

    def pre_iter0(self):
        if self.opt.multistage:
            raise RuntimeError("CrossScenarioExtension does not support "
                               "multi-stage problems at this time")

    def post_iter0(self):
        opt = self.opt
        # NOTE: the LShaped code negates the objective, so
        #       we do the same here for consistency
        if 'valid_eta_bound' in opt.options['cross_scen_options']:
            valid_eta_bound = opt.options['cross_scen_options']['valid_eta_bound']
            if not opt.is_minimizing:
                _eta_init = { k: -v for k,v in valid_eta_bound.items() }
            else:
                _eta_init = valid_eta_bound
            _eta_bounds = lambda m,k : (_eta_init[k], None)
        else:
            _eta_init = lambda m,k : -sys.maxsize
            _eta_bounds = lambda m,k : (-sys.maxsize, None)

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
        reenable_W = False
        reenable_prox = False
        if not opt.W_disabled and not opt.prox_disabled:
            opt._disable_W_and_prox()
            reenable_W = True
            reenable_prox = True 
        elif not opt.W_disabled:
            opt._disable_W()
            reenable_W = True
        elif not opt.prox_disabled:
            opt._disable_prox()
            reenable_prox = True 
        
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
                if id(x) not in nonant_ids:
                    quadratic_coefs[i] *= scen_prob
                if id(y) not in nonant_ids:
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
            for sn in opt.all_scenario_names:
                if sn not in these_scenarios:
                    linear_coefs.append(1)
                    linear_vars.append(s.eta[sn])

            expr = LinearExpression(constant=repn.constant, linear_coefs=linear_coefs,
                                    linear_vars=linear_vars)

            if repn.quadratic_vars:
                expr += pyo.quicksum(
                    (coef*x*y for coef,(x,y) in zip(quadratic_coefs, repn.quadratic_vars))
                )

            s._EF_obj = pyo.Expression(expr=expr)

            # add cut constraint dicts
            s._benders_cuts = pyo.Constraint(pyo.Any)
            s._ib_constr = pyo.Constraint(pyo.Any)

        if reenable_W and reenable_prox:
            opt._reenable_W_and_prox()
        elif reenable_W:
            opt._reenable_W()
        elif reenable_prox:
            opt._reenable_prox()

    def miditer(self):
        pass

    def enditer(self):
        pass

    def post_everything(self):
        pass
