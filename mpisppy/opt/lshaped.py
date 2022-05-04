# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import numpy as np
import itertools
import time
import sys
import mpisppy.spbase as spbase

from mpisppy import MPI
from pyomo.core.plugins.transform.discrete_vars import RelaxIntegerVars
from mpisppy.utils.sputils import find_active_objective
from mpisppy.utils.lshaped_cuts import LShapedCutGenerator
from mpisppy.spopt import set_instance_retry
from pyomo.core import (
    Objective, SOSConstraint, Constraint, Var
)
from pyomo.core.expr.visitor import identify_variables
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import LinearExpression

class LShapedMethod(spbase.SPBase):
    """ Base class for the L-shaped method for two-stage stochastic programs.

    Warning:
        This class explicitly assumes minimization.

    Args:
        options (dict):
            Dictionary of options. Possible (optional) options include

            - root_scenarios (list) - List of scenario names to include as
              part of the root problem (default [])
            - store_subproblems (boolean) - If True, the BendersDecomp object
              will maintain a dictionary containing the subproblems created by
              the BendersCutGenerator.
            - relax_root (boolean) - If True, the LP relaxation of the root
              problem is solved (i.e. integer variables in the root problem
              are relaxed).
            - scenario_creator_kwargs (dict) - Keyword args to pass to the scenario_creator.
            - valid_eta_lb (dict) - Dictionary mapping scenario names to valid
              lower bounds for the eta variables--i.e., a valid lower (outer)
              bound on the optimal objective value for each scenario. If none
              are provided, the lower bound is set to -sys.maxsize *
              scenario_prob, which may cause numerical errors.
            - indx_to_stage (dict) - Dictionary mapping the index of every
              variable in the model to the stage they belong to.
        all_scenario_names (list):
            List of all scenarios names present in the model (strings).
        scenario_creator (callable): 
            Function which take a scenario name (string) and returns a
            Pyomo Concrete model with some things attached.
        scenario_denouement (callable, optional):
            Function which does post-processing and reporting.
        all_nodenames (list, optional): 
            List of all node name (strings). Can be `None` for two-stage
            problems.
        mpicomm (MPI comm, optional):
            MPI communicator to use between all scenarios. Default is
            `MPI.COMM_WORLD`.
        scenario_creator_kwargs (dict, optional): 
            Keyword arguments to pass to `scenario_creator`.
    """
    def __init__(
        self, 
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement=None,
        all_nodenames=None,
        mpicomm=None,
        scenario_creator_kwargs=None,
    ):
        super().__init__(
            options,
            all_scenario_names,
            scenario_creator,
            scenario_denouement=scenario_denouement,
            all_nodenames=all_nodenames,
            mpicomm=mpicomm,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )
        if self.multistage:
            raise Exception("LShaped does not currently support multiple stages")
        self.options = options
        self.options_check()
        self.all_scenario_names = all_scenario_names
        self.root = None
        self.root_vars = None
        self.scenario_count = len(all_scenario_names)

        self.store_subproblems = False
        if "store_subproblems" in options:
            self.store_subproblems = options["store_subproblems"]

        self.root_scenarios = None
        if "root_scenarios" in options:
            self.root_scenarios = options["root_scenarios"]

        self.relax_root = False 
        if "relax_root" in options:
            self.relax_root = options["relax_root"]

        self.valid_eta_lb = None
        if "valid_eta_lb" in options:
            self.valid_eta_lb = options["valid_eta_lb"]
            self.compute_eta_bound = False
        else: # fit the user does not provide a bound, compute one
            self.valid_eta_lb = { scen :  (-sys.maxsize - 1) * 1. / len(self.all_scenario_names) \
                                    for scen in self.all_scenario_names }
            self.compute_eta_bound = True

        if scenario_creator_kwargs is None:
            self.scenario_creator_kwargs = dict()
        else:
            self.scenario_creator_kwargs = scenario_creator_kwargs
        self.indx_to_stage = None
        self.has_valid_eta_lb = self.valid_eta_lb is not None
        self.has_root_scens = self.root_scenarios is not None

        if self.store_subproblems:
            self.subproblems = dict.fromkeys(scenario_names)

    def options_check(self):
        """ Check to ensure that the user-specified options are valid. Requried
        options are:

        - root_solver (string) - Solver to use for the root problem.
        - sp_solver (string) - Solver to use for the subproblems.
        """
        required = ["root_solver", "sp_solver"]
        if "root_solver_options" not in self.options:
            self.options["root_solver_options"] = dict()
        if "sp_solver_options" not in self.options:
            self.options["sp_solver_options"] = dict()
        self._options_check(required, self.options)

    def _add_root_etas(self, root, index):
        def _eta_bounds(m, s):
            return (self.valid_eta_lb[s],None)
        root.eta = pyo.Var(index, within=pyo.Reals, bounds=_eta_bounds)

    def _create_root_no_scenarios(self):

        # using the first scenario as a basis
        root = self.scenario_creator(
            self.all_scenario_names[0], **self.scenario_creator_kwargs
        )

        if self.relax_root:
            RelaxIntegerVars().apply_to(root)

        nonant_list, nonant_ids = _get_nonant_ids(root)

        self.root_vars = nonant_list

        for constr_data in list(itertools.chain(
                root.component_data_objects(SOSConstraint, active=True, descend_into=True)
                , root.component_data_objects(Constraint, active=True, descend_into=True))):
            if not _first_stage_only(constr_data, nonant_ids):
                _del_con(constr_data)

        # delete the second stage variables
        for var in list(root.component_data_objects(Var, active=True, descend_into=True)):
            if id(var) not in nonant_ids:
                _del_var(var)

        self._add_root_etas(root, self.all_scenario_names)

        # pulls the current objective expression, adds in the eta variables,
        # and removes the second stage variables from the expression
        obj = find_active_objective(root)

        repn = generate_standard_repn(obj.expr, quadratic=True)
        if len(repn.nonlinear_vars) > 0:
            raise ValueError("LShaped does not support models with nonlinear objective functions")

        linear_vars = list()
        linear_coefs = list()
        quadratic_vars = list()
        quadratic_coefs = list()
        ## we'll assume the constant is part of stage 1 (wlog it is), just
        ## like the first-stage bits of the objective
        constant = repn.constant 

        ## only keep the first stage variables in the objective
        for coef, var in zip(repn.linear_coefs, repn.linear_vars):
            id_var = id(var)
            if id_var in nonant_ids:
                linear_vars.append(var)
                linear_coefs.append(coef)
        for coef, (x,y) in zip(repn.quadratic_coefs, repn.quadratic_vars):
            id_x = id(x)
            id_y = id(y)
            if id_x in nonant_ids and id_y in nonant_ids:
                quadratic_coefs.append(coef)
                quadratic_vars.append((x,y))

        # checks if model sense is max, if so negates the objective
        if not self.is_minimizing:
            for i,coef in enumerate(linear_coefs):
                linear_coefs[i] = -coef
            for i,coef in enumerate(quadratic_coefs):
                quadratic_coefs[i] = -coef

        # add the etas
        for var in root.eta.values():
            linear_vars.append(var)
            linear_coefs.append(1)

        expr = LinearExpression(constant=constant, linear_coefs=linear_coefs,
                                linear_vars=linear_vars)
        if quadratic_coefs:
            expr += pyo.quicksum(
                        (coef*x*y for coef,(x,y) in zip(quadratic_coefs, quadratic_vars))
                    )

        root.del_component(obj)

        # set root objective function
        root.obj = pyo.Objective(expr=expr, sense=pyo.minimize)

        self.root = root

    def _create_root_with_scenarios(self):

        ef_scenarios = self.root_scenarios

        ## we want the correct probabilities to be set when
        ## calling create_EF
        if len(ef_scenarios) > 1:
            def scenario_creator_wrapper(name, **creator_options):
                scenario = self.scenario_creator(name, **creator_options)
                if not hasattr(scenario, '_mpisppy_probability'):
                    scenario._mpisppy_probability = 1./len(self.all_scenario_names)
                return scenario
            root = sputils.create_EF(
                ef_scenarios,
                scenario_creator_wrapper,
                scenario_creator_kwargs=self.scenario_creator_kwargs,
            )

            nonant_list, nonant_ids = _get_nonant_ids_EF(root)
        else:
            root = self.scenario_creator(
                ef_scenarios[0],
                **self.scenario_creator_kwargs,
            )
            if not hasattr(root, '_mpisppy_probability'):
                root._mpisppy_probability = 1./len(self.all_scenario_names)

            nonant_list, nonant_ids = _get_nonant_ids(root)

        self.root_vars = nonant_list

        # creates the eta variables for scenarios that are NOT selected to be
        # included in the root problem
        eta_indx = [scenario_name for scenario_name in self.all_scenario_names
                        if scenario_name not in self.root_scenarios]
        self._add_root_etas(root, eta_indx)

        obj = find_active_objective(root)

        repn = generate_standard_repn(obj.expr, quadratic=True)
        if len(repn.nonlinear_vars) > 0:
            raise ValueError("LShaped does not support models with nonlinear objective functions")
        linear_vars = list(repn.linear_vars)
        linear_coefs = list(repn.linear_coefs)
        quadratic_coefs = list(repn.quadratic_coefs)

        # adjust coefficients by scenario/bundle probability
        scen_prob = root._mpisppy_probability
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
        if not self.is_minimizing:
            for i,coef in enumerate(linear_coefs):
                linear_coefs[i] = -coef
            for i,coef in enumerate(quadratic_coefs):
                quadratic_coefs[i] = -coef

        # add the etas
        for var in root.eta.values():
            linear_vars.append(var)
            linear_coefs.append(1)

        expr = LinearExpression(constant=repn.constant, linear_coefs=linear_coefs,
                                linear_vars=linear_vars)
        if repn.quadratic_vars:
            expr += pyo.quicksum(
                (coef*x*y for coef,(x,y) in zip(quadratic_coefs, repn.quadratic_vars))
            )

        root.del_component(obj)

        # set root objective function
        root.obj = pyo.Objective(expr=expr, sense=pyo.minimize)

        self.root = root

    def _create_shadow_root(self):

        root = pyo.ConcreteModel()

        arb_scen = self.local_scenarios[self.local_scenario_names[0]]
        nonants = arb_scen._mpisppy_node_list[0].nonant_vardata_list

        root_vars = list()
        for v in nonants:
            nonant_shadow = pyo.Var(name=v.name)
            root.add_component(v.name, nonant_shadow)
            root_vars.append(nonant_shadow)
        
        if self.has_root_scens:
            eta_indx = [scenario_name for scenario_name in self.all_scenario_names
                            if scenario_name not in self.root_scenarios]
        else:
            eta_indx = self.all_scenario_names
        self._add_root_etas(root, eta_indx)

        root.obj = None
        self.root = root
        self.root_vars = root_vars

    def set_eta_bounds(self):
        if self.compute_eta_bound:
            ## for scenarios not in self.local_scenarios, these will be a large negative number
            this_etas_lb = np.fromiter((self.valid_eta_lb[scen] for scen in self.all_scenario_names),
                                    float, count=len(self.all_scenario_names))

            all_etas_lb = np.empty_like(this_etas_lb)

            self.mpicomm.Allreduce(this_etas_lb, all_etas_lb, op=MPI.MAX)

            for idx, s in enumerate(self.all_scenario_names):
                self.valid_eta_lb[s] = all_etas_lb[idx]
            
            # root may not have etas for every scenarios
            for s, v in self.root.eta.items():
                v.setlb(self.valid_eta_lb[s])

    def create_root(self):
        """ creates a ConcreteModel from one of the problem scenarios then
            modifies the model to serve as the root problem 
        """
        if self.cylinder_rank == 0:
            if self.has_root_scens:
                self._create_root_with_scenarios()
            else:
                self._create_root_no_scenarios()
        else: 
            ## if we're not rank0, just create a root to
            ## hold the nonants and etas; rank0 will do 
            ## the optimizing
            self._create_shadow_root()
        
    def attach_nonant_var_map(self, scenario_name):
        instance = self.local_scenarios[scenario_name]

        subproblem_to_root_vars_map = pyo.ComponentMap()
        for var, rvar in zip(instance._mpisppy_data.nonant_indices.values(), self.root_vars):
            if var.name not in rvar.name:
                raise Exception("Error: Complicating variable mismatch, sub-problem variables changed order")
            subproblem_to_root_vars_map[var] = rvar 

        # this is for interefacing with PH code
        instance._mpisppy_model.subproblem_to_root_vars_map = subproblem_to_root_vars_map

    def create_subproblem(self, scenario_name):
        """ the subproblem creation function passed into the
            BendersCutsGenerator 
        """
        instance = self.local_scenarios[scenario_name]

        nonant_list, nonant_ids = _get_nonant_ids(instance) 

        # NOTE: since we use generate_standard_repn below, we need
        #       to unfix any nonants so they'll properly appear
        #       in the objective
        fixed_nonants = [ var for var in nonant_list if var.fixed ]
        for var in fixed_nonants:
            var.fixed = False

        # pulls the scenario objective expression, removes the first stage variables, and sets the new objective
        obj = find_active_objective(instance)

        if not hasattr(instance, "_mpisppy_probability"):
            instance._mpisppy_probability = 1. / self.scenario_count
        _mpisppy_probability = instance._mpisppy_probability

        repn = generate_standard_repn(obj.expr, quadratic=True)
        if len(repn.nonlinear_vars) > 0:
            raise ValueError("LShaped does not support models with nonlinear objective functions")

        linear_vars = list()
        linear_coefs = list()
        quadratic_vars = list()
        quadratic_coefs = list()
        ## we'll assume the constant is part of stage 1 (wlog it is), just
        ## like the first-stage bits of the objective
        constant = repn.constant 

        ## only keep the second stage variables in the objective
        for coef, var in zip(repn.linear_coefs, repn.linear_vars):
            id_var = id(var)
            if id_var not in nonant_ids:
                linear_vars.append(var)
                linear_coefs.append(_mpisppy_probability*coef)
        for coef, (x,y) in zip(repn.quadratic_coefs, repn.quadratic_vars):
            id_x = id(x)
            id_y = id(y)
            if id_x not in nonant_ids or id_y not in nonant_ids:
                quadratic_coefs.append(_mpisppy_probability*coef)
                quadratic_vars.append((x,y))

        # checks if model sense is max, if so negates the objective
        if not self.is_minimizing:
            for i,coef in enumerate(linear_coefs):
                linear_coefs[i] = -coef
            for i,coef in enumerate(quadratic_coefs):
                quadratic_coefs[i] = -coef

        expr = LinearExpression(constant=constant, linear_coefs=linear_coefs,
                                linear_vars=linear_vars)
        if quadratic_coefs:
            expr += pyo.quicksum(
                        (coef*x*y for coef,(x,y) in zip(quadratic_coefs, quadratic_vars))
                    )

        instance.del_component(obj)

        # set subproblem objective function
        instance.obj = pyo.Objective(expr=expr, sense=pyo.minimize)

        ## need to do this here for validity if computing the eta bound
        if self.relax_root:
            # relaxes any integrality constraints for the subproblem
            RelaxIntegerVars().apply_to(instance)

        if self.compute_eta_bound:
            for var in fixed_nonants:
                var.fixed = True
            opt = pyo.SolverFactory(self.options["sp_solver"])
            if self.options["sp_solver_options"]:
                for k,v in self.options["sp_solver_options"].items():
                    opt.options[k] = v

            if sputils.is_persistent(opt):
                set_instance_retry(instance, opt, scenario_name)
                res = opt.solve(tee=False)
            else:
                res = opt.solve(instance, tee=False)

            eta_lb = res.Problem[0].Lower_bound

            self.valid_eta_lb[scenario_name] = eta_lb

        # if not done above
        if not self.relax_root:
            # relaxes any integrality constraints for the subproblem
            RelaxIntegerVars().apply_to(instance)

        # iterates through constraints and removes first stage constraints from the model
        # the id dict is used to improve the speed of identifying the stage each variables belongs to
        for constr_data in list(itertools.chain(
                instance.component_data_objects(SOSConstraint, active=True, descend_into=True)
                , instance.component_data_objects(Constraint, active=True, descend_into=True))):
            if _first_stage_only(constr_data, nonant_ids):
                _del_con(constr_data)

        # creates the sub map to remove first stage variables from objective expression
        complicating_vars_map = pyo.ComponentMap()
        subproblem_to_root_vars_map = pyo.ComponentMap()

        # creates the complicating var map that connects the first stage variables in the sub problem to those in
        # the root problem -- also set the bounds on the subproblem root vars to be none for better cuts
        for var, rvar in zip(nonant_list, self.root_vars):
            if var.name not in rvar.name: # rvar.name may be part of a bundle
                raise Exception("Error: Complicating variable mismatch, sub-problem variables changed order")
            complicating_vars_map[rvar] = var
            subproblem_to_root_vars_map[var] = rvar 

            # these are already enforced in the root
            # don't need to be enfored in the subproblems
            var.setlb(None)
            var.setub(None)
            var.fixed = False

        # this is for interefacing with PH code
        instance._mpisppy_model.subproblem_to_root_vars_map = subproblem_to_root_vars_map

        if self.store_subproblems:
            self.subproblems[scenario_name] = instance

        return instance, complicating_vars_map

    def lshaped_algorithm(self, converger=None):
        """ function that runs the lshaped.py algorithm
        """
        if converger:
            converger = converger(self, self.cylinder_rank, self.n_proc)
        max_iter = 30
        if "max_iter" in self.options:
            max_iter = self.options["max_iter"]
        tol = 1e-8
        if "tol" in self.options:
            tol = self.options["tol"]
        verbose = True
        if "verbose" in self.options:
            verbose = self.options["verbose"]
        root_solver = self.options["root_solver"]
        sp_solver = self.options["sp_solver"]

        # creates the root problem
        self.create_root()
        m = self.root
        assert hasattr(m, "obj")

        # prevents problems from first stage variables becoming unconstrained
        # after processing
        _init_vars(self.root_vars)

        # sets up the BendersCutGenerator object
        m.bender = LShapedCutGenerator()

        m.bender.set_input(root_vars=self.root_vars, tol=tol, comm=self.mpicomm)

        # let the cut generator know who's using it, probably should check that this is called after set input
        m.bender.set_ls(self)

        # set the eta variables, removing this from the add_suproblem function so we can

        # Pass all the scenarios in the problem to bender.add_subproblem
        # and let it internally handle which ranks get which scenarios
        if self.has_root_scens:
            sub_scenarios = [
                scenario_name for scenario_name in self.local_scenario_names
                if scenario_name not in self.root_scenarios
            ]
        else:
            sub_scenarios = self.local_scenario_names
        for scenario_name in self.local_scenario_names:
            if scenario_name in sub_scenarios:
                subproblem_fn_kwargs = dict()
                subproblem_fn_kwargs['scenario_name'] = scenario_name
                m.bender.add_subproblem(
                    subproblem_fn=self.create_subproblem,
                    subproblem_fn_kwargs=subproblem_fn_kwargs,
                    root_eta=m.eta[scenario_name],
                    subproblem_solver=sp_solver,
                    subproblem_solver_options=self.options["sp_solver_options"]
                )
            else:
                self.attach_nonant_var_map(scenario_name)

        # set the eta bounds if computed
        # by self.create_subproblem
        self.set_eta_bounds()

        if self.cylinder_rank == 0:
            opt = pyo.SolverFactory(root_solver)
            if opt is None:
                raise Exception("Error: Failed to Create Master Solver")

            # set options
            for k,v in self.options["root_solver_options"].items():
                opt.options[k] = v

            is_persistent = sputils.is_persistent(opt)
            if is_persistent:
                set_instance_retry(m, opt, "root")

        t = time.time()
        res, t1, t2 = None, None, None

        # benders solve loop, repeats the benders root - subproblem
        # loop until either a no more cuts can are generated
        # or the maximum iterations limit is reached
        for self.iter in range(max_iter):
            if verbose and self.cylinder_rank == 0:
                if self.iter > 0:
                    print("Current Iteration:", self.iter + 1, "Time Elapsed:", "%7.2f" % (time.time() - t), "Time Spent on Last Master:", "%7.2f" % t1,
                          "Time Spent Generating Last Cut Set:", "%7.2f" % t2, "Current Objective:", "%7.2f" % m.obj.expr())
                else:
                    print("Current Iteration:", self.iter + 1, "Time Elapsed:", "%7.2f" % (time.time() - t), "Current Objective: -Inf")
            t1 = time.time()
            x_vals = np.zeros(len(self.root_vars))
            eta_vals = np.zeros(self.scenario_count)
            outer_bound = np.zeros(1)
            if self.cylinder_rank == 0:
                if is_persistent:
                    res = opt.solve(tee=False)
                else:
                    res = opt.solve(m, tee=False)
                # LShaped is always minimizing
                outer_bound[0] = res.Problem[0].Lower_bound
                for i, var in enumerate(self.root_vars):
                    x_vals[i] = var.value
                for i, eta in enumerate(m.eta.values()):
                    eta_vals[i] = eta.value

            self.mpicomm.Bcast(x_vals, root=0)
            self.mpicomm.Bcast(eta_vals, root=0)
            self.mpicomm.Bcast(outer_bound, root=0)

            if self.is_minimizing:
                self._LShaped_bound = outer_bound[0]
            else:
                # LShaped is always minimizing, so negate
                # the outer bound for sharing broadly
                self._LShaped_bound = -outer_bound[0]

            if self.cylinder_rank != 0:
                for i, var in enumerate(self.root_vars):
                    var._value = x_vals[i]
                for i, eta in enumerate(m.eta.values()):
                    eta._value = eta_vals[i]
            t1 = time.time() - t1

            # The hub object takes precedence over the converger
            # We'll send the nonants now, and check for a for
            # convergence
            if self.spcomm:
                self.spcomm.sync(send_nonants=True)
                if self.spcomm.is_converged():
                    break

            t2 = time.time()
            cuts_added = m.bender.generate_cut()
            t2 = time.time() - t2
            if self.cylinder_rank == 0:
                for c in cuts_added:
                    if is_persistent:
                        opt.add_constraint(c)
                if verbose and len(cuts_added) == 0:
                    print(
                        f"Converged in {self.iter+1} iterations.\n"
                        f"Total Time Elapsed: {time.time()-t:7.2f} "
                        f"Time Spent on Last Master: {t1:7.2f} "
                        f"Time spent verifying second stage: {t2:7.2f} "
                        f"Final Objective: {m.obj.expr():7.2f}"
                    )
                    self.first_stage_solution_available = True
                    self.tree_solution_available = True
                    break
                if verbose and self.iter == max_iter - 1:
                    print("WARNING MAX ITERATION LIMIT REACHED !!! ")
            else:
                if len(cuts_added) == 0:
                    break
            # The hub object takes precedence over the converger
            if self.spcomm:
                self.spcomm.sync(send_nonants=False)
                if self.spcomm.is_converged():
                    break
            if converger:
                converger.convergence_value()
                if converger.is_converged():
                    if verbose and self.cylinder_rank == 0:
                        print(
                            f"Converged to user criteria in {self.iter+1} iterations.\n"
                            f"Total Time Elapsed: {time.time()-t:7.2f} "
                            f"Time Spent on Last Master: {t1:7.2f} "
                            f"Time spent verifying second stage: {t2:7.2f} "
                            f"Final Objective: {m.obj.expr():7.2f}"
                        )
                    break
        return res

def _del_con(c):
    parent = c.parent_component()
    if parent.is_indexed():
        parent.__delitem__(c.index())
    else:
        assert parent is c
        c.parent_block().del_component(c)

def _del_var(v):
    parent = v.parent_component()
    if parent.is_indexed():
        parent.__delitem__(v.index())
    else:
        assert parent is v
        block = v.parent_block()
        block.del_component(v)

def _get_nonant_ids(instance):
    assert len(instance._mpisppy_node_list) == 1
    # set comprehension
    nonant_list = instance._mpisppy_node_list[0].nonant_vardata_list
    return nonant_list, { id(var) for var in nonant_list }

def _get_nonant_ids_EF(instance):
    assert len(instance._mpisppy_data.nlens) == 1

    ndn, nlen = list(instance._mpisppy_data.nlens.items())[0]

    ## this is for the cut variables, so we just need (and want)
    ## exactly one set of them
    nonant_list = list(instance.ref_vars[ndn,i] for i in range(nlen))

    ## this is for adjusting the objective, so needs all the nonants
    ## in the EF
    snames = instance._ef_scenario_names

    nonant_ids = set()
    for s in snames:
        nonant_ids.update( (id(v) for v in \
                getattr(instance, s)._mpisppy_node_list[0].nonant_vardata_list)
                )
    return nonant_list, nonant_ids 

def _first_stage_only(constr_data, nonant_ids):
    """ iterates through the constraint in a scenario and returns if it only
        has first stage variables
    """
    for var in identify_variables(constr_data.body):
        if id(var) not in nonant_ids: 
            return False
    return True

def _init_vars(varlist):
    '''
    for every pyomo var in varlist without a value,
    sets it to the lower bound (if it exists), or
    the upper bound (if it exists, and the lower bound
    does note) or 0 (if neither bound exists).
    '''
    value = pyo.value
    for var in varlist:
        if var.value is not None:
            continue
        if var.lb is not None:
            var.set_value(value(var.lb))
        elif var.ub is not None:
            var.set_value(value(var.ub))
        else:
            var.set_value(0)



def main():
    import mpisppy.tests.examples.farmer as ref
    import os
    # Turn off output from all ranks except rank 1
    if MPI.COMM_WORLD.Get_rank() != 0:
        sys.stdout = open(os.devnull, 'w')
    scenario_names = ['scen' + str(i) for i in range(3)]
    bounds = {i:-432000 for i in scenario_names}
    options = {
        "root_solver": "gurobi_persistent",
        "sp_solver": "gurobi_persistent",
        "sp_solver_options" : {"threads" : 1},
        "valid_eta_lb": bounds,
        "max_iter": 10,
   }

    ls = LShapedMethod(options, scenario_names, ref.scenario_creator)
    res = ls.lshaped_algorithm()
    if ls.cylinder_rank == 0:
        print(res)


if __name__ == '__main__':
    main()
