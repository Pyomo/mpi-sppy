# This software is distributed under the 3-clause BSD License.
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import numpy as np
import itertools
import time
import sys
import mpisppy.spbase as spbase

from mpi4py import MPI
# from pyomo.core.plugins.transform.discrete_vars import RelaxIntegerVars
from pyomo.core.plugins.transform.relax_integrality import RelaxIntegrality
from mpisppy.utils.lshaped_cuts import LShapedCutGenerator
from pyomo.core import (
    Objective, SOSConstraint, Constraint, Var
)
from pyomo.core.expr.visitor import identify_variables, replace_expressions


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


def _seperate_variables(instance):
    """ Pulls the non_ant variable list from the PySPnode_list, identifies
        these as first stage variables and identifies the remaining as second
        stage variables
    """
    var_dict = dict()
    var_dict['stage1'] = list()
    var_dict['stage2'] = list()
    stage1_ids = list()
    indx_to_stage = dict()
    indx = 0

    for nonant in instance._PySPnode_list:
        for var in nonant.nonant_list:
            if var.is_indexed():
                for var_data in var.values():
                    var_dict['stage1'].append(var_data)
                    stage1_ids.append(id(var_data))
            else:
                var_dict['stage1'].append(var)
                stage1_ids.append(id(var))

    for var in instance.component_objects(Var, active=True, descend_into=True):
        if var.is_indexed():
            for var_data in var.values():
                if id(var_data) not in stage1_ids:
                    var_dict['stage2'].append(var_data)
                    indx_to_stage[indx] = 2
                    indx += 1
                else:
                    indx_to_stage[indx] = 1
                    indx += 1
        else:
            if id(var) not in stage1_ids:
                indx_to_stage[indx] = 2
                indx += 1
                var_dict['stage2'].append(var)
            else:
                indx_to_stage[indx] = 1
                indx += 1

    return var_dict, indx_to_stage


def _get_stage(constr_data, id_dict):
    """ iterates through the constraints in a scenario and labels them as first
        or second stage based on the highest stage variable in the constraint
    """
    has_stage_1 = False
    has_stage_2 = False
    for var in identify_variables(constr_data.body):
        if id(var) in id_dict['stage2']:
            has_stage_2 = True
            break
        elif id(var) in id_dict['stage1']:
            has_stage_1 = True
        else:
            print("Variable unaccounted for?")
            exit(-1)
    if has_stage_2:
        return 1
    elif has_stage_1:
        return 0
    else:
        print("should never reach this point...")
        exit(-1)


class LShapedMethod(spbase.SPBase):
    def __init__(
        self, 
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement=None,
        all_nodenames=None,
        mpicomm=None,
        rank0=0,
        cb_data=None,
    ):
        """
        REQUIRED
        --------
        scenario_creator: a pysp callback that takes a scenario name and returns the deterministic model
        all_scenario_names: a list of scenario names

        OPTIONAL (can be passed in the options dict)
        ---------
        master_scenarios: a list of scenario names to include as part of the master problem
                          (defaults to an empty list)
        store_subproblems: True/False the BendersDecomp object will maintain a dict containing the subproblems created
                           by the BendersCutGenerator
        relax_master: True/False use the linear relaxation of the master problem
        cb_data: Callback data for the scenario creator
        valid_eta_lb: a dict mapping scenarios to valid lower bound for the eta variables (if none provided sets
                      lower bound to -sys.maxsize * scenario_prob, in some cases this can cause numerical problems)
        indx_to_stage: a dict mapping the index of every variables in the model (determined by the order they are
                       pulled out using the identify components method) to the stage they belong to
        """
        super().__init__(
            options,
            all_scenario_names,
            scenario_creator,
            scenario_denouement=scenario_denouement,
            all_nodenames=all_nodenames,
            mpicomm=mpicomm,
            rank0=rank0,
            cb_data=cb_data,
        )
        self.options = options
        self.options_check()
        self.all_scenario_names = all_scenario_names
        self.master = None
        self.master_vars = list()
        self.scenario_count = len(all_scenario_names)

        self.store_subproblems = False
        if "store_subproblems" in options:
            self.store_subproblems = options["store_subproblems"]

        self.master_scenarios = None
        if "master_scenarios" in options:
            self.master_scenarios = options["master_scenarios"]

        self.relax_master = False 
        if "relax_master" in options:
            self.relax_master = options["relax_master"]

        self.valid_eta_lb = None
        if "valid_eta_lb" in options:
            self.valid_eta_lb = options["valid_eta_lb"]

        self.cb_data = cb_data
        self.indx_to_stage = None
        self.has_valid_eta_lb = self.valid_eta_lb is not None
        self.has_master_scens = self.master_scenarios is not None

        if self.store_subproblems:
            self.subproblems = dict.fromkeys(scenario_names)

    def options_check(self):
        required = ["master_solver", "sp_solver"]
        if "master_solver_options" not in self.options:
            self.options["master_solver_options"] = dict()
        if "sp_solver_options" not in self.options:
            self.options["sp_solver_options"] = dict()
        self._options_check(required, self.options)

    def create_master(self, scenario_name):
        """ creates a ConcreteModel from one of the problem scenarios then
            modifies the model to serve as the master problem 
        """
        self.master = self.scenario_creator(scenario_name, cb_data=self.cb_data)
        master = self.master
        if self.relax_master:
            RelaxIntegrality().apply_to(master)

        # check if the indx to stage mapping exists
        # if it does use it to create stage dict, if not call _serperate_variables
        if self.indx_to_stage is not None:
            var_dict = split_up_vars_by_stage()
        else:
            var_dict, self.indx_to_stage = _seperate_variables(master)

        # iterates through the variables and stores the first stage variables
        # as master vars and creates a sub map used to remove second stage
        # variables from the objective
        sub_map = dict()
        for var_data in var_dict['stage2']:
            sub_map[id(var_data)] = 0.
        for var_data in var_dict['stage1']:
            self.master_vars.append(var_data)
            sub_map[id(var_data)] = var_data

        # iterates through constraints and removes second stage constraints from the model
        # the id dict is used to improve the speed of identifying the stage each variables belongs to
        id_dict = {'stage1':dict(), 'stage2':dict()}
        for v in var_dict['stage1']:
            id_dict['stage1'][id(v)] = 0.
        for v in var_dict['stage2']:
            id_dict['stage2'][id(v)] = 0.

        for constr_data in list(itertools.chain(
                master.component_data_objects(SOSConstraint, active=True, descend_into=True)
                , master.component_data_objects(Constraint, active=True, descend_into=True))):
            if _get_stage(constr_data, id_dict) > 0:
                _del_con(constr_data)

        # creates the eta variables for scenarios that are NOT selected to be
        # included in the master problem
        if self.has_master_scens:
            eta_indx = [scenario_name for scenario_name in self.all_scenario_names
                        if scenario_name not in self.master_scenarios]
        else:
            eta_indx = self.all_scenario_names

        master.eta = pyo.Var(eta_indx, within=pyo.Reals)
        if self.has_valid_eta_lb:
            for scen, eta in master.eta.items():
                eta.setlb(self.valid_eta_lb[scen])
        else:
            for eta in master.eta.values():
                eta.setlb((-sys.maxsize - 1) * 1. / len(self.all_scenario_names))

        # pulls the current objective expression, adds in the eta variables,
        # and removes the second stage variables from the expression
        objs = list(master.component_data_objects(Objective, descend_into=False, active=True))
        if len(objs) > 1:
            raise Exception("Error: Cannot handle multiple objectives")
        obj_sense = objs[0].sense
        expr = replace_expressions(objs[0], sub_map, remove_named_expressions=False)

        # checks if model sense is max, if so negates the objective
        if obj_sense == pyo.maximize:
            expr = -expr

        expr = expr + sum(master.eta.values())
        master.del_component(objs[0])

        # deletes the second stage variables
        for var in var_dict['stage2']:
            _del_var(var)

        # add selected scenarios to the master problem
        if self.has_master_scens:
            expr = self.add_master_scenarios(master, expr, obj_sense)

        # set master objective function
        master.obj = pyo.Objective(expr=expr, sense=pyo.minimize)
        
    def add_master_scenarios(self, master, expr, obj_sense):
        for scenario_name in self.master_scenarios:
            # Scenarios have not yet been assigned 
            # to ranks so you have to call the 
            # scenario_creator again
            instance = self.scenario_creator(scenario_name, cb_data=self.cb_data)
            RelaxIntegrality().apply_to(instance)
            var_dict = self.split_up_vars_by_stage(instance)


            # create sub map to remove first stage variables from scenario objective function
            sub_map = dict()
            for var_data in var_dict['stage1']:
                sub_map[id(var_data)] = 0.
            for var_data in var_dict['stage2']:
                sub_map[id(var_data)] = var_data

            # pull the scenario objective expression and add it to the master objective expression
            scen_objs = list(instance.component_data_objects(Objective, descend_into=False, active=True))
            scen_expr = replace_expressions(scen_objs[0], sub_map, remove_named_expressions=False)

            if not hasattr(instance, "PySP_prob"):
                instance.PySP_prob = 1. / self.scenario_count

            # checks if model sense is max, if so negates scenario objective expression
            if obj_sense == pyo.maximize:
                scen_expr = -scen_expr

            expr += instance.PySP_prob * scen_expr
            instance.del_component(scen_objs[0])

            # add the scenario model block to the master problem
            master.add_component(scenario_name, instance)
            constr_list = pyo.ConstraintList()
            master.add_component(scenario_name + "NonAntConstr", constr_list)

            # add nonanticipatory constraints
            temp_var_list = list()
            for var_data in var_dict['stage1']:
                temp_var_list.append(var_data)
            for var_data, mvar_data in zip(temp_var_list, self.master_vars):
                constr_list.add(var_data - mvar_data == 0.)
        return expr

    def split_up_vars_by_stage(self, instance):
        var_dict = {"stage1": list(), "stage2": list()}
        indx = 0
        for var in instance.component_objects(Var, active=True, descend_into=True):
            if var.is_indexed():
                for var_data in var.values():
                    stage = self.indx_to_stage[indx]
                    if stage == 1:
                        var_dict['stage1'].append(var_data)
                    elif stage == 2:
                        var_dict['stage2'].append(var_data)
                    else:
                        raise RuntimeError("Variable is neither stage 1 or stage 2")
                    indx += 1
            else:
                stage = self.indx_to_stage[indx]
                if stage == 1:
                    var_dict['stage1'].append(var)
                elif stage == 2:
                    var_dict['stage2'].append(var)
                else:
                    raise RuntimeError("Variable is neither stage 1 or stage 2")
                indx += 1
        return var_dict

    def create_subproblem(self, scenario_name):
        """ the subproblem creation function passed into the
            BendersCutsGenerator 
        """
        instance = self.local_scenarios[scenario_name]
        # relaxes any integrality constraints for the subproblem
        RelaxIntegrality().apply_to(instance)
        # var_dict = _seperate_variables(instance)

        # check if the indx to stage mapping exists
        # if it does use it to create stage dict, if not call _serperate_variables
        if self.indx_to_stage is not None:
            indx_to_stage = self.indx_to_stage
            var_dict = dict()
            var_dict['stage1'] = list()
            var_dict['stage2'] = list()
            indx = 0
            for var in instance.component_objects(Var, active=True, descend_into=True):
                if var.is_indexed():
                    for var_data in var.values():
                        stage = indx_to_stage[indx]
                        if stage == 1:
                            var_dict['stage1'].append(var_data)
                        elif stage == 2:
                            var_dict['stage2'].append(var_data)
                        else:
                            print("should not have reached this point")
                            exit(-1)
                        indx += 1
                else:
                    stage = indx_to_stage[indx]
                    if stage == 1:
                        var_dict['stage1'].append(var)
                    elif stage == 2:
                        var_dict['stage2'].append(var)
                    else:
                        print("should not have reached this point")
                        exit(-1)
                    indx += 1
        else:
            var_dict, self.indx_to_stage = _seperate_variables(instance)

        # iterates through constraints and removes first stage constraints from the model
        # the id dict is used to improve the speed of identifying the stage each variables belongs to
        id_dict = {'stage1':dict(),
                   'stage2':dict()}
        for v in var_dict['stage1']:
            id_dict['stage1'][id(v)] = 0.
        for v in var_dict['stage2']:
            id_dict['stage2'][id(v)] = 0.

        for constr_data in list(itertools.chain(
                instance.component_data_objects(SOSConstraint, active=True, descend_into=True)
                , instance.component_data_objects(Constraint, active=True, descend_into=True))):
            if _get_stage(constr_data, id_dict) < 1:
                _del_con(constr_data)

        # creates the sub map to remove first stage variables from objective expression
        complicating_vars_map = pyo.ComponentMap()
        var_list = list()
        sub_map = dict()
        for var_data in var_dict['stage1']:
            var_data.setlb(None)
            var_data.setub(None)
            var_list.append(var_data)
            sub_map[id(var_data)] = 0.0
        for var_data in var_dict['stage2']:
            sub_map[id(var_data)] = var_data

        # creates the complicating var map that connects the first stage variables in the sub problem to those in
        # the master problem
        for var, mvar in zip(var_list, self.master_vars):
            if var.name != mvar.name:
                raise Exception("Error: Complicating variable mismatch, sub-problem variables changed order")
            complicating_vars_map[mvar] = var

        # pulls the scenario objective expression, removes the first stage variables, and sets the new objective
        objs = list(instance.component_data_objects(Objective, descend_into=False, active=True))

        if len(objs) > 1:
            raise Exception("Error: Can not handle multiple objectives")

        obj_sense = objs[0].sense
        expr = replace_expressions(objs[0], sub_map)

        # checks if model sense is max, if so negates the objective
        if obj_sense == pyo.maximize:
            expr = -expr

        instance.del_component(objs[0])

        if not hasattr(instance, "PySP_prob"):
            instance.PySP_prob = 1. / self.scenario_count

        # set sub problem objective function
        instance.obj = pyo.Objective(expr=expr * instance.PySP_prob, sense=pyo.minimize)

        if self.store_subproblems:
            self.subproblems[scenario_name] = instance

        return instance, complicating_vars_map

    def lshaped_algorithm(self, converger=None, spcomm=None):
        """ function that runs the lshaped.py algorithm
        """
        if converger:
            converger = converger(self, self.rank, self.n_proc)
        max_iter = 30
        if "max_iter" in self.options:
            max_iter = self.options["max_iter"]
        tol = 1e-8
        if "tol" in self.options:
            tol = self.options["tol"]
        verbose = True
        if "verbose" in self.options:
            verbose = self.options["verbose"]
        master_solver = self.options["master_solver"]
        sp_solver = self.options["sp_solver"]

        # sets the first scenario in the scenario name list to serve as the "basis" for creating the master problem
        master_basis = self.all_scenario_names[0]

        # creates the master problem
        self.create_master(master_basis)
        m = self.master
        assert hasattr(m, "obj")

        # prevents problems from first stage variables becoming unconstrained
        # after processing
        for var in self.master_vars:
            if var.stale:
                var.set_value(0)

        # sets up the BendersCutGenerator object
        m.bender = LShapedCutGenerator()

        m.bender.set_input(master_vars=self.master_vars, tol=tol, comm=self.mpicomm)

        # let the cut generator know who's using it, probably should check that this is called after set input
        m.bender.set_ls(self)

        # set the eta variables, removing this from the add_suproblem function so we can

        # Pass all the scenarios in the problem to bender.add_subproblem
        # and let it internally handle which ranks get which scenarios
        if self.has_master_scens:
            sub_scenarios = [
                scenario_name for scenario_name in self.all_scenario_names
                if scenario_name not in self.master_scenarios
            ]
        else:
            sub_scenarios = self.all_scenario_names
        for scenario_name in self.local_scenario_names:
            subproblem_fn_kwargs = dict()
            subproblem_fn_kwargs['scenario_name'] = scenario_name
            m.bender.add_subproblem(
                subproblem_fn=self.create_subproblem,
                subproblem_fn_kwargs=subproblem_fn_kwargs,
                master_eta=m.eta[scenario_name],
                subproblem_solver=sp_solver,
                subproblem_solver_options=self.options["sp_solver_options"]
            )

        opt = pyo.SolverFactory(master_solver)
        if opt is None:
            raise Exception("Error: Failed to Create Master Solver")

        # set options
        for k,v in self.options["master_solver_options"].items():
            opt.options[k] = v

        is_persistent = sputils.is_persistent(opt)
        if is_persistent:
            opt.set_instance(m)

        t = time.time()
        res, t1, t2 = None, None, None

        # benders solve loop, repeats the benders master - subproblem
        # loop until either a no more cuts can are generated
        # or the maximum iterations limit is reached
        for self.iter in range(max_iter):
            if verbose and self.rank == self.rank0:
                if self.iter > 0:
                    print("Current Iteration:", self.iter + 1, "Time Elapsed:", "%7.2f" % (time.time() - t), "Time Spent on Last Master:", "%7.2f" % t1,
                          "Time Spent Generating Last Cut Set:", "%7.2f" % t2, "Current Objective:", "%7.2f" % m.obj.expr())
                else:
                    print("Current Iteration:", self.iter + 1, "Time Elapsed:", "%7.2f" % (time.time() - t), "Current Objective: -Inf")
            t1 = time.time()
            x_vals = np.zeros(len(self.master_vars))
            eta_vals = np.zeros(self.scenario_count)
            outer_bound = np.zeros(1)
            if self.rank == self.rank0:
                if is_persistent:
                    res = opt.solve(tee=False)
                else:
                    res = opt.solve(m, tee=False)
                # LShaped is always minimizing
                outer_bound[0] = res.Problem[0].Lower_bound
                for i, var in enumerate(self.master_vars):
                    x_vals[i] = var.value
                for i, eta in enumerate(m.eta.values()):
                    eta_vals[i] = eta.value

            self.mpicomm.Bcast(x_vals, root=self.rank0)
            self.mpicomm.Bcast(eta_vals, root=self.rank0)
            self.mpicomm.Bcast(outer_bound, root=self.rank0)

            if self.is_minimizing:
                self._LShaped_bound = outer_bound[0]
            else:
                # LShaped is always minimizing, so negate
                # the outer bound for sharing broadly
                self._LShaped_bound = -outer_bound[0]

            if self.rank != self.rank0:
                for i, var in enumerate(self.master_vars):
                    var.set_value(x_vals[i])
                for i, eta in enumerate(m.eta.values()):
                    eta.set_value(eta_vals[i])
            t1 = time.time() - t1
            t2 = time.time()
            cuts_added = m.bender.generate_cut()
            t2 = time.time() - t2
            if self.rank == self.rank0:
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
                    break
                if verbose and self.iter == max_iter - 1:
                    print("WARNING MAX ITERATION LIMIT REACHED !!! ")
            else:
                if len(cuts_added) == 0:
                    break
            # The hub object takes precedence over the converger
            if spcomm:
                converged = spcomm.opt_callback()
                if converged:
                    break
            if converger:
                converger.convergence_value()
                if converger.is_converged():
                    if verbose and self.rank == self.rank0:
                        print(
                            f"Converged to user criteria in {self.iter+1} iterations.\n"
                            f"Total Time Elapsed: {time.time()-t:7.2f} "
                            f"Time Spent on Last Master: {t1:7.2f} "
                            f"Time spent verifying second stage: {t2:7.2f} "
                            f"Final Objective: {m.obj.expr():7.2f}"
                        )
                    break
        return res


def main():
    import mpisppy.examples.farmer.farmer as ref
    import os
    # Turn off output from all ranks except rank 1
    if MPI.COMM_WORLD.Get_rank() != 0:
        sys.stdout = open(os.devnull, 'w')
    scenario_names = ['scen' + str(i) for i in range(3)]
    bounds = {i:-432000 for i in scenario_names}
    options = {
        "master_solver": "gurobi_persistent",
        "sp_solver": "gurobi_persistent",
        "sp_solver_options" : {"threads" : 1},
        "valid_eta_lb": bounds,
        "max_iter": 10,
   }

    ls = LShapedMethod(options, scenario_names, ref.scenario_creator)
    res = ls.lshaped_algorithm()
    if ls.rank == 0:
        print(res)


if __name__ == '__main__':
    main()
