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

def _get_nonant_ids(instance):
    assert len(instance._PySPnode_list) == 1
    # set comprehension
    nonant_list = instance._PySPnode_list[0].nonant_vardata_list
    return nonant_list, { id(var) for var in nonant_list }

def _first_stage_only(constr_data, nonant_ids):
    """ iterates through the constraint in a scenario and returns if it only
        has first stage variables
    """
    for var in identify_variables(constr_data.body):
        if id(var) not in nonant_ids: 
            return False
    return True

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
        if self.multistage:
            raise Exception("LShaped does not currently support multiple stages")
        self.options = options
        self.options_check()
        self.all_scenario_names = all_scenario_names
        self.master = None
        self.master_vars = None
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

    def _add_master_etas(self, master, index):
        master.eta = pyo.Var(index, within=pyo.Reals)
        if self.has_valid_eta_lb:
            for scen, eta in master.eta.items():
                eta.setlb(self.valid_eta_lb[scen])
        else:
            for eta in master.eta.values():
                eta.setlb((-sys.maxsize - 1) * 1. / len(self.all_scenario_names))

    def _create_master_no_scenarios(self):

        # using the first scenario as a basis
        master = self.scenario_creator(self.all_scenario_names[0], 
                                    node_names=None, cb_data=self.cb_data)

        if self.relax_master:
            RelaxIntegrality().apply_to(master)

        nonant_list, nonant_ids = _get_nonant_ids(master)

        self.master_vars = nonant_list

        for constr_data in list(itertools.chain(
                master.component_data_objects(SOSConstraint, active=True, descend_into=True)
                , master.component_data_objects(Constraint, active=True, descend_into=True))):
            if not _first_stage_only(constr_data, nonant_ids):
                _del_con(constr_data)

        # delete the second stage variables
        for var in list(master.component_data_objects(Var, active=True, descend_into=True)):
            if id(var) not in nonant_ids:
                _del_var(var)

        self._add_master_etas(master, self.all_scenario_names)

        # pulls the current objective expression, adds in the eta variables,
        # and removes the second stage variables from the expression
        objs = list(master.component_data_objects(Objective, descend_into=False, active=True))
        if len(objs) > 1:
            raise Exception("Error: Cannot handle multiple objectives")

        ## only keep the first stage variables in the objective
        sub_map = dict()
        for var in identify_variables(objs[0]):
            id_var = id(var)
            if id_var in nonant_ids:
                sub_map[id_var] = var
            else:
                sub_map[id_var] = 0.

        obj_sense = objs[0].sense
        expr = replace_expressions(objs[0], sub_map, remove_named_expressions=False)

        # checks if model sense is max, if so negates the objective
        if obj_sense == pyo.maximize:
            expr = -expr

        expr = expr + sum(master.eta.values())
        master.del_component(objs[0])

        # set master objective function
        master.obj = pyo.Objective(expr=expr, sense=pyo.minimize)

        self.master = master
        master.pprint()

    def _create_master_with_scenarios(self):

        # creates the eta variables for scenarios that are NOT selected to be
        # included in the master problem
        eta_indx = [scenario_name for scenario_name in self.all_scenario_names
                        if scenario_name not in self.master_scenarios]
        self._add_master_etas(master, eta_indx)


    def create_master(self):
        """ creates a ConcreteModel from one of the problem scenarios then
            modifies the model to serve as the master problem 
        """

        if self.has_master_scens:
            self._create_master_with_scenarios()
        else:
            self._create_master_no_scenarios()
        
    ## TODO: FIXME
    def add_master_scenarios(self, master, expr, obj_sense):
        for scenario_name in self.master_scenarios:
            # Scenarios have not yet been assigned 
            # to ranks so you have to call the 
            # scenario_creator again
            instance = self.scenario_creator(scenario_name, node_names=None,
                                                cb_data=self.cb_data)
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

    def attach_nonant_var_map(self, scenario_name):
        instance = self.local_scenarios[scenario_name]

        for var, mvar in zip(instance._nonant_indexes.values(), self.master_vars):
            if var.name != mvar.name:
                raise Exception("Error: Complicating variable mismatch, sub-problem variables changed order")
            subproblem_to_master_vars_map[var] = mvar 

        # this is for interefacing with PH code
        instance._subproblem_to_master_vars_map = subproblem_to_master_vars_map

    def create_subproblem(self, scenario_name):
        """ the subproblem creation function passed into the
            BendersCutsGenerator 
        """
        instance = self.local_scenarios[scenario_name]
        # relaxes any integrality constraints for the subproblem
        RelaxIntegrality().apply_to(instance)

        nonant_list, nonant_ids = _get_nonant_ids(instance) 

        # iterates through constraints and removes first stage constraints from the model
        # the id dict is used to improve the speed of identifying the stage each variables belongs to
        for constr_data in list(itertools.chain(
                instance.component_data_objects(SOSConstraint, active=True, descend_into=True)
                , instance.component_data_objects(Constraint, active=True, descend_into=True))):
            if _first_stage_only(constr_data, nonant_ids):
                _del_con(constr_data)

        # creates the sub map to remove first stage variables from objective expression
        complicating_vars_map = pyo.ComponentMap()
        subproblem_to_master_vars_map = pyo.ComponentMap()

        # creates the complicating var map that connects the first stage variables in the sub problem to those in
        # the master problem
        for var, mvar in zip(nonant_list, self.master_vars):
            if var.name != mvar.name:
                raise Exception("Error: Complicating variable mismatch, sub-problem variables changed order")
            complicating_vars_map[mvar] = var
            subproblem_to_master_vars_map[var] = mvar 

        # this is for interefacing with PH code
        instance._subproblem_to_master_vars_map = subproblem_to_master_vars_map

        # pulls the scenario objective expression, removes the first stage variables, and sets the new objective
        objs = list(instance.component_data_objects(Objective, descend_into=False, active=True))

        if len(objs) > 1:
            raise Exception("Error: Can not handle multiple objectives")

        sub_map = dict()

        for var in identify_variables(objs[0]):
            id_var = id(var)
            if id_var in nonant_ids:
                sub_map[id_var] = 0.0
            else:
                sub_map[id_var] = var

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

        # creates the master problem
        self.create_master()
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
                scenario_name for scenario_name in self.local_scenario_names
                if scenario_name not in self.master_scenarios
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
                    master_eta=m.eta[scenario_name],
                    subproblem_solver=sp_solver,
                    subproblem_solver_options=self.options["sp_solver_options"]
                )
            else:
                self.attach_nonant_var_map(scenario_name)

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
                    var._value = x_vals[i]
                for i, eta in enumerate(m.eta.values()):
                    eta._value = eta_vals[i]
            t1 = time.time() - t1

            # The hub object takes precedence over the converger
            # We'll send the nonants now, and check for a for
            # convergence
            if spcomm:
                spcomm.sync_with_spokes(send_nonants=True)
                converged = spcomm.is_converged()
                if converged:
                    break

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
                spcomm.sync_with_spokes(send_nonants=False)
                converged = spcomm.is_converged()
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
