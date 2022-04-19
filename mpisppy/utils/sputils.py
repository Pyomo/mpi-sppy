# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Base and utility functions for mpisppy
# Note to developers: things called spcomm are way more than just a comm; SPCommunicator

import pyomo.environ as pyo
import sys
import os
import re
import time
import numpy as np
import mpisppy.scenario_tree as scenario_tree
from pyomo.core import Objective

from mpisppy import MPI, haveMPI
global_rank = MPI.COMM_WORLD.Get_rank()
from pyomo.core.expr.numeric_expr import LinearExpression

from mpisppy import tt_timer, global_toc

_spin_the_wheel_move_msg = \
        "spin_the_wheel should now be used as the class "\
        "mpisppy.spin_the_wheel.WheelSpinner using the method `spin()`. Output "\
        "writers are now methods of the class WheelSpinner."

def spin_the_wheel(hub_dict, list_of_spoke_dict, comm_world=None):
    raise RuntimeError(
            _spin_the_wheel_move_msg + \
            " See the example code below for a fix:\n"
    '''
    from mpisppy.spin_the_wheel import WheelSpinner
    ws = WheelSpinner(hub_dict, list_of_spoke_dict)
    ws.spin(comm_world=comm_world)
    '''
    )

def first_stage_nonant_npy_serializer(file_name, scenario, bundling):
    # write just the nonants for ROOT in an npy file (e.g. for Conf Int)
    root = scenario._mpisppy_node_list[0]
    assert root.name == "ROOT"
    root_nonants = np.fromiter((pyo.value(var) for var in root.nonant_vardata_list), float)
    np.save(file_name, root_nonants)

def first_stage_nonant_writer( file_name, scenario, bundling ):
    with open(file_name, 'w') as f:
        root = scenario._mpisppy_node_list[0]
        assert root.name == "ROOT"
        for var in root.nonant_vardata_list:
            var_name = var.name
            if bundling:
                dot_index = var_name.find('.')
                assert dot_index >= 0
                var_name = var_name[(dot_index+1):]
            f.write(f"{var_name},{pyo.value(var)}\n")

def scenario_tree_solution_writer( directory_name, scenario_name, scenario, bundling ):
    with open(os.path.join(directory_name, scenario_name+'.csv'), 'w') as f:
        for var in scenario.component_data_objects(
                ctype=(pyo.Var, pyo.Expression),
                descend_into=True,
                active=True,
                sort=True):
            var_name = var.name
            if bundling:
                dot_index = var_name.find('.')
                assert dot_index >= 0
                var_name = var_name[(dot_index+1):]
            f.write(f"{var_name},{pyo.value(var)}\n")
        
def write_spin_the_wheel_first_stage_solution(spcomm, opt_dict, solution_file_name,
        first_stage_solution_writer=first_stage_nonant_writer):
    raise RuntimeError(_spin_the_wheel_move_msg)

def write_spin_the_wheel_tree_solution(spcomm, opt_dict, solution_directory_name,
        scenario_tree_solution_writer=scenario_tree_solution_writer):
    raise RuntimeError(_spin_the_wheel_move_msg)

def local_nonant_cache(spcomm):
    raise RuntimeError(_spin_the_wheel_move_msg)


def get_objs(scenario_instance, allow_none=False):
    """ return the list of objective functions for scenario_instance"""
    scenario_objs = scenario_instance.component_data_objects(pyo.Objective,
                    active=True, descend_into=True)
    scenario_objs = list(scenario_objs)
    if (len(scenario_objs) == 0) and not allow_none:
        raise RuntimeError(f"Scenario {scenario_instance.name} has no active "
                           "objective functions.")
    if (len(scenario_objs) > 1):
        print("WARNING: Scenario", sname, "has multiple active "
              "objectives. Selecting the first objective for "
                  "inclusion in the extensive form.")
    return scenario_objs


def stash_ref_objs(scenario_instance):
    """Stash a reference to active objs so
        Reactivate_obj can use the reference to reactivate them/it later.
    """
    scenario_instance._mpisppy_data.obj_list = get_objs(scenario_instance)


def deact_objs(scenario_instance):
    """ Deactivate objs 
    Args:
        scenario_instance (Pyomo ConcreteModel): the scenario
    Returns:
        obj_list (list of Pyomo Objectives): the deactivated objs
    Note: If none are active, just do nothing
    """
    obj_list = get_objs(scenario_instance, allow_none=True)
    for obj in obj_list:
        obj.deactivate()
    return obj_list


def reactivate_objs(scenario_instance):
    """ Reactivate ojbs stashed by stash_ref_objs """
    if not hasattr(scenario_instance._mpisppy_data, "obj_list"):
        raise RuntimeError("reactivate_objs called with prior call to stash_ref_objs")
    for obj in scenario_instance._mpisppy_data.obj_list:
        obj.activate()


def create_EF(scenario_names, scenario_creator, scenario_creator_kwargs=None,
              EF_name=None, suppress_warnings=False,
              nonant_for_fixed_vars=True):
    """ Create a ConcreteModel of the extensive form.

        Args:
            scenario_names (list of str):
                Names for each scenario to be passed to the scenario_creator
                function.
            scenario_creator (callable):
                Function which takes a scenario name as its first argument and
                returns a concrete model corresponding to that scenario.
            scenario_creator_kwargs (dict, optional):
                Options to pass to `scenario_creator`.
            EF_name (str, optional):
                Name of the ConcreteModel of the EF.
            suppress_warnings (boolean, optional):
                If true, do not display warnings. Default False.
            nonant_for_fixed_vars (bool--optional): If True, enforces
                non-anticipativity constraints for all variables, including
                those which have been fixed. Default is True.

        Returns:
            EF_instance (ConcreteModel):
                ConcreteModel of extensive form with explicit
                non-anticipativity constraints.

        Note:
            If any of the scenarios produced by scenario_creator do not have a
            ._mpisppy_probability attribute, this function displays a warning, and assumes
            that all scenarios are equally likely.
    """
    if scenario_creator_kwargs is None:
        scenario_creator_kwargs = dict()
    scen_dict = {
        name: scenario_creator(name, **scenario_creator_kwargs)
        for name in scenario_names
    }

    if (len(scen_dict) == 0):
        raise RuntimeError("create_EF() received empty scenario list")
    elif (len(scen_dict) == 1):
        scenario_instance = list(scen_dict.values())[0]
        scenario_instance._ef_scenario_names = list(scen_dict.keys())
        if not suppress_warnings:
            print("WARNING: passed single scenario to create_EF()")
        # special code to patch in ref_vars
        scenario_instance.ref_vars = dict()
        scenario_instance._nlens = {node.name: len(node.nonant_vardata_list) 
                                for node in scenario_instance._mpisppy_node_list}
        for node in scenario_instance._mpisppy_node_list:
            ndn = node.name

            for i in range(scenario_instance._nlens[ndn]):
                v = node.nonant_vardata_list[i]
                if (ndn, i) not in scenario_instance.ref_vars:
                    scenario_instance.ref_vars[(ndn, i)] = v
        # patch in EF_Obj        
        scenario_objs = deact_objs(scenario_instance)        
        obj = scenario_objs[0]            
        sense = pyo.minimize if obj.is_minimizing() else pyo.maximize
        scenario_instance.EF_Obj = pyo.Objective(expr=obj.expr, sense=sense)

        return scenario_instance  #### special return for single scenario

    # Check if every scenario has a specified probability
    probs_specified = \
        all([hasattr(scen, '_mpisppy_probability') for scen in scen_dict.values()])
    if not probs_specified:
        for scen in scen_dict.values():
            scen._mpisppy_probability = 1 / len(scen_dict)
        if not suppress_warnings:
            print('WARNING: At least one scenario is missing _mpisppy_probability attribute.',
                  'Assuming equally-likely scenarios...')

    EF_instance = _create_EF_from_scen_dict(scen_dict,
                                            EF_name=EF_name,
                                            nonant_for_fixed_vars=True)
    return EF_instance

def _create_EF_from_scen_dict(scen_dict, EF_name=None,
                                nonant_for_fixed_vars=True):
    """ Create a ConcreteModel of the extensive form from a scenario
        dictionary.

        Args:
            scen_dict (dict): Dictionary whose keys are scenario names and
                values are ConcreteModel objects corresponding to each
                scenario.
            EF_name (str--optional): Name of the resulting EF model.
            nonant_for_fixed_vars (bool--optional): If True, enforces
                non-anticipativity constraints for all variables, including
                those which have been fixed. Default is True.

        Returns:
            EF_instance (ConcreteModel): ConcreteModel of extensive form with
                explicity non-anticipativity constraints.

        Notes:
            The non-anticipativity constraints are enforced by creating
            "reference variables" at each node in the scenario tree (excluding
            leaves) and enforcing that all the variables for each scenario at
            that node are equal to the reference variables.

            This function is called directly when creating bundles for PH.
 
            Does NOT assume that each scenario is equally likely. Raises an
            AttributeError if a scenario object is encountered which does not
            have a ._mpisppy_probability attribute.

            Added the flag nonant_for_fixed_vars because original code only
            enforced non-anticipativity for non-fixed vars, which is not always
            desirable in the context of bundling. This allows for more
            fine-grained control.
    """
    is_min, clear = _models_have_same_sense(scen_dict)
    if (not clear):
        raise RuntimeError('Cannot build the extensive form out of models '
                           'with different objective senses')
    sense = pyo.minimize if is_min else pyo.maximize
    EF_instance = pyo.ConcreteModel(name=EF_name)
    EF_instance.EF_Obj = pyo.Objective(expr=0.0, sense=sense)

    # we don't strictly need these here, but it allows for eliding
    # of single scenarios and bundles when convenient
    EF_instance._mpisppy_data = pyo.Block(name="For non-Pyomo mpi-sppy data")
    EF_instance._mpisppy_model = pyo.Block(name="For mpi-sppy Pyomo additions to the scenario model")
    EF_instance._mpisppy_data.scenario_feasible = None

    EF_instance._ef_scenario_names = []
    EF_instance._mpisppy_probability = 0
    for (sname, scenario_instance) in scen_dict.items():
        EF_instance.add_component(sname, scenario_instance)
        EF_instance._ef_scenario_names.append(sname)
        # Now deactivate the scenario instance Objective
        scenario_objs = deact_objs(scenario_instance)
        obj_func = scenario_objs[0] # Select the first objective
        try:
            EF_instance.EF_Obj.expr += scenario_instance._mpisppy_probability * obj_func.expr
            EF_instance._mpisppy_probability   += scenario_instance._mpisppy_probability
        except AttributeError as e:
            raise AttributeError("Scenario " + sname + " has no specified "
                        "probability. Specify a value for the attribute "
                        " _mpisppy_probability and try again.") from e
    # Normalization does nothing when solving the full EF, but is required for
    # appropraite scaling of EFs used as bundles.
    EF_instance.EF_Obj.expr /= EF_instance._mpisppy_probability

    # For each node in the scenario tree, we need to collect the
    # nonanticipative vars and create the constraints for them,
    # which we do using a reference variable.
    ref_vars = dict() # keys are _nonant_indices (i.e. a node name and a
                      # variable number)

    ref_suppl_vars = dict()

    EF_instance._nlens = dict() 

    nonant_constr = pyo.Constraint(pyo.Any, name='_C_EF_')
    EF_instance.add_component('_C_EF_', nonant_constr)

    nonant_constr_suppl = pyo.Constraint(pyo.Any, name='_C_EF_suppl')
    EF_instance.add_component('_C_EF_suppl', nonant_constr_suppl)

    for (sname, s) in scen_dict.items():
        nlens = {node.name: len(node.nonant_vardata_list) 
                            for node in s._mpisppy_node_list}
        
        for (node_name, num_nonant_vars) in nlens.items(): # copy nlens to EF
            if (node_name in EF_instance._nlens.keys() and
                num_nonant_vars != EF_instance._nlens[node_name]):
                raise RuntimeError("Number of non-anticipative variables is "
                    "not consistent at node " + node_name + " in scenario " +
                    sname)
            EF_instance._nlens[node_name] = num_nonant_vars

        nlens_ef_suppl = {node.name: len(node.nonant_ef_suppl_vardata_list)
                                   for node in s._mpisppy_node_list}

        for node in s._mpisppy_node_list:
            ndn = node.name
            for i in range(nlens[ndn]):
                v = node.nonant_vardata_list[i]
                if (ndn, i) not in ref_vars:
                    # create the reference variable as a singleton with long name
                    # xxxx maybe index by _nonant_index ???? rather than singleton VAR ???
                    ref_vars[(ndn, i)] = v
                # Add a non-anticipativity constraint, except in the case when
                # the variable is fixed and nonant_for_fixed_vars=False.
                elif (nonant_for_fixed_vars) or (not v.is_fixed()):
                    expr = LinearExpression(linear_coefs=[1,-1],
                                            linear_vars=[v,ref_vars[(ndn,i)]],
                                            constant=0.)
                    nonant_constr[(ndn,i,sname)] = (expr, 0.0)

            for i in range(nlens_ef_suppl[ndn]):
                v = node.nonant_ef_suppl_vardata_list[i]
                if (ndn, i) not in ref_suppl_vars:
                    # create the reference variable as a singleton with long name
                    # xxxx maybe index by _nonant_index ???? rather than singleton VAR ???
                    ref_suppl_vars[(ndn, i)] = v
                # Add a non-anticipativity constraint, expect in the case when
                # the variable is fixed and nonant_for_fixed_vars=False.
                elif (nonant_for_fixed_vars) or (not v.is_fixed()):
                        expr = LinearExpression(linear_coefs=[1,-1],
                                                linear_vars=[v,ref_suppl_vars[(ndn,i)]],
                                                constant=0.)
                        nonant_constr_suppl[(ndn,i,sname)] = (expr, 0.0)

    EF_instance.ref_vars = ref_vars
    EF_instance.ref_suppl_vars = ref_suppl_vars
                        
    return EF_instance

def _models_have_same_sense(models):
    ''' Check if every model in the provided dict has the same objective sense.

        Input:
            models (dict) -- Keys are scenario names, values are Pyomo
                ConcreteModel objects.
        Returns:
            is_minimizing (bool) -- True if and only if minimizing. None if the
                check fails.
            check (bool) -- True only if all the models have the same sense (or
                no models were provided)
        Raises:
            ValueError -- If any of the models has either none or multiple
                active objectives.
    '''
    if (len(models) == 0):
        return True, True
    senses = [find_active_objective(scenario).is_minimizing()
                for scenario in models.values()]
    sense = senses[0]
    check = all(val == sense for val in senses)
    if (check):
        return (sense == pyo.minimize), check
    return None, check

def is_persistent(solver):
    return isinstance(solver,
        pyo.pyomo.solvers.plugins.solvers.persistent_solver.PersistentSolver)
    
def ef_scenarios(ef):
    """ An iterator to give the scenario sub-models in an ef
    Args:
        ef (ConcreteModel): the full extensive form model

    Yields:
        scenario name, scenario instance (str, ConcreteModel)
    """    
    for sname in ef._ef_scenario_names:
        yield (sname, getattr(ef, sname))

def ef_nonants(ef):
    """ An iterator to give representative Vars subject to non-anticipitivity
    Args:
        ef (ConcreteModel): the full extensive form model

    Yields:
        tree node name, full EF Var name, Var value

    Note:
        not on an EF object because not all ef's are part of an EF object
    """
    for (ndn,i), var in ef.ref_vars.items():
        yield (ndn, var, pyo.value(var))

        
def ef_nonants_csv(ef, filename):
    """ Dump the nonant vars from an ef to a csv file; truly a dump...
    Args:
        ef (ConcreteModel): the full extensive form model
        filename (str): the full name of the csv output file
    """
    with open(filename, "w") as outfile:
        outfile.write("Node, EF_VarName, Value\n")
        for (ndname, varname, varval) in ef_nonants(ef):
            outfile.write("{}, {}, {}\n".format(ndname, varname, varval))

            
def nonant_cache_from_ef(ef,verbose=False):
    """ Populate a nonant_cache from an ef. Also works with multi-stage
    Args:
        ef (mpi-sppy ef): a solved ef
    Returns:
        nonant_cache (dict of numpy arrays): a special structure for nonant values
    """     
    nonant_cache = dict()
    nodenames = set([ndn for (ndn,i) in ef.ref_vars])
    for ndn in sorted(nodenames):
        nonant_cache[ndn]=[]
        i = 0
        while ((ndn,i) in ef.ref_vars):
            xvar = pyo.value(ef.ref_vars[(ndn,i)])
            nonant_cache[ndn].append(xvar)
            if verbose:
                print("barfoo", i, xvar)
            i+=1
    return nonant_cache


def ef_ROOT_nonants_npy_serializer(ef, filename):
    """ write the root node nonants to be ready by a numpy load
    Args:
        ef (ConcreteModel): the full extensive form model
        filename (str): the full name of the .npy output file
    """
    root_nonants = np.fromiter((v for ndn,var,v in ef_nonants(ef) if ndn == "ROOT"), float)
    np.save(filename, root_nonants)

def write_ef_first_stage_solution(ef,
                                  solution_file_name,
                                  first_stage_solution_writer=first_stage_nonant_writer):
    """ 
    Write a solution file, if a solution is available, to the solution_file_name provided
    Args:
        ef : A Concrete Model of the Extensive Form (output of create_EF). 
             We assume it has already been solved.
        solution_file_name : filename to write the solution to
        first_stage_solution_writer (optional) : custom first stage solution writer function
    
    NOTE:
        This utility is replicating WheelSpinner.write_first_stage_solution for EF
    """
    if global_rank==0:
        dirname = os.path.dirname(solution_file_name)
        if dirname != '':
            os.makedirs(os.path.dirname(solution_file_name), exist_ok=True)
            representative_scenario = getattr(ef,ef._ef_scenario_names[0])
            first_stage_solution_writer(solution_file_name, 
                                        representative_scenario,
                                        bundling=False)

def write_ef_tree_solution(ef, solution_directory_name,
        scenario_tree_solution_writer=scenario_tree_solution_writer):
    """ Write a tree solution directory, if available, to the solution_directory_name provided
    Args:
        ef : A Concrete Model of the Extensive Form (output of create_EF). 
             We assume it has already been solved.
        solution_file_name : filename to write the solution to
        scenario_tree_solution_writer (optional) : custom scenario solution writer function
        
    NOTE:
        This utility is replicating WheelSpinner.write_tree_solution for EF
    """
    if global_rank==0:
        os.makedirs(solution_directory_name, exist_ok=True)
        for scenario_name, scenario in ef_scenarios(ef):
            scenario_tree_solution_writer(solution_directory_name,
                                          scenario_name, 
                                          scenario,
                                          bundling=False)
    

def extract_num(string):
    ''' Given a string, extract the longest contiguous
        integer from the right-hand side of the string.

        Example:
            scenario324 -> 324

        TODO: Add Exception Handling
    '''
    return int(re.compile(r'(\d+)$').search(string).group(1))

def node_idx(node_path,branching_factors):
    '''
    Computes a unique id for a given node in a scenario tree.
    It follows the path to the node, computing the unique id for each ascendant.

    Parameters
    ----------
    node_path : list of int
        A list of integer, specifying the path of the node.
    branching_factors : list of int
        branching_factors of the scenario tree.

    Returns
    -------
    node_idx
        Node unique id.
        
    NOTE: Does not work with unbalanced trees.

    '''
    if node_path == []: #ROOT node
        return 0
    else:
        stage_id = 0 #node unique id among stage t+1 nodes.
        for t in range(len(node_path)):
            stage_id = node_path[t]+branching_factors[t]*stage_id
        node_idx = _nodenum_before_stage(len(node_path),branching_factors)+stage_id
        return node_idx

def _extract_node_idx(nodename,branching_factors):
    """
    

    Parameters
    ----------
    nodename : str
        The name of a node, e.g. 'ROOT_2_0_4'.
    branching_factors : list
        Branching factor of a scenario tree, e.g. [3,2,8,4,3].

    Returns
    -------
    node_idx : int
        A unique integer that can be used as a key to designate this scenario.

    """
    if nodename =='ROOT':
        return 0
    else:
        to_list = [int(x) for x in re.findall(r'\d+',nodename)]
        return node_idx(to_list,branching_factors)

def parent_ndn(nodename):
    if nodename == 'ROOT':
        return None
    else:
        return re.search('(.+)_(\d+)',nodename).group(1)
        
def option_string_to_dict(ostr):
    """ Convert a string to the standard dict for solver options.
    Intended for use in the calling program; not internal use here.

    Args:
        ostr (string): space seperated options with = for arguments

    Returns:
        solver_options (dict): solver options

    """
    def convert_value_string_to_number(s):
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s

    solver_options = dict()
    if ostr is None or ostr == "":
        return None
    for this_option_string in ostr.split():
        this_option_pieces = this_option_string.strip().split("=")
        if len(this_option_pieces) == 2:
            option_key = this_option_pieces[0]
            option_value = convert_value_string_to_number(this_option_pieces[1])
            solver_options[option_key] = option_value
        elif len(this_option_pieces) == 1:
            option_key = this_option_pieces[0]
            solver_options[option_key] = None
        else:
            raise RuntimeError("Illegally formed subsolve directive"\
                               + " option=%s detected" % this_option)
    return solver_options


################################################################################
# Various utilities related to scenario rank maps (some may not be in use)


def scens_to_ranks(scen_count, n_proc, rank, branching_factors = None):
    """ Determine the rank assignments that are made in spbase.
    NOTE: Callers to this should call _scentree.scen_names_to_ranks
    Args:
        scen_count (int): number of scenarios
        n_proc (int): the number of intra ranks (within the cylinder)
        rank (int): my rank (i.e., intra; i.e., within the cylinder)
    Returns:
        slices (list of ranges): the indices into all all_scenario_names to assign to rank
                                 (the list entries are ranges that correspond to ranks)
        scenario_name_to_rank (dict of dict): only for multi-stage
                keys are comms (i.e., tree nodes); values are dicts with keys
                that are scenario names and values that are ranks

    """
    if not haveMPI:
        raise RuntimeError("scens_to_ranks called, but cannot import mpi4py")
    if scen_count < n_proc:
        raise RuntimeError(
            "More MPI ranks (%d) supplied than needed given the number of scenarios (%d) "
            % (n_proc, scen_count)
        )

    # for now, we are treating two-stage as a special case
    if (branching_factors is None):
        avg = scen_count / n_proc
        slices = [list(range(int(i * avg), int((i + 1) * avg))) for i in range(n_proc)]
        return slices, None
    else:
        # OCT 2020: this block is never triggered and would fail.
        # indecision as of May 2020 (delete this comment DLW)
        # just make sure things are consistent with what xhat will do...
        # TBD: streamline
        all_scenario_names = ["ID"+str(i) for i in range(scen_count)]
        tree = _ScenTree(branching_factors, all_scenario_names)
        scenario_names_to_ranks, slices, = tree.scen_name_to_rank(n_proc, rank)
        return slices, scenario_names_to_ranks

def _nodenum_before_stage(t,branching_factors):
    #How many nodes in a tree of stage 1,2,...,t ?
    #Only works with branching factors
    return int(sum(np.prod(branching_factors[0:i]) for i in range(t)))

def find_leaves(all_nodenames):
    #Take a list of all nodenames from a tree, and find the leaves of it.
    #WARNING: We do NOT check that the tree is well constructed
    
    if all_nodenames is None or all_nodenames == ['ROOT']:
        return {'ROOT':False} # 2 stage problem: no leaf nodes in all_nodenames
    #A leaf is simply a root with no child n°0
    is_leaf = dict()
    for ndn in all_nodenames:
        if ndn+"_0" in all_nodenames:
            is_leaf[ndn] = False
        else:
            is_leaf[ndn] = True
    return is_leaf
            
    
class _TreeNode():
    #Create the subtree generated by a node, with associated scenarios
    # stages are 1-based, everything else is 0-based
    # scenario lists are stored as (first, last) indices in all_scenarios
    #This is also checking that the nodes from all_nodenames are well-named.
    def __init__(self, Parent, scenfirst, scenlast, desc_leaf_dict, name):
        #desc_leaf_dict is the output of find_leaves
        self.scenfirst = scenfirst #id of the first scenario with this node
        self.scenlast = scenlast #id of the last scenario with this node
        self.name = name
        numscens = scenlast - scenfirst + 1 #number of scenarios with this node
        self.is_leaf = False
        if Parent is None:
            assert(self.name == "ROOT")
            self.stage = 1
        else:
            self.stage = Parent.stage + 1
        if len(desc_leaf_dict)==1 and list(desc_leaf_dict.keys()) == ['ROOT']: 
            #2-stage problem, we don't create leaf nodes
            self.kids = []
        elif not name+"_0" in desc_leaf_dict:
            self.is_leaf = True
            self.kids = []
        else:
            if len(desc_leaf_dict) < numscens:                
                raise RuntimeError(f"There are more scenarios ({numscens}) than remaining leaves, for the node {name}")
            # make children
            first = scenfirst
            self.kids = list()
            child_regex = re.compile(name+'_\d*\Z')
            child_list = [x for x in desc_leaf_dict if child_regex.match(x) ]
            for i in range(len(desc_leaf_dict)):
                childname = name+f"_{i}"
                if not childname in desc_leaf_dict:
                    if len(child_list) != i:
                        raise RuntimeError("The all_nodenames argument is giving an inconsistent tree."
                                           f"The node {name} has {len(child_list)} children, but {childname} is not one of them.")
                    break
                childdesc_regex = re.compile(childname+'(_\d*)*\Z')
                child_leaf_dict = {ndn:desc_leaf_dict[ndn] for ndn in desc_leaf_dict \
                                   if childdesc_regex.match(ndn)}
                #We determine the number of children of this node
                child_scens_num = sum(child_leaf_dict.values())
                last = first+child_scens_num - 1
                self.kids.append(_TreeNode(self, first, last, 
                                           child_leaf_dict, childname))
                first += child_scens_num
            if last != scenlast:
                print("numscens, last, scenlast", numscens, last, scenlast)
                raise RuntimeError(f"Tree node did not initialize correctly for node {name}")


    def stage_max(self):
        #Return the number of stages of a subtree.
        #Also check that all the subtrees have the same number of stages
        #i.e. that the leaves are always on the same stage. 
        if self.is_leaf:
            return 1
        else:
            l = [child.stage_max() for child in self.kids]
            if l.count(l[0]) != len(l):
                maxstage = max(l)+ self.stage
                minstage = min(l)+ self.stage
                raise RuntimeError("The all_nodenames argument is giving an inconsistent tree. "
                                   f"The node {self.name} has descendant leaves with stages going from {minstage} to {maxstage}")
            return 1+l[0]
            
                    

                
class _ScenTree():
    def __init__(self, all_nodenames, ScenNames):
        if all_nodenames is None:
            all_nodenames = ['ROOT'] #2 stage problem: no leaf nodes
        self.ScenNames = ScenNames
        self.NumScens = len(ScenNames)
        first = 0
        last = self.NumScens - 1
        desc_leaf_dict = find_leaves(all_nodenames)
        self.rootnode = _TreeNode(None, first, last, desc_leaf_dict, "ROOT")
        def _nonleaves(nd):
            if nd.is_leaf:
                return []
            else:
                retval = [nd]
                for child in nd.kids:
                    retval+=_nonleaves(child)
                return retval
        self.nonleaves = _nonleaves(self.rootnode)
        
        self.NumStages = \
            2 if all_nodenames == ['ROOT'] else self.rootnode.stage_max() 
        self.NonLeafTerminals = \
            [nd for nd in self.nonleaves if nd.stage == self.NumStages-1]
        
        self.NumLeaves = len(desc_leaf_dict) - len(self.nonleaves)
        if self.NumStages>2 and self.NumLeaves != self.NumScens:
            raise RuntimeError("The all_nodenames argument is giving an inconsistent tree."
                               f"There are {self.NumLeaves} leaves for this tree, but {self.NumScens} scenarios are given.")
    def scen_names_to_ranks(self, n_proc):
        """ 
        Args:
            n_proc: number of ranks in the cylinder (i.e., intra)

        Returns:
            scenario_names_to_rank (dict of dict):
                keys are comms (i.e., tree nodes); values are dicts with keys
                that are scenario names and values that are ranks within that comm
            slices (list of lists)
                indices correspond to ranks in self.mpicomm and the values are a list
                of scenario indices
                rank -> list of scenario indices for that rank
            list_of_ranks_per_scenario_idx (list)
                indices are scenario indices and values are the rank of that scenario
                within self.mpicomm
                scenario index -> rank

        NOTE:
            comm names are the same as the corresponding scenario tree node name

        """
        scenario_names_to_rank = dict()  # scenario_name_to_rank dict of dicts
        # one processor for the cylinder is a special case
        if n_proc == 1:
            for nd in self.nonleaves:
                scenario_names_to_rank[nd.name] = {s: 0 for s in self.ScenNames}
            return scenario_names_to_rank, [list(range(self.NumScens))], [0]*self.NumScens

        scen_count = len(self.ScenNames)
        avg = scen_count / n_proc

        # rank -> list of scenario indices for that rank
        slices = [list(range(int(i * avg), int((i + 1) * avg))) for i in range(n_proc)]

        # scenario index -> rank
        list_of_ranks_per_scenario_idx = [ rank for rank, scen_idxs in enumerate(slices) for _ in scen_idxs ]

        scenario_names_to_rank["ROOT"] = { s: rank for s,rank in zip(self.ScenNames, list_of_ranks_per_scenario_idx) }
         
        def _recurse_do_node(node):
            for child in node.kids:

                first_scen_idx = child.scenfirst
                last_scen_idx = child.scenlast

                ranks_in_node = list_of_ranks_per_scenario_idx[first_scen_idx:last_scen_idx+1]
                minimum_rank_in_node = ranks_in_node[0]

                # IMPORTANT:
                # this accords with the way SPBase.create_communicators assigns the "key" when
                # creating its comm for this node. E.g., the key is the existing rank, which
                # will then be offset by the minimum rank. As the ranks within each node are
                # contiguous, this is enough to infer the rank each scenario will have in this
                # node's comm
                within_comm_ranks_in_node = [(rank-minimum_rank_in_node) for rank in ranks_in_node]

                scenarios_in_nodes = self.ScenNames[first_scen_idx:last_scen_idx+1]

                scenario_names_to_rank[child.name] = { s : rank for s,rank in zip(scenarios_in_nodes, within_comm_ranks_in_node) }

                if child not in self.NonLeafTerminals:
                    _recurse_do_node(child)

        _recurse_do_node(self.rootnode)

        return scenario_names_to_rank, slices, list_of_ranks_per_scenario_idx
    
    
######## Utility to attach the one and only node to a two-stage scenario #######
def attach_root_node(model, firstobj, varlist, nonant_ef_suppl_list=None):
    """ Create a root node as a list to attach to a scenario model
    Args:
        model (ConcreteModel): model to which this will be attached
        firstobj (Pyomo Expression): First stage cost (e.g. model.FC)
        varlist (list): Pyomo Vars in first stage (e.g. [model.A, model.B])
        nonant_ef_suppl_list (list of pyo Var, Vardata or slices):
              vars for which nonanticipativity constraints tighten the EF
              (important for bundling)

    Note: 
       attaches a list consisting of one scenario node to the model
    """
    model._mpisppy_node_list = [
        scenario_tree.ScenarioNode("ROOT", 1.0, 1, firstobj, varlist, model,
                                   nonant_ef_suppl_list = nonant_ef_suppl_list)
    ]

### utilities to check the slices and the map ###
def check4losses(numscens, branching_factors,
                 scenario_names_to_rank,slices,list_of_ranks_per_scenario_idx):
    """ Check the data structures; gag and die if it looks bad.
    Args:
        numscens (int): number of scenarios
        branching_factors (list of int): branching factors
        scenario_names_to_rank (dict of dict):
            keys are comms (i.e., tree nodes); values are dicts with keys
            that are scenario names and values that are ranks within that comm
        slices (list of lists)
            indices correspond to ranks in self.mpicomm and the values are a list
            of scenario indices
            rank -> list of scenario indices for that rank
        list_of_ranks_per_scenario_idx (list)
            indices are scenario indices and values are the rank of that scenario
            within self.mpicomm
            scenario index -> rank

    """

    present = [False for _ in range(numscens)]
    for rank, scenlist in enumerate(slices):
        for scen in scenlist:
            present[scen] = True
    missingsome = False
    for scen, there in enumerate(present):
        if not there:
            print(f"Scenario {scen} is not in slices")
            missingsome = True
    if missingsome:
        raise RuntimeError("Internal error: slices is not correct")

    # not stage presence...
    stagepresents = {stage: [False for _ in range(numscens)] for stage in range(len(branching_factors))}
    # loop over the entire structure, marking those found as present
    for nodename, scenlist in scenario_names_to_rank.items():
        stagenum = nodename.count('_')
        for s in scenlist:
            snum = int(s[8:])
            stagepresents[stagenum][snum] = True
    missingone = False
    for stage in stagepresents:
        for scen, there in enumerate(stagepresents[stage]):
            if not there:
                print(f"Scenario number {scen} missing from stage {stage}.")
                missingsome = True
    if missingsome:
        raise RuntimeError("Internal error: scenario_name_to_rank")
    print("check4losses: OK")

    
def disable_tictoc_output():
    f = open(os.devnull,"w")
    tt_timer._ostream = f

def reenable_tictoc_output():
    # Primarily to re-enable after a disable
    tt_timer._ostream.close()
    tt_timer._ostream = sys.stdout

    
def find_active_objective(pyomomodel):
    # return the only active objective or raise and error
    obj = list(pyomomodel.component_data_objects(
        Objective, active=True, descend_into=True))
    if len(obj) != 1:
        raise RuntimeError("Could not identify exactly one active "
                           "Objective for model '%s' (found %d objectives)"
                           % (pyomomodel.name, len(obj)))
    return obj[0]

def create_nodenames_from_branching_factors(BFS):
    """
    This function creates the node names of a tree without creating the whole tree.

    Parameters
    ----------
    BFS : list of integers
        Branching factors.

    Returns
    -------
    nodenames : list of str
        a list of the node names induced by branching_factors, including leaf nodes.

    """
    stage_nodes = ["ROOT"]
    nodenames = ['ROOT']
    if len(BFS)==1 : #2stage
        return(nodenames)
    for bf in BFS[:(len(BFS))]:
        old_stage_nodes = stage_nodes
        stage_nodes = []
        for k in range(len(old_stage_nodes)):
            stage_nodes += ['%s_%i'%(old_stage_nodes[k],b) for b in range(bf)]
        nodenames += stage_nodes
    return nodenames

def get_branching_factors_from_nodenames(all_nodenames):
    #WARNING: Do not work with unbalanced trees
    staget_node = "ROOT"
    branching_factors = []
    while staget_node+"_0" in all_nodenames:
        child_regex = re.compile(staget_node+'_\d*\Z')
        child_list = [x for x in all_nodenames if child_regex.match(x) ]
        
        branching_factors.append(len(child_list))
        staget_node += "_0"
    if len(branching_factors)==1:
        #2stage
        return None
    else:
        return branching_factors
    
def number_of_nodes(branching_factors):
    #How many nodes does a tree with a given branching_factors have ?
    last_node_stage_num = [i-1 for i in branching_factors]
    return node_idx(last_node_stage_num, branching_factors)
    


          

if __name__ == "__main__":
    branching_factors = [2,2,2,3]
    numscens = np.prod(branching_factors)
    scennames = ["Scenario"+str(i) for i in range(numscens)]
    testtree = _ScenTree(branching_factors, scennames)
    print("nonleaves:")
    for nd in testtree.nonleaves:
        print("   ", nd.name)
    print("NonLeafTerminals:")
    for nd in testtree.NonLeafTerminals:
        print("   ", nd.name)
    n_proc = 8
    sntr, slices, ranks_per_scenario = testtree.scen_names_to_ranks(n_proc)
    print("map:")
    for ndn,v in sntr.items():
        print(ndn, v)
    print(f"slices: {slices}")
    check4losses(numscens, branching_factors, sntr, slices, ranks_per_scenario)
