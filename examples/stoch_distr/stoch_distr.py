###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Network Flow - various formulations
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import mpisppy.tests.examples.distr.distr_data as distr_data
import numpy as np

# In this file, we create a (linear) inter-region minimal cost distribution problem.
# Our data, gives the constraints inside each in region in region_dict_creator
# and amongst the different regions in inter_region_dict_creator
# The data slightly differs depending on the number of regions (num_scens) which is created for 2, 3 or 4 regions

# Note: regions in our model will be represented in mpi-sppy by scenarios and to ensure the inter-region constraints
#       we will impose the inter-region arcs to be consensus-variables. They will be represented as non anticipative variables in mpi-sppy


def inter_arcs_adder(region_dict, inter_region_dict):
    """This function adds to the region_dict the inter-region arcs 

    Args:
        region_dict (dict): dictionary for the current scenario \n
        inter_region_dict (dict): dictionary of the inter-region relations

    Returns:
        local_dict (dict): 
            This dictionary copies region_dict, completes the already existing fields of 
            region_dict to represent the inter-region arcs and their capcities and costs
    """
    ### Note: The cost of the arc is here chosen to be split equally between the source region and target region

    # region_dict is renamed as local_dict because it no longer contains only information about the region
    # but also contains local information that can be linked to the other scenarios (with inter-region arcs)
    local_dict = region_dict
    for inter_arc in inter_region_dict["arcs"]:
        source,target = inter_arc
        region_source,node_source = source
        region_target,node_target = target

        if region_source == region_dict["name"] or region_target == region_dict["name"]:
            arc = node_source, node_target
            local_dict["arcs"].append(arc)
            local_dict["flow capacities"][arc] = inter_region_dict["capacities"][inter_arc]
            local_dict["flow costs"][arc] = inter_region_dict["costs"][inter_arc]/2
    #print(f"scenario name = {region_dict['name']} \n {local_dict=}  ")
    return local_dict


###Creates the model when local_dict is given, local_dict depends on the subproblem
def min_cost_distr_problem(local_dict, cfg, stoch_scenario_name, sense=pyo.minimize, max_revenue=None):
    """ Create an arcs formulation of network flow for the region and stochastic scenario considered.

    Args:
        local_dict (dict): dictionary representing a region including the inter region arcs \n
        cfg (): config argument used here for the random parameters \n
        stoch_scenario_name(str): name of the stochastic scenario. In each region, the model is scenario dependant \n
        sense (=pyo.minimize): we aim to minimize the cost, this should always be minimize \n
        max_revenue (float, opt): higher than all the possible revenues, this value allows to add a penalty slack making sure that the optimal value of the new model is obtained with a 0 slack

    Returns:
        model (Pyomo ConcreteModel) : the instantiated model
    """
    scennum = sputils.extract_num(stoch_scenario_name)

    # Assert sense == pyo.minimize, "sense should be equal to pyo.minimize"
    # First, make the special In, Out arc lists for each node
    arcsout = {n: list() for n in local_dict["nodes"]}
    arcsin = {n: list() for n in local_dict["nodes"]}
    for a in local_dict["arcs"]:
        if a[0] in local_dict["nodes"]:
            arcsout[a[0]].append(a)
        if a[1] in local_dict["nodes"]:
            arcsin[a[1]].append(a)

    model = pyo.ConcreteModel(name='MinCostFlowArcs')
    def flowBounds_rule(model, i,j):
        return (0, local_dict["flow capacities"][(i,j)])
    model.flow = pyo.Var(local_dict["arcs"], bounds=flowBounds_rule)  # x

    def slackBounds_rule(model, n):
        if n in local_dict["factory nodes"]:
            return (0, local_dict["supply"][n])
        elif n in local_dict["buyer nodes"]:
            return (local_dict["supply"][n], 0)
        elif n in local_dict["distribution center nodes"]:
            if cfg.ensure_xhat_feas:
                # Should be (0,0) but to avoid infeasibility we add a negative slack variable
                return (None, 0)
            else:
                return (0,0)
        else:
            raise ValueError(f"unknown node type for node {n}")
    
    model.y = pyo.Var(local_dict["nodes"], bounds=slackBounds_rule)

    def inventoryBounds_rule(model, n):
        return (0,local_dict["supply"][n]) 
        ### We can store everything to make sure it is feasible, but it will never be interesting

    model.inventory = pyo.Var(local_dict["factory nodes"], bounds=inventoryBounds_rule)

    model.FirstStageCost = pyo.Expression(expr=\
                    sum(local_dict["production costs"][n]*(local_dict["supply"][n]-model.y[n]) for n in local_dict["factory nodes"]))
    
    if cfg.ensure_xhat_feas:
        # too big penalty to allow the stack to be non-zero
        model.SecondStageCost = pyo.Expression(expr=\
                        sum(local_dict["flow costs"][a]*model.flow[a] for a in local_dict["arcs"]) \
                        + sum(local_dict["revenues"][n]*(local_dict["supply"][n]-model.y[n]) for n in local_dict["buyer nodes"]) \
                        + sum(2*max_revenue*(-model.y[n]) for n in local_dict["distribution center nodes"]) \
                        - sum(local_dict["production costs"][n]/2*model.inventory[n] for n in local_dict["factory nodes"]))
    else:
        model.SecondStageCost = pyo.Expression(expr=\
                        sum(local_dict["flow costs"][a]*model.flow[a] for a in local_dict["arcs"]) \
                        + sum(local_dict["revenues"][n]*(local_dict["supply"][n]-model.y[n]) for n in local_dict["buyer nodes"]) \
                        - sum(local_dict["production costs"][n]/2*model.inventory[n] for n in local_dict["factory nodes"])
                            )

    model.MinCost = pyo.Objective(expr=model.FirstStageCost + model.SecondStageCost, sense=sense)
    
    def FlowBalance_rule(m, n):
        if n in local_dict["factory nodes"]:
            # We generate pseudo randomly the loss on each factory node
            node_type, region_num, count = distr_data.parse_node_name(n)
            node_num = distr_data._node_num(cfg.mnpr, node_type, region_num, count)
            np.random.seed(node_num+cfg.initial_seed+(scennum+1)*2**20) #2**20 avoids the correlation with the scalable example data
            return sum(m.flow[a] for a in arcsout[n])\
            - sum(m.flow[a] for a in arcsin[n])\
            + m.inventory[n] \
            == (local_dict["supply"][n] - m.y[n]) * min(1,max(0,1-np.random.normal(cfg.spm,cfg.cv)/100)) # We add the loss
        else:
            return sum(m.flow[a] for a in arcsout[n])\
            - sum(m.flow[a] for a in arcsin[n])\
            + m.y[n] == local_dict["supply"][n]
    model.FlowBalance= pyo.Constraint(local_dict["nodes"], rule=FlowBalance_rule)

    return model


###Functions required in other files, which constructions are specific to the problem

###Creates the scenario
def scenario_creator(admm_stoch_subproblem_scenario_name, inter_region_dict=None, cfg=None, data_params=None, all_nodes_dict=None):
    """Creates the model, which should include the consensus variables. \n
    However, this function shouldn't attach the consensus variables for the admm subproblems as it is done in stoch_admmWrapper. 
    Therefore, only the stochastic tree as it would be represented without the decomposition needs to be created.

    Args:
        admm_stoch_subproblem_scenario_name (str): the name given to the admm problem for the stochastic scenario. \n
        inter_region_dict (dict): contains all the links between the regions
        num_scens (int): number of scenarios (regions). Useful to create the corresponding inter-region dictionary \n
        other parameters are key word arguments for scenario_creator

    Returns:
        Pyomo ConcreteModel: the instantiated model
    """
    assert (inter_region_dict is not None)
    assert (cfg is not None)
    # Use the package default split so this module does not have to
    # ship its own combining/split inverse pair.  See
    # mpisppy.utils.stoch_admmWrapper for the default convention.
    from mpisppy.utils.stoch_admmWrapper import (
        default_split_admm_stoch_subproblem_scenario_name,
    )
    admm_subproblem_name, stoch_scenario_name = (
        default_split_admm_stoch_subproblem_scenario_name(
            admm_stoch_subproblem_scenario_name))
    if cfg.scalable:
        assert (data_params is not None)
        assert (all_nodes_dict is not None)
        region_dict = distr_data.scalable_region_dict_creator(admm_subproblem_name, all_nodes_dict=all_nodes_dict, cfg=cfg, data_params=data_params)
    else:
        region_dict = distr_data.region_dict_creator(admm_subproblem_name)

    # Adding inter region arcs nodes and associated features
    local_dict = inter_arcs_adder(region_dict, inter_region_dict)
    # Generating the model
    model = min_cost_distr_problem(local_dict, cfg, stoch_scenario_name, max_revenue=data_params["max revenue"])

    # Stash the original first-stage variable list so the module-level
    # first_stage_varlist hook can retrieve it.  Stoch_AdmmWrapper will
    # call sputils.attach_root_node on our behalf using the hooks.
    model._first_stage_vars = [model.y[n] for n in local_dict["factory nodes"]]

    return model


def first_stage_cost(scenario):
    """Return the original problem's first-stage cost expression.

    Used by Stoch_AdmmWrapper to attach the root node automatically;
    the user no longer needs to call sputils.attach_root_node in
    scenario_creator.  See doc/src/generic_admm.rst.
    """
    return scenario.FirstStageCost


def first_stage_varlist(scenario):
    """Return the original problem's first-stage variables (factory
    production decisions for stoch_distr).  Companion to first_stage_cost.
    """
    return scenario._first_stage_vars


def scenario_denouement(rank, admm_stoch_subproblem_scenario_name, scenario, eps=10**(-6)):
    """For each admm stochastic scenario subproblem prints its name and the final variable values

    Args:
        rank (int): rank in which the scenario is placed
        admm_stoch_subproblem_scenario_name (str): name of the admm stochastic scenario subproblem
        scenario (Pyomo ConcreteModel): the instantiated model
        eps (float, opt): ensures that the dummy slack variables introduced have small values
    """
    #print(f"slack values for the distribution centers for {admm_stoch_subproblem_scenario_name=} at {rank=}")
    for var in scenario.y:
        if 'DC' in var:
            if (abs(scenario.y[var].value) > eps):
                print(f"The penalty slack {scenario.y[var].name} is too big, its absolute value is {abs(scenario.y[var].value)}")
            #assert (abs(scenario.y[var].value) < eps), "the penalty slack is too big"
            #scenario.y[var].pprint()
        if 'F' in var:
            if (scenario.inventory[var].value > 10*eps):
                print(f"In {rank=} for {admm_stoch_subproblem_scenario_name}, the inventory {scenario.inventory[var].name} is big, its value is {scenario.inventory[var].value}. Although it is rare after convergence, it makes sense if in a scenario it is really interesting to produce.")
                scenario.inventory[var].pprint()
            #assert (abs(scenario.inventory[var].value) < eps), "the inventory is too big"
            #scenario.inventory[var].pprint()
    return
    print(f"flow values for {admm_stoch_subproblem_scenario_name=} at {rank=}")
    scenario.flow.pprint()
    print(f"slack values for {admm_stoch_subproblem_scenario_name=} at {rank=}")
    scenario.y.pprint()


def consensus_vars_creator(admm_subproblem_names, stoch_scenario_name, inter_region_dict=None, cfg=None, data_params=None, all_nodes_dict=None):
    """The following function creates the consensus variables dictionary thanks to the inter-region dictionary. \n
    This dictionary has redundant information, but is useful for admmWrapper.

    Args:
        admm_subproblem_names (list of str): name of the admm subproblems (regions) \n
        stoch_scenario_name (str): name of any stochastic_scenario, it is only used 
        in this example to access the non anticipative variables (which are common to 
        every stochastic scenario) and their stage. \n
        kwargs (opt): necessary in this example to call the scenario creator
    
    Returns:
        dict: dictionary which keys are the subproblems and values are the list of 
        pairs (consensus_variable_name (str), stage (int)).
    """
    consensus_vars = {}
    for inter_arc in inter_region_dict["arcs"]:
        source,target = inter_arc
        region_source,node_source = source
        region_target,node_target = target
        arc = node_source, node_target
        vstr = f"flow[{arc}]" #variable name as string, y is the slack

        #adds inter_region_arcs in the source region
        if region_source not in consensus_vars: #initiates consensus_vars[region_source]
            consensus_vars[region_source] = list()
        consensus_vars[region_source].append((vstr,2))

        #adds inter_region_arcs in the target region
        if region_target not in consensus_vars: #initiates consensus_vars[region_target]
            consensus_vars[region_target] = list()
        consensus_vars[region_target].append((vstr,2))

    # With the scalable example some regions may have no consensus_vars
    # This is not realistic because it would imply that the region is not linked
    for admm_subproblem_name in admm_subproblem_names:
        if admm_subproblem_name not in consensus_vars:
            print(f"WARNING: {admm_subproblem_name} has no consensus_vars")
            consensus_vars[admm_subproblem_name] = list()
    # Each ADMM subproblem's first-stage Vars are added to its
    # consensus list by the wrapper (Stoch_AdmmWrapper / AdmmBundler)
    # at construction time, driven by the first_stage_varlist hook.
    return consensus_vars


def stoch_scenario_names_creator(cfg):
    """Creates the name of every stochastic scenario.

    Args:
        cfg: config object

    Returns:
        list (str): the list of stochastic scenario names
    """
    return [f"StochasticScenario{i+1}" for i in range(cfg.num_stoch_scens)]


def admm_subproblem_names_creator(cfg):
    """Creates the name of every admm subproblem.

    Args:
        cfg: config object

    Returns:
        list (str): the list of admm subproblem names
    """
    return [f"Region{i+1}" for i in range(cfg.num_admm_subproblems)]


def kw_creator(cfg):
    """
    Args:
        cfg (config): specifications for the problem given on the command line

    Returns:
        dict (str): the kwargs that are used in stoch_distr.scenario_creator
    """
    if cfg.scalable:
        import json
        json_file_path = cfg.get("json_file_path", ifmissing="../distr/data_params.json")
        with open(json_file_path, 'r') as file:
            data_params = json.load(file)
        # In distr_data num_admm_subproblems is called num_scens
        if cfg.get("num_scens") is None:
            cfg.add_to_config("num_scens",
                      description="num admm subproblems",
                      domain=int,
                      default=cfg.num_admm_subproblems)
        all_nodes_dict = distr_data.all_nodes_dict_creator(cfg, data_params)
        all_DC_nodes = [DC_node for region in all_nodes_dict
                        for DC_node in all_nodes_dict[region]["distribution center nodes"]]
        inter_region_dict = distr_data.scalable_inter_region_dict_creator(all_DC_nodes, cfg, data_params)
    else:
        inter_region_dict = distr_data.inter_region_dict_creator(num_scens=cfg.num_admm_subproblems)
        all_nodes_dict = None
        data_params = {"max revenue": 1200}

    kwargs = {
        "all_nodes_dict" : all_nodes_dict,
        "inter_region_dict" : inter_region_dict,
        "cfg" : cfg,
        "data_params" : data_params,
              }
    return kwargs


def inparser_adder(cfg):
    """Adding to the config argument, specific elements to our problem:
    elements used for random number generation + possibility to scale.

    num_admm_subproblems / num_stoch_scens are registered by
    mpisppy.generic.admm.admm_args; setup_stoch_admm checks that they
    were actually set on the command line.

    Args:
        cfg (config): specifications for the problem given on the command line
    """
    ### For the pseudo-random scenarios
    cfg.add_to_config("spm",
                      description="mean percentage of scrap loss at the production",
                      domain=float,
                      default=5)
    
    cfg.add_to_config("cv",
                      description="coefficient of variation of the loss at the production",
                      domain=float,
                      default=20)
    
    cfg.add_to_config("initial_seed",
                      description="initial seed for generating the loss",
                      domain=int,
                      default=0)
    
    ### For the scalable example
    cfg.add_to_config("scalable",
                      description="generate pseudo-random data parameterized by --mnpr instead of using the hardwired 2/3/4-region datasets",
                      domain=bool,
                      default=False)

    cfg.add_to_config("mnpr",
                      description="max number of nodes per region and per type",
                      domain=int,
                      default=4)
    
    cfg.add_to_config("ensure_xhat_feas",
                      description="adds slacks with high costs to ensure the feasibility of xhat yet maintaining the optimal",
                      domain=bool,
                      default=False)
