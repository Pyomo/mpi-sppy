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
import examples.distr.distr_data as distr_data
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
    admm_subproblem_name, stoch_scenario_name = split_admm_stoch_subproblem_scenario_name(admm_stoch_subproblem_scenario_name)
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

    sputils.attach_root_node(model, model.FirstStageCost, [model.y[n] for n in  local_dict["factory nodes"]])
    
    return model


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
    
    # now add the parents. It doesn't depend on the stochastic scenario so we chose one and
    # then we go through the models (created by scenario creator) for all the admm_stoch_subproblem_scenario 
    # which have this scenario as an ancestor (parent) in the tree
    for admm_subproblem_name in admm_subproblem_names:
        admm_stoch_subproblem_scenario_name = combining_names(admm_subproblem_name,stoch_scenario_name)
        model = scenario_creator(admm_stoch_subproblem_scenario_name, inter_region_dict=inter_region_dict, cfg=cfg, data_params=data_params, all_nodes_dict=all_nodes_dict)
        for node in model._mpisppy_node_list:
            for var in node.nonant_list:
                if var.name not in consensus_vars[admm_subproblem_name]:
                    consensus_vars[admm_subproblem_name].append((var.name, node.stage))
    return consensus_vars


def stoch_scenario_names_creator(num_stoch_scens):
    """Creates the name of every stochastic scenario.

    Args:
        num_stoch_scens (int): number of stochastic scenarios

    Returns:
        list (str): the list of stochastic scenario names
    """
    return [f"StochasticScenario{i+1}" for i in range(num_stoch_scens)]


def admm_subproblem_names_creator(num_admm_subproblems):
    """Creates the name of every admm subproblem.

    Args:
        num_subproblems (int): number of admm subproblems

    Returns:
        list (str): the list of admm subproblem names
    """
    return [f"Region{i+1}" for i in range(num_admm_subproblems)]


def combining_names(admm_subproblem_name,stoch_scenario_name):
    # Used to create the admm_stoch_subproblem_scenario_name
    return f"ADMM_STOCH_{admm_subproblem_name}_{stoch_scenario_name}"


def admm_stoch_subproblem_scenario_names_creator(admm_subproblem_names,stoch_scenario_names):
    """ Creates the list of the admm stochastic subproblem scenarios, which are the admm subproblems given a scenario

    Args:
        admm_subproblem_names (list of str): names of the admm subproblem
        stoch_scenario_names (list of str): names of the stochastic scenario

    Returns:
        list of str: name of the admm stochastic subproblem scenarios
    """
    ### The order is important, we may want all the subproblems to appear consecutively in a scenario
    return [combining_names(admm_subproblem_name,stoch_scenario_name) \
            for stoch_scenario_name in stoch_scenario_names \
                for admm_subproblem_name in admm_subproblem_names ]


def split_admm_stoch_subproblem_scenario_name(admm_stoch_subproblem_scenario_name):
    """ Returns the admm_subproblem_name and the stoch_scenario_name given an admm_stoch_subproblem_scenario_name.
    This function, specific to the problem, is the reciprocal function of ``combining_names`` which creates the 
    admm_stoch_subproblem_scenario_name given the admm_subproblem_name and the stoch_scenario_name.

    Args:
        admm_stoch_subproblem_scenario_name (str)

    Returns:
        (str,str): admm_subproblem_name, stoch_scenario_name
    """
    # Method specific to our example and because the admm_subproblem_name and stoch_scenario_name don't include "_"
    splitted = admm_stoch_subproblem_scenario_name.split('_')
    assert (len(splitted) == 4), "no underscore should be attached to admm_subproblem_name nor stoch_scenario_name"
    admm_subproblem_name = splitted[2]
    stoch_scenario_name = splitted[3]
    return admm_subproblem_name, stoch_scenario_name


def kw_creator(all_nodes_dict, cfg, inter_region_dict, data_params):
    """
    Args:
        cfg (config): specifications for the problem given on the command line

    Returns:
        dict (str): the kwargs that are used in distr.scenario_creator, which are included in cfg.
    """
    kwargs = {
        "all_nodes_dict" : all_nodes_dict,
        "inter_region_dict" : inter_region_dict,
        "cfg" : cfg,
        "data_params" : data_params,
              }
    return kwargs


def inparser_adder(cfg):
    """Adding to the config argument, specific elements to our problems. In this case the numbers of stochastic scenarios 
    and admm subproblems which are required + elements used for random number generation + possibility to scale

    Args:
        cfg (config): specifications for the problem given on the command line
    """
    cfg.add_to_config(
            "num_stoch_scens",
            description="Number of stochastic scenarios (default None)",
            domain=int,
            default=None,
            argparse_args = {"required": True}
        )
    
    cfg.add_to_config(
            "num_admm_subproblems",
            description="Number of admm subproblems (regions)",
            domain=int,
            default=None,
            argparse_args = {"required": True}
        )

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
                      description="decides whether a scalable model is used",
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