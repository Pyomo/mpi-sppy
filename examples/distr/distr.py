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
import distr_data
import time

# In this file, we create a (linear) inter-region minimal cost distribution problem.
# Our data, gives the constraints inside each in region in region_dict_creator
# and amongst the different regions in inter_region_dict_creator
# The data slightly differs depending on the number of regions (num_scens) which is created for 2, 3 or 4 regions

# Note: regions in our model will be represented in mpi-sppy by scenarios and to ensure the inter-region constraints
#       we will impose the inter-region arcs to be consensus-variables. They will be represented as non-ants in mpi-sppy


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

###Creates the model when local_dict is given
def min_cost_distr_problem(local_dict, cfg, sense=pyo.minimize, max_revenue=None):
    """ Create an arcs formulation of network flow

    Args:
        local_dict (dict): dictionary representing a region including the inter region arcs \n
        sense (=pyo.minimize): we aim to minimize the cost, this should always be minimize

    Returns:
        model (Pyomo ConcreteModel) : the instantiated model
    """
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

    if cfg.ensure_xhat_feas:
        # too big penalty to allow the stack to be non-zero
        model.MinCost = pyo.Objective(expr=\
                    sum(local_dict["flow costs"][a]*model.flow[a] for a in local_dict["arcs"]) \
                    + sum(local_dict["production costs"][n]*(local_dict["supply"][n]-model.y[n]) for n in local_dict["factory nodes"]) \
                    + sum(local_dict["revenues"][n]*(local_dict["supply"][n]-model.y[n]) for n in local_dict["buyer nodes"]) \
                    + sum(2*max_revenue*(-model.y[n]) for n in local_dict["distribution center nodes"]) ,
                      sense=sense)
    
    else:
        model.MinCost = pyo.Objective(expr=\
                        sum(local_dict["flow costs"][a]*model.flow[a] for a in local_dict["arcs"]) \
                        + sum(local_dict["production costs"][n]*(local_dict["supply"][n]-model.y[n]) for n in local_dict["factory nodes"]) \
                        + sum(local_dict["revenues"][n]*(local_dict["supply"][n]-model.y[n]) for n in local_dict["buyer nodes"]) ,
                        sense=sense)
    
    def FlowBalance_rule(m, n):
        return sum(m.flow[a] for a in arcsout[n])\
        - sum(m.flow[a] for a in arcsin[n])\
        + m.y[n] == local_dict["supply"][n]
    model.FlowBalance = pyo.Constraint(local_dict["nodes"], rule=FlowBalance_rule)

    return model


###Creates the scenario
def scenario_creator(scenario_name, inter_region_dict=None, cfg=None, data_params=None, all_nodes_dict=None):
    """Creates the model, which should include the consensus variables. \n
    However, this function shouldn't attach the consensus variables to root nodes, as it is done in admmWrapper.

    Args:
        scenario_name (str): the name of the scenario that will be created. Here is of the shape f"Region{i}" with 1<=i<=num_scens \n
        num_scens (int): number of scenarios (regions). Useful to create the corresponding inter-region dictionary

    Returns:
        Pyomo ConcreteModel: the instantiated model
    """        
    assert (inter_region_dict is not None)
    assert (cfg is not None)
    if cfg.scalable:
        assert (data_params is not None)
        assert (all_nodes_dict is not None)
        region_creation_starting_time = time.time()
        region_dict = distr_data.scalable_region_dict_creator(scenario_name, all_nodes_dict=all_nodes_dict, cfg=cfg, data_params=data_params)
        region_creation_end_time = time.time()
        print(f"time for creating region {scenario_name}: {region_creation_end_time - region_creation_starting_time}")   
    else:
        region_dict = distr_data.region_dict_creator(scenario_name)
    # Adding inter region arcs nodes and associated features
    local_dict = inter_arcs_adder(region_dict, inter_region_dict)
    # Generating the model
    model = min_cost_distr_problem(local_dict, cfg, max_revenue=data_params["max revenue"])

    #varlist = list()
    #sputils.attach_root_node(model, model.MinCost, varlist)    
    
    return model


###Functions required in other files, which constructions are specific to the problem

def scenario_denouement(rank, scenario_name, scenario, eps=10**(-6)):
    """for each scenario prints its name and the final variable values

    Args:
        rank (int): not used here, but could be helpful to know the location
        scenario_name (str): name of the scenario
        scenario (Pyomo ConcreteModel): the instantiated model
        eps (float, opt): ensures that the dummy slack variables introduced have small values
    """
    for var in scenario.y:
        if 'DC' in var:
            if (abs(scenario.y[var].value) > eps):
                print(f"The penalty slack {scenario.y[var].name} is too big, its absolute value is {abs(scenario.y[var].value)}")
    return
    print(f"flow values for {scenario_name}")
    scenario.flow.pprint()
    print(f"slack values for {scenario_name}")
    scenario.y.pprint()
    pass


def consensus_vars_creator(num_scens, inter_region_dict, all_scenario_names):
    """The following function creates the consensus_vars dictionary thanks to the inter-region dictionary. \n
    This dictionary has redundant information, but is useful for admmWrapper.

    Args:
        num_scens (int): select the number of scenarios (regions) wanted
    
    Returns:
        dict: dictionary which keys are the regions and values are the list of consensus variables 
        present in the region
    """
    # Due to the small size of inter_region_dict, it is not given as argument but rather created. 
    consensus_vars = {}
    for inter_arc in inter_region_dict["arcs"]:
        source,target = inter_arc
        region_source,node_source = source
        region_target,node_target = target
        arc = node_source, node_target
        vstr = f"flow[{arc}]" #variable name as string

        #adds inter region arcs in the source region
        if region_source not in consensus_vars: #initiates consensus_vars[region_source]
            consensus_vars[region_source] = list()
        consensus_vars[region_source].append(vstr)

        #adds inter region arcs in the target region
        if region_target not in consensus_vars: #initiates consensus_vars[region_target]
            consensus_vars[region_target] = list()
        consensus_vars[region_target].append(vstr)
    for region in all_scenario_names:
        if region not in consensus_vars:
            print(f"WARNING: {region} has no consensus_vars")
            consensus_vars[region] = list()
    return consensus_vars


def scenario_names_creator(num_scens):
    """Creates the name of every scenario.

    Args:
        num_scens (int): number of scenarios

    Returns:
        list (str): the list of names
    """
    return [f"Region{i+1}" for i in range(num_scens)]


def kw_creator(all_nodes_dict, cfg, inter_region_dict, data_params):
    """
    Args:
        cfg (config): specifications for the problem. We only look at the number of scenarios

    Returns:
        dict (str): the kwargs that are used in distr.scenario_creator, here {"num_scens": num_scens}
    """
    kwargs = {
        "all_nodes_dict" : all_nodes_dict,
        "inter_region_dict" : inter_region_dict,
        "cfg" : cfg,
        "data_params" : data_params,
              }
    return kwargs


def inparser_adder(cfg):
    #requires the user to give the number of scenarios
    cfg.num_scens_required()

    cfg.add_to_config("scalable",
                      description="decides whether a sclale model is used",
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