###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# This file is used in the tests and should not be modified!

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils

# In this file, we create a (linear) inter-region minimal cost distribution problem.
# Our data, gives the constraints inside each in region in region_dict_creator
# and amongst the different regions in inter_region_dict_creator
# The data slightly differs depending on the number of regions (num_scens) which is created for 2, 3 or 4 regions

# Note: regions in our model will be represented in mpi-sppy by scenarios and to ensure the inter-region constraints
#       we will use dummy-nodes, which will be represented in mpi-sppy by non-anticipative variables
#       The following association of terms are made: regions = scenarios, and dummy-nodes = nonants = consensus-vars

### The following functions create the data 

def inter_region_dict_creator(num_scens):
    """Creates the oriented arcs between the regions, with their capacities and costs. \n
    This dictionary represents the inter-region constraints and flows. It indicates where to add dummy nodes.

    Args:
        num_scens (int): select the number of scenarios (regions) wanted

    Returns:
        dict: 
            Each arc is presented as a pair (source, target) with source and target containing (scenario_name, node_name) \n
            The arcs are used as keys for the dictionaries of costs and capacities
    """
    inter_region_dict={}

    if num_scens == 2:
        inter_region_dict["arcs"]=[(("Region1","DC1"),("Region2","DC2"))]
        inter_region_dict["costs"]={(("Region1","DC1"),("Region2","DC2")): 100}
        inter_region_dict["capacities"]={(("Region1","DC1"),("Region2","DC2")): 70}

    elif num_scens == 3:
        inter_region_dict["arcs"] = [(("Region1","DC1"),("Region2","DC2")),(("Region2","DC2"),("Region1","DC1")),\
                                    (("Region1","DC1"),("Region3","DC3_1")),(("Region3","DC3_1"),("Region1","DC1")),\
                                    (("Region2","DC2"),("Region3","DC3_2")),(("Region3","DC3_2"),("Region2","DC2")),\
                                        ]
        inter_region_dict["costs"] = {(("Region1","DC1"),("Region2","DC2")): 100, (("Region2","DC2"),("Region1","DC1")): 50,\
                                    (("Region1","DC1"),("Region3","DC3_1")): 200, (("Region3","DC3_1"),("Region1","DC1")): 200,\
                                    (("Region2","DC2"),("Region3","DC3_2")): 200, (("Region3","DC3_2"),("Region2","DC2")): 200,\
                                        }
        inter_region_dict["capacities"] = {(("Region1","DC1"),("Region2","DC2")): 70, (("Region2","DC2"),("Region1","DC1")): 100,\
                                    (("Region1","DC1"),("Region3","DC3_1")): 50, (("Region3","DC3_1"),("Region1","DC1")): 50,\
                                    (("Region2","DC2"),("Region3","DC3_2")): 50, (("Region3","DC3_2"),("Region2","DC2")): 50,\
                                        }
    
    elif num_scens == 4:
        inter_region_dict["arcs"] = [(("Region1","DC1"),("Region2","DC2")),(("Region2","DC2"),("Region1","DC1")),\
                                 (("Region1","DC1"),("Region3","DC3_1")),(("Region3","DC3_1"),("Region1","DC1")),\
                                 (("Region2","DC2"),("Region3","DC3_2")),(("Region3","DC3_2"),("Region2","DC2")),\
                                 (("Region1","DC1"),("Region4","DC4")),(("Region4","DC4"),("Region1","DC1")),\
                                 (("Region4","DC4"),("Region2","DC2")),(("Region2","DC2"),("Region4","DC4")),\
                                    ]
    inter_region_dict["costs"] = {(("Region1","DC1"),("Region2","DC2")): 100, (("Region2","DC2"),("Region1","DC1")): 50,\
                                  (("Region1","DC1"),("Region3","DC3_1")): 200, (("Region3","DC3_1"),("Region1","DC1")): 200,\
                                  (("Region2","DC2"),("Region3","DC3_2")): 200, (("Region3","DC3_2"),("Region2","DC2")): 200,\
                                  (("Region1","DC1"),("Region4","DC4")): 30, (("Region4","DC4"),("Region1","DC1")): 50,\
                                  (("Region4","DC4"),("Region2","DC2")): 100, (("Region2","DC2"),("Region4","DC4")): 70,\
                                    }
    inter_region_dict["capacities"] = {(("Region1","DC1"),("Region2","DC2")): 70, (("Region2","DC2"),("Region1","DC1")): 100,\
                                  (("Region1","DC1"),("Region3","DC3_1")): 50, (("Region3","DC3_1"),("Region1","DC1")): 50,\
                                  (("Region2","DC2"),("Region3","DC3_2")): 50, (("Region3","DC3_2"),("Region2","DC2")): 50,\
                                  (("Region1","DC1"),("Region4","DC4")): 100, (("Region4","DC4"),("Region1","DC1")): 60,\
                                  (("Region4","DC4"),("Region2","DC2")): 20, (("Region2","DC2"),("Region4","DC4")): 40,\
                                    }
    
    return inter_region_dict


def dummy_nodes_generator(region_dict, inter_region_dict):
    """This function creates a new dictionary ``local_dict similar`` to ``region_dict`` with the dummy nodes and their constraints

    Args:
        region_dict (dict): dictionary for the current scenario \n
        inter_region_dict (dict): dictionary of the inter-region relations

    Returns:
        local_dict (dict): 
            This dictionary copies region_dict, completes the already existing fields of 
            region_dict to represent the dummy nodes, and adds the following keys:\n
            dummy nodes source (resp. target): the list of dummy nodes for which the source (resp. target) 
            is in the considered region. \n
            dummy nodes source (resp. target) slack bounds: dictionary on dummy nodes source (resp. target)
    """
    ### Note: The cost of the arc is here chosen to be split equally between the source region and target region

    # region_dict is renamed as local_dict because it no longer contains only information about the region
    # but also contains local information that can be linked to the other scenarios (with dummy nodes)
    local_dict = region_dict
    local_dict["dummy nodes source"] = list()
    local_dict["dummy nodes target"] = list()
    local_dict["dummy nodes source slack bounds"] = {}
    local_dict["dummy nodes target slack bounds"] = {}
    for arc in inter_region_dict["arcs"]:
        source,target = arc
        region_source,node_source = source
        region_target,node_target = target

        if region_source == region_dict["name"]:
            dummy_node=node_source+node_target
            local_dict["nodes"].append(dummy_node)
            local_dict["supply"][dummy_node] = 0
            local_dict["dummy nodes source"].append(dummy_node)
            local_dict["arcs"].append((node_source, dummy_node))
            local_dict["flow costs"][(node_source, dummy_node)] = inter_region_dict["costs"][(source, target)]/2 #should be adapted to model
            local_dict["flow capacities"][(node_source, dummy_node)] = inter_region_dict["capacities"][(source, target)]
            local_dict["dummy nodes source slack bounds"][dummy_node] = inter_region_dict["capacities"][(source, target)]

        if region_target == local_dict["name"]:
            dummy_node = node_source + node_target
            local_dict["nodes"].append(dummy_node)
            local_dict["supply"][dummy_node] = 0
            local_dict["dummy nodes target"].append(dummy_node)
            local_dict["arcs"].append((dummy_node, node_target))
            local_dict["flow costs"][(dummy_node, node_target)] = inter_region_dict["costs"][(source, target)]/2 #should be adapted to model
            local_dict["flow capacities"][(dummy_node, node_target)] = inter_region_dict["capacities"][(source, target)]
            local_dict["dummy nodes target slack bounds"][dummy_node] = inter_region_dict["capacities"][(source, target)]
    return local_dict

def _is_partition(L, *lists):
    # Step 1: Verify that the union of all sublists contains all elements of L
    if set(L) != set().union(*lists):
        return False
    
    # Step 2: Ensure each element in L appears in exactly one sublist
    for item in L:
        count = 0
        for sublist in lists:
            if item in sublist:
                count += 1
        if count != 1:
            return False
    
    return True

def region_dict_creator(scenario_name):
    """ Create a scenario for the inter-region max profit distribution example.

        The convention for node names is: 
            Symbol + number of the region (+ _ + number of the example if needed), \n
            with symbols DC for distribution centers, F for factory nodes, B for buyer nodes. \n
            For instance: F3_1 is the 1st factory node of region 3. \n

    Args:
        scenario_name (str):
            Name of the scenario to construct.

    Returns:
        region_dict (dict): contains all the information in the given region to create the model. It is composed of:\n
            "nodes" (list of str): all the nodes. The following subsets are also nodes: \n
            "factory nodes", "buyer nodes", "distribution center nodes", \n
            "arcs" (list of 2 tuples of str) : (node, node) pairs\n
            "supply" (dict[n] of float): supply; keys are nodes (negative for demand)\n
            "production costs" (dict of float): at each factory node\n
            "revenues" (dict of float): at each buyer node \n
            "flow costs" (dict[a] of float) : costs per unit flow on each arc \n
            "flow capacities" (dict[a] of floats) : upper bound capacities of each arc \n
    """
    if scenario_name == "Region1" :
        # Creates data for Region1
        region_dict={"name": "Region1"}
        region_dict["nodes"] = ["F1_1", "F1_2", "DC1", "B1_1", "B1_2"]
        region_dict["factory nodes"] = ["F1_1","F1_2"]
        region_dict["buyer nodes"] = ["B1_1","B1_2"]
        region_dict["distribution center nodes"]= ["DC1"]
        region_dict["supply"] = {"F1_1": 80, "F1_2": 70, "B1_1": -60, "B1_2": -90, "DC1": 0}
        region_dict["arcs"] = [("F1_1","DC1"), ("F1_2","DC1"), ("DC1","B1_1"),
            ("DC1","B1_2"), ("F1_1","B1_1"), ("F1_2","B1_2")]

        region_dict["production costs"] = {"F1_1": 50, "F1_2": 80}
        region_dict["revenues"] = {"B1_1": 800, "B1_2": 900}
        # most capacities are 50, so start there and then override
        region_dict["flow capacities"] = {a: 50 for a in region_dict["arcs"]}
        region_dict["flow capacities"][("F1_1","B1_1")] = None
        region_dict["flow capacities"][("F1_2","B1_2")] = None
        region_dict["flow costs"] = {("F1_1","DC1"): 300, ("F1_2","DC1"): 500, ("DC1","B1_1"): 200,
            ("DC1","B1_2"): 400, ("F1_1","B1_1"): 700,  ("F1_2","B1_2"): 1000}
        
    elif scenario_name=="Region2":
        # Creates data for Region2
        region_dict={"name": "Region2"}
        region_dict["nodes"] = ["DC2", "B2_1", "B2_2", "B2_3"]
        region_dict["factory nodes"] = list()
        region_dict["buyer nodes"] = ["B2_1","B2_2","B2_3"]
        region_dict["distribution center nodes"]= ["DC2"]
        region_dict["supply"] = {"B2_1": -200, "B2_2": -150, "B2_3": -100, "DC2": 0}
        region_dict["arcs"] = [("DC2","B2_1"), ("DC2","B2_2"), ("DC2","B2_3")]

        region_dict["production costs"] = {}
        region_dict["revenues"] = {"B2_1": 900, "B2_2": 800, "B2_3":1200}
        region_dict["flow capacities"] = {("DC2","B2_1"): 200, ("DC2","B2_2"): 150, ("DC2","B2_3"): 100}
        region_dict["flow costs"] = {("DC2","B2_1"): 100, ("DC2","B2_2"): 200, ("DC2","B2_3"): 300}

    elif scenario_name == "Region3" :
        # Creates data for Region3
        region_dict={"name": "Region3"}
        region_dict["nodes"] = ["F3_1", "F3_2", "DC3_1", "DC3_2", "B3_1", "B3_2"]
        region_dict["factory nodes"] = ["F3_1","F3_2"]
        region_dict["buyer nodes"] = ["B3_1","B3_2"]
        region_dict["distribution center nodes"]= ["DC3_1","DC3_2"]
        region_dict["supply"] = {"F3_1": 80, "F3_2": 60, "B3_1": -100, "B3_2": -100, "DC3_1": 0, "DC3_2": 0}
        region_dict["arcs"] = [("F3_1","DC3_1"), ("F3_2","DC3_2"), ("DC3_1","B3_1"),
            ("DC3_2","B3_2"), ("DC3_1","DC3_2"), ("DC3_2","DC3_1")]

        region_dict["production costs"] = {"F3_1": 50, "F3_2": 50}
        region_dict["revenues"] = {"B3_1": 900, "B3_2": 700}
        region_dict["flow capacities"] = {("F3_1","DC3_1"): 80, ("F3_2","DC3_2"): 60, ("DC3_1","B3_1"): 100,
            ("DC3_2","B3_2"): 100, ("DC3_1","DC3_2"): 70, ("DC3_2","DC3_1"): 50}
        region_dict["flow costs"] = {("F3_1","DC3_1"): 100, ("F3_2","DC3_2"): 100, ("DC3_1","B3_1"): 201,
            ("DC3_2","B3_2"): 200, ("DC3_1","DC3_2"): 100, ("DC3_2","DC3_1"): 100}
    
    elif scenario_name == "Region4":
        # Creates data for Region4
        region_dict={"name": "Region4"}
        region_dict["nodes"] = ["F4_1", "F4_2", "DC4", "B4_1", "B4_2"]
        region_dict["factory nodes"] = ["F4_1","F4_2"]
        region_dict["buyer nodes"] = ["B4_1","B4_2"]
        region_dict["distribution center nodes"] = ["DC4"]
        region_dict["supply"] = {"F4_1": 200, "F4_2": 30, "B4_1": -100, "B4_2": -100, "DC4": 0}
        region_dict["arcs"] = [("F4_1","DC4"), ("F4_2","DC4"), ("DC4","B4_1"), ("DC4","B4_2")]

        region_dict["production costs"] = {"F4_1": 50, "F4_2": 50}
        region_dict["revenues"] = {"B4_1": 900, "B4_2": 700}
        region_dict["flow capacities"] = {("F4_1","DC4"): 80, ("F4_2","DC4"): 60, ("DC4","B4_1"): 100, ("DC4","B4_2"): 100}
        region_dict["flow costs"] = {("F4_1","DC4"): 100, ("F4_2","DC4"): 80, ("DC4","B4_1"): 90, ("DC4","B4_2"): 70}
        
    else:
        raise RuntimeError (f"unknown Region name {scenario_name}")

    assert _is_partition(region_dict["nodes"], region_dict["factory nodes"], region_dict["buyer nodes"], region_dict["distribution center nodes"])

    return region_dict


###Creates the model when local_dict is given
def min_cost_distr_problem(local_dict, sense=pyo.minimize):
    """ Create an arcs formulation of network flow

    Args:
        local_dict (dict): dictionary representing a region including the dummy nodes \n
        sense (=pyo.minimize): we aim to minimize the cost, this should always be minimize

    Returns:
        model (Pyomo ConcreteModel) : the instantiated model
    """
    # Assert sense == pyo.minimize, "sense should be equal to pyo.minimize"
    # First, make the special In, Out arc lists for each node
    arcsout = {n: list() for n in local_dict["nodes"]}
    arcsin = {n: list() for n in local_dict["nodes"]}
    for a in local_dict["arcs"]:
        arcsout[a[0]].append(a)
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
        elif n in local_dict["dummy nodes source"]: 
            return (0,local_dict["dummy nodes source slack bounds"][n])
        elif n in local_dict["dummy nodes target"]: #this slack will respect the opposite flow balance rule
            return (0,local_dict["dummy nodes target slack bounds"][n])
        elif n in local_dict["distribution center nodes"]:
            return (0,0)
        else:
            raise ValueError(f"unknown node type for node {n}")
        
    model.y = pyo.Var(local_dict["nodes"], bounds=slackBounds_rule)

    model.MinCost = pyo.Objective(expr=\
                    sum(local_dict["flow costs"][a]*model.flow[a] for a in local_dict["arcs"]) \
                    + sum(local_dict["production costs"][n]*(local_dict["supply"][n]-model.y[n]) for n in local_dict["factory nodes"]) \
                    + sum(local_dict["revenues"][n]*(local_dict["supply"][n]-model.y[n]) for n in local_dict["buyer nodes"]) ,
                      sense=sense)
    
    def FlowBalance_rule(m, n):
        #we change the definition of the slack for target dummy nodes so that we have the slack from the source and from the target equal
        if n in local_dict["dummy nodes target"]:
            return sum(m.flow[a] for a in arcsout[n])\
            - sum(m.flow[a] for a in arcsin[n])\
            - m.y[n] == local_dict["supply"][n]
        else:
            return sum(m.flow[a] for a in arcsout[n])\
            - sum(m.flow[a] for a in arcsin[n])\
            + m.y[n] == local_dict["supply"][n]
    model.FlowBalance= pyo.Constraint(local_dict["nodes"], rule=FlowBalance_rule)

    return model


###Creates the scenario
def scenario_creator(scenario_name, num_scens=None):
    """Creates the model, which should include the consensus variables. \n
    However, this function shouldn't attach the consensus variables to root nodes, as it is done in admmWrapper.

    Args:
        scenario_name (str): the name of the scenario that will be created. Here is of the shape f"Region{i}" with 1<=i<=num_scens \n
        num_scens (int): number of scenarios (regions). Useful to create the corresponding inter-region dictionary

    Returns:
        Pyomo ConcreteModel: the instantiated model
    """
    assert (num_scens is not None)
    inter_region_dict = inter_region_dict_creator(num_scens)
    region_dict = region_dict_creator(scenario_name)
    # Adding dummy nodes and associated features
    local_dict = dummy_nodes_generator(region_dict, inter_region_dict)
    # Generating the model
    model = min_cost_distr_problem(local_dict)
    
    varlist = list()
    sputils.attach_root_node(model, model.MinCost, varlist)    
    
    return model


###Functions required in other files, which constructions are specific to the problem

def scenario_denouement(rank, scenario_name, scenario):
    """for each scenario prints its name and the final variable values

    Args:
        rank (int): not used here, but could be helpful to know the location
        scenario_name (str): name of the scenario
        scenario (Pyomo ConcreteModel): the instantiated model
    """
    print(f"flow values for {scenario_name}")
    scenario.flow.pprint()
    print(f"slack values for {scenario_name}")
    scenario.y.pprint()


def consensus_vars_creator(num_scens):
    """The following function creates the consensus_vars dictionary thanks to the inter-region dictionary. \n
    This dictionary has redundant information, but is useful for admmWrapper.

    Args:
        num_scens (int): select the number of scenarios (regions) wanted
    
    Returns:
        dict: dictionary which keys are the regions and values are the list of consensus variables 
        present in the region
    """
    # Due to the small size of inter_region_dict, it is not given as argument but rather created. 
    inter_region_dict = inter_region_dict_creator(num_scens)
    consensus_vars = {}
    for arc in inter_region_dict["arcs"]:
        source,target = arc
        region_source,node_source = source
        region_target,node_target = target
        dummy_node = node_source + node_target
        vstr = f"y[{dummy_node}]" #variable name as string, y is the slack

        #adds dummy_node in the source region
        if region_source not in consensus_vars: #initiates consensus_vars[region_source]
            consensus_vars[region_source] = list()
        consensus_vars[region_source].append(vstr)

        #adds dummy_node in the target region
        if region_target not in consensus_vars: #initiates consensus_vars[region_target]
            consensus_vars[region_target] = list()
        consensus_vars[region_target].append(vstr)
    return consensus_vars


def scenario_names_creator(num_scens):
    """Creates the name of every scenario.

    Args:
        num_scens (int): number of scenarios

    Returns:
        list (str): the list of names
    """
    return [f"Region{i+1}" for i in range(num_scens)]


def kw_creator(cfg):
    """
    Args:
        cfg (config): specifications for the problem. We only look at the number of scenarios

    Returns:
        dict (str): the kwargs that are used in distr.scenario_creator, here {"num_scens": num_scens}
    """
    kwargs = {"num_scens" : cfg.get('num_scens', None),
              }
    if kwargs["num_scens"] not in [2, 3, 4]:
        RuntimeError (f"unexpected number of regions {cfg.num_scens}, whould be in  [2, 3, 4]")
    return kwargs


def inparser_adder(cfg):
    #requires the user to give the number of scenarios
    cfg.num_scens_required()