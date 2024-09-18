###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# This file is used in the tests and should not be modified!

# Network Flow - various formulations
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import numpy as np
import re
import mpisppy.scenario_tree as scenario_tree

# This version present in the test slightly differs from the original 
# as it allows a three-stage problem

# In this file, we create a stochastic (linear) inter-region minimal cost distribution problem.
# Our data, gives the constraints inside each in region in region_dict_creator
# and amongst the different regions in inter_region_dict_creator
# The data slightly differs depending on the number of regions (num_admm_subproblems) which is created for 2, 3 or 4 regions

### The following functions create the data 

def inter_region_dict_creator(num_admm_subproblems): #doesn't depend on the scenarios in this example
    """Creates the oriented arcs between the regions, with their capacities and costs. \n
    This dictionary represents the inter-region constraints and flows. It indicates where to add dummy nodes.

    Args:
        num_admm_subproblems (int): select the number of subproblems (regions) wanted

    Returns:
        dict: 
            Each arc is presented as a pair (source, target) with source and target containing (scenario_name, node_name) \n
            The arcs are used as keys for the dictionaries of costs and capacities
    """
    inter_region_dict={}

    if num_admm_subproblems == 2:
        inter_region_dict["arcs"]=[(("Region1","DC1"),("Region2","DC2"))]
        inter_region_dict["costs"]={(("Region1","DC1"),("Region2","DC2")): 100}
        inter_region_dict["capacities"]={(("Region1","DC1"),("Region2","DC2")): 70}

    elif num_admm_subproblems == 3:
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
    
    elif num_admm_subproblems == 4:
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
    # Verification that the node list is a partition of the other nodes lists
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

def region_dict_creator(admm_subproblem_name): #in this precise example region_dict doesn't depend on the scenario
    """ Create a scenario for the inter-region max profit distribution example.
    In this precise example region_dict doesn't depend on the  stochastic scenario

    The convention for node names is: 
        Symbol + number of the region (+ _ + number of the example if needed),
        with symbols DC for distribution centers, F for factory nodes, B for buyer nodes. \n
        For instance: F3_1 is the 1st factory node of region 3. \n

    Args:
        scenario_name (str):
            Name of the scenario to construct.

    Returns:
        region_dict (dict): contains all the information in the given region to create the model. It is composed of:\n
            "nodes" (list of str): all the nodes. The following subsets are also nodes:
            "factory nodes", "buyer nodes", "distribution center nodes", \n
            "arcs" (list of 2 tuples of str) : (node, node) pairs\n
            "supply" (dict[n] of float): supply; keys are nodes (negative for demand)\n
            "production costs" (dict of float): at each factory node\n
            "revenues" (dict of float): at each buyer node \n
            "flow costs" (dict[a] of float) : costs per unit flow on each arc \n
            "flow capacities" (dict[a] of floats) : upper bound capacities of each arc \n
    """
    if admm_subproblem_name == "Region1" :
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
        
    elif admm_subproblem_name == "Region2":
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

    elif admm_subproblem_name == "Region3" :
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
    
    elif admm_subproblem_name == "Region4":
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
        raise RuntimeError (f"unknown Region name {admm_subproblem_name}")

    assert _is_partition(region_dict["nodes"], region_dict["factory nodes"], region_dict["buyer nodes"], region_dict["distribution center nodes"])

    return region_dict


def _scenario_number(stoch_scenario_name, demand=None):
    if demand is None:
        scennum = sputils.extract_num(stoch_scenario_name)
    else: # 3-stage
        production_num = int(re.search(r'\d+', stoch_scenario_name).group())
        if demand == "high":
            scennum = (production_num-1)*2
        else:
            scennum = (production_num-1)*2+1
    return scennum


###Creates the model when local_dict is given, local_dict depends on the subproblem
def min_cost_distr_problem(local_dict, stoch_scenario_name, cfg, demand=None, sense=pyo.minimize):
    """ Create an arcs formulation of network flow for the region and stochastic scenario considered.

    Args:
        local_dict (dict): dictionary representing a region including the dummy nodes \n
        stoch_scenario_name (str): name of the stochastic scenario \n
        cfg (): config argument used here for the random parameters \n
        sense (=pyo.minimize): we aim to minimize the cost, this should always be minimize

    Returns:
        model (Pyomo ConcreteModel) : the instantiated model
    """
    # Helps to define pseudo randomly the percentage of loss at production
    scennum = _scenario_number(stoch_scenario_name, demand=demand)

    # Assert sense == pyo.minimize, "sense should be equal to pyo.minimize"
    # First, make the special In, Out arc lists for each node
    arcsout = {n: list() for n in local_dict["nodes"]}
    arcsin = {n: list() for n in local_dict["nodes"]}
    for a in local_dict["arcs"]:
        arcsout[a[0]].append(a)
        arcsin[a[1]].append(a)

    # Creating the random demand based on the supply if it is a three-stage example. In which case the original
    # local_dict["supply"][n] is the average of demand for a node n 
    if demand is not None:
        if demand == "high":
            factor = 1.2
        elif demand == "low":
            factor = 0.8
        else:
            raise RuntimeError (f"for the 3-stage problem demand should be 'high' or 'low', but it is {demand=}")
        for n in local_dict["buyer nodes"]:
            local_dict["supply"][n] *= factor

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

    if demand is not None: # 3-stage example, defining exportation
        def export_bounds(m,i):
            return (0,10)
        model.export = pyo.Var(local_dict["buyer nodes"], bounds=export_bounds)

    model.FirstStageCost = pyo.Expression(expr=\
                    sum(local_dict["production costs"][n]*(local_dict["supply"][n]-model.y[n]) for n in local_dict["factory nodes"]))
    
    if demand is not None: # 3-stage example
        model.SecondStageCost = pyo.Expression(expr=\
                        sum(-810*model.export[n] for n in local_dict["buyer nodes"]))
    
    model.LastStageCost = pyo.Expression(expr=\
                    sum(local_dict["flow costs"][a]*model.flow[a] for a in local_dict["arcs"]) \
                    + sum(local_dict["revenues"][n]*(local_dict["supply"][n]-model.y[n]) for n in local_dict["buyer nodes"]))

    if demand is not None: # 3-stage example
        model.MinCost = pyo.Objective(expr=model.FirstStageCost + model.SecondStageCost +model.LastStageCost, sense=sense)
    else: # 2-stage example
        model.MinCost = pyo.Objective(expr=model.FirstStageCost +model.LastStageCost, sense=sense)
    
    def FlowBalance_rule(m, n):
        #we change the definition of the slack for target dummy nodes so that we have the slack from the source and from the target equal
        if n in local_dict["dummy nodes target"]:
            return sum(m.flow[a] for a in arcsout[n])\
            - sum(m.flow[a] for a in arcsin[n])\
            - m.y[n] == local_dict["supply"][n]
        elif n in local_dict["factory nodes"]:
            # We generate pseudo randomly the loss on each factory node
            numbers = re.findall(r'\d+', n)
            # Concatenate the numbers together
            node_num = int(''.join(numbers))
            np.random.seed(scennum+node_num+cfg.initial_seed)
            return sum(m.flow[a] for a in arcsout[n])\
            - sum(m.flow[a] for a in arcsin[n])\
            == (local_dict["supply"][n] - m.y[n]) * min(1,max(0,1-np.random.normal(cfg.spm,cfg.cv)/100)) # We add the loss
        elif n in local_dict["buyer nodes"] and demand is not None: # for a 3-stage scenario the export changes the flow balance
            return sum(m.flow[a] for a in arcsout[n])\
            - sum(m.flow[a] for a in arcsin[n])\
            + m.y[n] == local_dict["supply"][n] - m.export[n]
        else:
            return sum(m.flow[a] for a in arcsout[n])\
            - sum(m.flow[a] for a in arcsin[n])\
            + m.y[n] == local_dict["supply"][n]
    model.FlowBalance= pyo.Constraint(local_dict["nodes"], rule=FlowBalance_rule)

    return model


###Functions required in other files, which constructions are specific to the problem

def scenario_denouement(rank, admm_stoch_subproblem_scenario_name, scenario):
    """For each admm stochastic scenario subproblem prints its name and the final variable values

    Args:
        rank (int): rank in which the scenario is placed
        admm_stoch_subproblem_scenario_name (str): name of the admm stochastic scenario subproblem
        scenario (Pyomo ConcreteModel): the instantiated model
    """
    return
    print(f"flow values for {admm_stoch_subproblem_scenario_name=} at {rank=}")
    scenario.flow.pprint()
    print(f"slack values for {admm_stoch_subproblem_scenario_name=} at {rank=}")
    scenario.y.pprint()
    if hasattr(scenario,"export"): # ie 3-stage problem
        print(f"export values for {admm_stoch_subproblem_scenario_name=} at {rank=}")
        scenario.export.pprint()


# Only if 3-stage problem for the example
def MakeNodesforScen(model, BFs, scennum, local_dict):
    """ Make just those scenario tree nodes needed by a scenario.
        Return them as a list.
        NOTE: the nodes depend on the scenario model and are, in some sense,
              local to it.
        Args:
            BFs (list of int): branching factors
    """
    ndn = "ROOT_"+str((scennum-0) // BFs[1]) # scennum is 0-based, 
    # In a more general way we should divide by: BFs[1:]
    #objfunc = pyo.Expression(expr=0) # Nothing is done so there is no cost
    retval = [scenario_tree.ScenarioNode("ROOT",
                                         1.0,
                                         1,
                                         model.FirstStageCost,
                                         [model.y[n] for n in  local_dict["factory nodes"]],
                                         model),
              scenario_tree.ScenarioNode(ndn,
                                         1.0/BFs[0],
                                         2,
                                         model.FirstStageCost, 
                                         [model.export[n] for n in  local_dict["buyer nodes"]], # No consensus_variable for now to simplify the model
                                         model,
                                         parent_name="ROOT")
              ]
    return retval


def branching_factors_creator(num_stoch_scens, num_stage):
    # BFs is only used with multiple stages
    if num_stage == 3:
        # There should always be a high demand and a low demand version
        assert num_stoch_scens % 2 == 0, "there should be an even number of stochastic scenarios for this 3 stage problem, but it is odd"
        BFs = [num_stoch_scens//2, 2]
    elif num_stage == 2:
        BFs = None # No need to use branching factors (or even to define them with two stage problems)
    else:
        raise RuntimeError (f"the example is only made for 2 or 3-stage problems, but {num_stage} were given")
    return BFs


###Creates the scenario
def scenario_creator(admm_stoch_subproblem_scenario_name, **kwargs):
    """Creates the model, which should include the consensus variables. \n
    However, this function shouldn't attach the consensus variables for the admm subproblems as it is done in admmWrapper.

    Args:
        admm_stoch_subproblem_scenario_name (str): the name given to the admm problem for the stochastic scenario. \n
        num_scens (int): number of scenarios (regions). Useful to create the corresponding inter-region dictionary

    Returns:
        Pyomo ConcreteModel: the instantiated model
    """
    cfg = kwargs.get("cfg")
    
    # assert(cfg.num_admm_subproblems is not None)
    # assert (cfg.num_stoch_scens is not None)
    admm_subproblem_name, stoch_scenario_name = split_admm_stoch_subproblem_scenario_name(admm_stoch_subproblem_scenario_name)

    inter_region_dict = inter_region_dict_creator(cfg.num_admm_subproblems)
    region_dict = region_dict_creator(admm_subproblem_name)

    # Adding dummy nodes and associated features
    local_dict = dummy_nodes_generator(region_dict, inter_region_dict)

    # Generating the model according to the number of stages and creatung the tree
    if cfg.num_stage == 2:
        model = min_cost_distr_problem(local_dict, stoch_scenario_name, cfg)
        sputils.attach_root_node(model, model.FirstStageCost, [model.y[n] for n in  local_dict["factory nodes"]])
    else:
        BFs = branching_factors_creator(cfg.num_stoch_scens, cfg.num_stage)

        if stoch_scenario_name.endswith('high'):
            demand = "high"
        elif stoch_scenario_name.endswith('low'):
            demand = "low"
        else:
            raise RuntimeError (f"the stochastic scenario name should end with 'high' or 'low', but it is {stoch_scenario_name}")
        model = min_cost_distr_problem(local_dict, stoch_scenario_name, cfg, demand=demand)
        scennum = _scenario_number(stoch_scenario_name, demand=demand)
        model._mpisppy_node_list = MakeNodesforScen(model, BFs, scennum, local_dict)
        model._mpisppy_probability = 1/cfg.num_stoch_scens
    
    return model


def consensus_vars_creator(admm_subproblem_names, stoch_scenario_name, kwargs, num_stage=2):
    """The following function creates the consensus_vars dictionary thanks to the inter-region dictionary. \n
    This dictionary has redundant information, but is useful for admmWrapper.

    Args:
        admm_subproblem_names (list of str): name of the admm subproblems (regions) \n
        stoch_scenario_name (str): name of any stochastic_scenario, it is only used 
        in this example to access the non anticipative variables (which are common to 
        every stochastic scenario) and their stage.
    
    Returns:
        dict: dictionary which keys are the subproblems and values are the list of 
        pairs (consensus_variable_name (str), stage (int)).
    """
    # Due to the small size of inter_region_dict, it is not given as argument but rather created. 
    inter_region_dict = inter_region_dict_creator(len(admm_subproblem_names))
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
        consensus_vars[region_source].append((vstr,num_stage))

        #adds dummy_node in the target region
        if region_target not in consensus_vars: #initiates consensus_vars[region_target]
            consensus_vars[region_target] = list()
        consensus_vars[region_target].append((vstr,num_stage))
    # now add the parents. It doesn't depend on the stochastic scenario so we chose one and
    # then we go through the models (created by scenario creator) for all the admm_stoch_subproblem_scenario 
    # which have this scenario as an ancestor (parent) in the tree
    for admm_subproblem_name in admm_subproblem_names:
        admm_stoch_subproblem_scenario_name = combining_names(admm_subproblem_name,stoch_scenario_name)
        model = scenario_creator(admm_stoch_subproblem_scenario_name, **kwargs)
        for node in model._mpisppy_node_list:
            for var in node.nonant_list:
                #print(f"{admm_subproblem_name=}")
                #print(f"{var.name=}")
                if var.name not in consensus_vars[admm_subproblem_name]:
                    consensus_vars[admm_subproblem_name].append((var.name, node.stage))
    return consensus_vars


def stoch_scenario_names_creator(num_stoch_scens, num_stage=2):
    """Creates the name of every stochastic scenario.

    Args:
        num_stoch_scens (int): number of stochastic scenarios

    Returns:
        list (str): the list of stochastic scenario names
    """
    if num_stage == 3:
        return [f"StochasticScenario{i+1}_{demand}" for i in range(num_stoch_scens//2) for demand in ["high","low"]]
    else:
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
    """ Gives the admm_subproblem_name and the stoch_scenario_name given an admm_stoch_subproblem_scenario_name.
    This function, specific to the problem, is the reciprocal function of ``combining_names`` which creates the 
    admm_stoch_subproblem_scenario_name given the admm_subproblem_name and the stoch_scenario_name.

    Args:
        admm_stoch_subproblem_scenario_name (str)

    Returns:
        (str,str): admm_subproblem_name, stoch_scenario_name
    """
    # Method specific to our example and because the admm_subproblem_name and stoch_scenario_name don't include "_"
    splitted = admm_stoch_subproblem_scenario_name.split('_')
    # The next line would require adding the argument num_stage and transferring it to stoch_admmWrapper
    #assert (len(splitted) == num_stage+2), f"appart from the 'ADMM_STOCH_' prefix, underscore should only separate stages"
    admm_subproblem_name = splitted[2]
    stoch_scenario_name = '_'.join(splitted[3:])
    return admm_subproblem_name, stoch_scenario_name


def kw_creator(cfg):
    """
    Args:
        cfg (config): specifications for the problem given on the command line

    Returns:
        dict (str): the kwargs that are used in distr.scenario_creator, chich are included in cfg.
    """
    kwargs = {
        "cfg": cfg
    }
    return kwargs


def inparser_adder(cfg):
    """Adding to the config argument, specific elements to our problems. In this case the numbers of stochastic scenarios 
    and admm subproblems which are required. And elements used for random number generations.

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
            description="Number of admm subproblems per stochastic scenario (default None)",
            domain=int,
            default=None,
            argparse_args = {"required": True}
        )

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
    
    cfg.add_to_config("num_stage",
                      description="choice of the number of stages for the example, by default two-stage",
                      domain=int,
                      default=2)
    