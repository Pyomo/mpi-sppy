###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Our data, gives the constraints inside each in region in region_dict_creator
# and amongst the different regions in inter_region_dict_creator
# The data slightly differs depending on the number of regions (num_scens) which is created for 2, 3 or 4 regions

### This file creates the data through the region_dict and inter_region_dict
# First there is a hard wired data_set, then there is a scalable dataset

# Hardwired data sets
import json
import re
import numpy as np


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


def region_dict_creator(scenario_name):
    """ Create a scenario for the inter-region max profit distribution example.

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

if __name__ == "__main__":
    #creating the json files
    for num_scens in range(2,5):
        inter_region_dict_path = f'json_dicts.inter_region_dict{num_scens}.json'
        data = inter_region_dict_creator(num_scens)

        # Write the data to the JSON file
        with open(inter_region_dict_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    for i in range(1,5):
        region_dict_path = f'json_dicts.region{i}_dict.json'
        data = region_dict_creator(f"Region{i}")

        # Write the data to the JSON file
        with open(region_dict_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

########################################################################################################################
# Scalable datasets

def parse_node_name(name):
    """ decomposes the name, for example "DC1_2 gives" "DC",1,2
    Args:
        name (str): name of the node

    Returns:
        triplet (str, int, int): type of node ("DC", "F" or "B"), number of the region, and number of the node
    """
    # Define the regular expression pattern
    pattern = r'^([A-Za-z]+)(\d+)(?:_(\d+))?$'
    
    # Match the pattern against the provided name
    match = re.match(pattern, name)
    
    if match:
        # Extract the prefix, the first number, and the second number (if present)
        prefix = match.group(1)
        first_number = match.group(2)
        second_number = match.group(3) if match.group(3) is not None else '1'
        assert prefix in ["DC", "F", "B"]
        return prefix, int(first_number), int(second_number)
    else:
        raise RuntimeError (f"the node {name} can't be well decomposed")


def _node_num(max_node_per_region, node_type, region_num, count):
    """
    Args:
        max_node_per_region (int): maximum number of node per region per type

    Returns:
        int: a number specific to the node. This allows to have unrelated seeds
    """
    node_types = ["DC", "F", "B"] #no need to include the dummy nodes as they are added automatically
    return (max_node_per_region * region_num + count) * len(node_types) + node_types.index(node_type)

def _pseudo_random_arc(node_1, node_2, prob, cfg, intra=True): #in a Region
    """decides pseudo_randomly whether an arc will be created based on the two nodes 

    Args:
        node_1 (str): name of the source node
        node_2 (str): name of the target node
        prob (float): probability that the arc is created
        cfg (pyomo config): the config arguments
        intra (bool, optional): True if the arcs are inside a region, false if the arc is between different regions. Defaults to True.

    Returns:
        _type_: _description_
    """
    max_node_per_region = cfg.mnpr
    node_type1, region_num1, count1 = parse_node_name(node_1)
    node_type2, region_num2, count2 = parse_node_name(node_2)     
    node_num1 = _node_num(max_node_per_region, node_type1, region_num1, count1)
    node_num2 = _node_num(max_node_per_region, node_type2, region_num2, count2)
    if intra:
        assert region_num1 == region_num2, f"supposed to happen in a region ({intra=}), but {region_num1, region_num2=}"
    else:
        if region_num1 == region_num2: # inside a region, however intra=False so no connexion
            return False
    max_node_num = _node_num(max_node_per_region, "B", cfg.num_scens+1, max_node_per_region+1) # maximum number possible
    np.random.seed(min(node_num1, node_num2) * max_node_num + max(node_num1, node_num2)) # it is symmetrical
    random_number = np.random.rand()
    # Determine if the event occurs
    boo = random_number < prob
    return boo


def _intra_arc_creator(my_dict, node_1, node_2, cfg, arc, arc_params, my_seed, intra=True):
    """if the arc is chosen randomly to be constructed, it is added to the dictionary with its cost and capacity

    Args:
        my_dict (dict): either a region_dict if intra=True, otherwise the inter_region_dict
        node_1 (str): name of the source node
        node_2 (str): name of the target node
        cfg (pyomo config): the config arguments
        arc (pair of pair of strings): of the shape source,target with source = region_source, node_source
        arc_params (dict of bool): parameters for random cost and capacity
        my_seed (int): unique number used as seed
        intra (bool, optional): True if the arcs are inside a region, false if the arc is between different regions. Defaults to True.
    """
    prob = arc_params["prob"]
    mean_cost = arc_params["mean cost"]
    cv_cost = arc_params["cv cost"]
    mean_capacity = arc_params["mean capacity"]
    cv_capacity = arc_params["cv capacity"]
    if _pseudo_random_arc(node_1, node_2, prob, cfg, intra=intra):
        my_dict["arcs"].append(arc)
        np.random.seed(my_seed % 2**32)
        if intra: # not the same names used
            cost_name = "flow costs"
            capacity_name = "flow capacities"
        else:
            cost_name = "costs"
            capacity_name = "capacities"
        my_dict[cost_name][arc] = max(np.random.normal(mean_cost,cv_cost),0)
        np.random.seed((2**31+my_seed) % 2**32)
        my_dict[capacity_name][arc] = max(np.random.normal(mean_capacity,cv_capacity),0)


def scalable_inter_region_dict_creator(all_DC_nodes, cfg, data_params): # same as inter_region_dict_creator but the scalable version
    inter_region_arc_params = data_params["inter_region_arc"]
    inter_region_dict={}
    inter_region_dict["arcs"] = list()
    inter_region_dict["costs"] = {}
    inter_region_dict["capacities"] = {}
    count = 0
    for node_1 in all_DC_nodes: #although inter_region_dict["costs"] and ["capacities"] could be done with comprehension, "arcs" can't
        for node_2 in all_DC_nodes:
            if node_1 != node_2:
                _, region_num1, _ = parse_node_name(node_1)
                source = f"Region{region_num1}", node_1
                _, region_num2, _ = parse_node_name(node_2)
                target = f"Region{region_num2}", node_2
                arc = source, target
                _intra_arc_creator(inter_region_dict, node_1, node_2, cfg, arc, inter_region_arc_params, count, intra=False)
                count += 1
    return inter_region_dict


def all_nodes_dict_creator(cfg, data_params):
    """
    Args:
        cfg (pyomo config): configuration arguments
        data_params (nested dict): allows to construct the random probabilities

    Returns:
        (dict of str): the keys are regions containing all their nodes.
    """
    all_nodes_dict = {}
    num_scens = cfg.num_scens
    max_node_per_region = cfg.mnpr # maximum node node of a certain type in any region
    all_nodes_dict = {}
    production_costs_mean = data_params["production costs mean"]
    production_costs_cv = data_params["production costs cv"] #coefficient of variation
    revenues_mean = data_params["revenues mean"]
    revenues_cv = data_params["revenues cv"]
    supply_factory_mean = data_params["supply factory mean"]
    supply_factory_cv = data_params["supply factory cv"]
    supply_buyer_mean = data_params["supply buyer mean"]
    supply_buyer_cv = data_params["supply buyer cv"]
    for i in range(1, num_scens+1):
        region_name = f"Region{i}"
        all_nodes_dict[region_name] = {}
        node_types = ["DC", "F", "B"]
        all_nodes_dict[region_name]["nodes"] = []
        association_types = {"DC": "distribution center nodes", "F": "factory nodes", "B": "buyer nodes"}
        all_nodes_dict[region_name]["production costs"] = {}
        all_nodes_dict[region_name]["revenues"] = {}
        all_nodes_dict[region_name]["supply"] = {}        
        for node_type in node_types:
            node_base_num = _node_num(max_node_per_region, node_type, i, 1) #the first node that will be created will have this number
            # That helps us to have a seed, thanks to that we choose an integer which will be the number of nodes of this type
            np.random.seed(node_base_num)
            if node_type == "F" or node_type == "B":
                m = np.random.randint(0, max_node_per_region)
            else:
                m = np.random.randint(1, int(np.sqrt(max_node_per_region))+1)
            all_nodes_dict[region_name][association_types[node_type]] = [node_type + str(i) + "_" +str(j) for j in range(1, m+1)]
            all_nodes_dict[region_name]["nodes"] += all_nodes_dict[region_name][association_types[node_type]]
            if node_type == "F":
                count = 1
                for node_name in all_nodes_dict[region_name][association_types[node_type]]:
                    np.random.seed(_node_num(max_node_per_region, node_type, i, count) + 2**28)
                    all_nodes_dict[region_name]["production costs"][node_name] = max(0,np.random.normal(production_costs_mean,production_costs_cv))
                    np.random.seed(_node_num(max_node_per_region, node_type, i, count) + 2*2**28)
                    all_nodes_dict[region_name]["supply"][node_name] = max(0,np.random.normal(supply_factory_mean,supply_factory_cv)) #positive
                    count += 1
            if node_type == "B":
                count = 1
                for node_name in all_nodes_dict[region_name][association_types[node_type]]:
                    np.random.seed(_node_num(max_node_per_region, node_type, i, count) + 2**28)
                    all_nodes_dict[region_name]["revenues"][node_name] = min(max(0, np.random.normal(revenues_mean,revenues_cv)), data_params["max revenue"])
                    np.random.seed(_node_num(max_node_per_region, node_type, i, count) + 2*2**28)
                    all_nodes_dict[region_name]["supply"][node_name] = - max(0, np.random.normal(supply_buyer_mean,supply_buyer_cv)) #negative
                    count += 1
            if node_type == "DC":
                for node_name in all_nodes_dict[region_name][association_types[node_type]]:
                    all_nodes_dict[region_name]["supply"][node_name] = 0
    return all_nodes_dict


def scalable_region_dict_creator(scenario_name, all_nodes_dict=None, cfg=None, data_params=None): # same as region_dict_creator but the scalable version
    assert all_nodes_dict is not None
    assert cfg is not None
    assert data_params is not None
    local_nodes_dict = all_nodes_dict[scenario_name]
    region_dict = local_nodes_dict
    region_dict["name"] = scenario_name
    region_dict["arcs"] = list()
    region_dict["flow costs"] = {}
    region_dict["flow capacities"] = {}
    count = 2**30 # to have unrelated data with the production_costs
    for node_1 in local_nodes_dict["nodes"]: #although inter_region_dict["costs"] and ["capacities"] could be done with comprehension, "arcs" can't
        for node_2 in local_nodes_dict["nodes"]:
            if node_1 != node_2:
                node_type1, _, _ = parse_node_name(node_1)
                node_type2, _, _ = parse_node_name(node_2)
                arcs_association = {("F","DC") : data_params["arc_F_DC"], ("DC", "B") : data_params["arc_DC_B"], ("F", "B") : data_params["arc_F_B"], ("DC", "DC"): data_params["arc_DC_DC"]}
                arc_type = (node_type1, node_type2)
                if arc_type in arcs_association:
                    arc_params = arcs_association[arc_type]
                    arc = (node_1, node_2)
                    _intra_arc_creator(region_dict, node_1, node_2, cfg, arc, arc_params, my_seed=count, intra=True)
                    count += 1
    return region_dict
