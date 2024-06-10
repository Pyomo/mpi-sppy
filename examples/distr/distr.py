# Network Flow - various formulations
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import distr_data
import time

# In this file, we create a (linear) inter-region minimal cost distribution problem.
# Our data, gives the constraints inside each in region in region_dict_creator
# and amongst the different regions in inter_region_dict_creator
# The data slightly differs depending on the number of regions (num_scens) which is created for 2, 3 or 4 regions

# Note: regions in our model will be represented in mpi-sppy by scenarios and to ensure the inter-region constraints
#       we will use dummy-nodes, which will be represented in mpi-sppy by non-anticipative variables
#       The following association of terms are made: regions = scenarios, and dummy-nodes = nonants = consensus-vars


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
    #print(f"flow values for {scenario_name}")
    #scenario.flow.pprint()
    #print(f"slack values for {scenario_name}")
    #scenario.y.pprint()
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
    for arc in inter_region_dict["arcs"]:
        source,target = arc
        region_source,node_source = source
        region_target,node_target = target
        dummy_node = node_source + node_target
        vstr = f"y[{dummy_node}]" #variable name as string, y is the slack

        #adds dummy_node in the source region
        if not region_source in consensus_vars: #initiates consensus_vars[region_source]
            consensus_vars[region_source] = list()
        consensus_vars[region_source].append(vstr)

        #adds dummy_node in the target region
        if not region_target in consensus_vars: #initiates consensus_vars[region_target]
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