# Distribution example without decomposition
import pyomo.environ as pyo
import distr
from mpisppy.utils import config

def _parse_args():
    # create a config object and parse
    cfg = config.Config()
    distr.inparser_adder(cfg)
    cfg.add_to_config("solver_name",
                      description="which solver",
                      domain=str,
                      default=None,
                      argparse_args = {"required": True},
                      )
    cfg.parse_command_line("globalmodel")
    
    return cfg


##we need to create a new model solver, to avoid dummy nodes
def min_cost_distr_problem(region_dict, sense=pyo.minimize):
    """ Create an arcs formulation of network flow, the function is the simplification without dummy nodes of this function in distr

    Arg: 
        region_dict (dict): given by region_dict_creator with test_without_dummy=True so that there is no dummy node

    Returns:
        model (Pyomo ConcreteModel) : the instantiated model
    """
    # First, make the special In, Out arc lists for each node
    arcsout = {n: list() for n in region_dict["nodes"]}
    arcsin = {n: list() for n in region_dict["nodes"]}
    for a in region_dict["arcs"]:
        arcsout[a[0]].append(a)
        arcsin[a[1]].append(a)

    model = pyo.ConcreteModel(name='MinCostFlowArcs')
    def flowBounds_rule(model, i,j):
        return (0, region_dict["flow capacities"][(i,j)])
    model.flow = pyo.Var(region_dict["arcs"], bounds=flowBounds_rule)  # x

    def slackBounds_rule(model, n):
        if n in region_dict["factory nodes"]:
            return (0, region_dict["supply"][n])
        elif n in region_dict["buyer nodes"]:
            return (region_dict["supply"][n], 0)
        elif n in region_dict["distribution center nodes"]:
            return (0,0)
        else:
            raise ValueError(f"unknown node type for node {n}")
        
    model.y = pyo.Var(region_dict["nodes"], bounds=slackBounds_rule)

    model.MinCost = pyo.Objective(expr=\
                                  sum(region_dict["flow costs"][a]*model.flow[a] for a in region_dict["arcs"]) \
                                + sum(region_dict["production costs"][n]*(region_dict["supply"][n]-model.y[n]) for n in region_dict["factory nodes"]) \
                                + sum(region_dict["revenues"][n]*(region_dict["supply"][n]-model.y[n])for n in region_dict["buyer nodes"]) ,
                                  sense=sense)
    
    def FlowBalance_rule(m, n):
        return sum(m.flow[a] for a in arcsout[n])\
            - sum(m.flow[a] for a in arcsin[n])\
            + m.y[n] == region_dict["supply"][n]
    model.FlowBalance= pyo.Constraint(region_dict["nodes"], rule=FlowBalance_rule)

    return model


def global_dict_creator(num_scens, start=0):
    """Merges the different region_dict thanks to the inter_region_dict in distr to create a global dictionary

    Args:
        num_scens (int): number of regions wanted
        start (int, optional): unuseful here. Defaults to 0.

    Returns:
        dict: dictionary with the information to create a min cost distribution problem
    """
    inter_region_dict = distr.inter_region_dict_creator(num_scens)
    global_dict={}
    for i in range(start,start+num_scens):
        scenario_name=f"Region{i+1}"
        region_dict = distr.region_dict_creator(scenario_name)
        for key in region_dict:
            if key in ["nodes","factory nodes", "buyer nodes", "distribution center nodes", "arcs"]:
                if i == start:
                    global_dict[key]=[]
                for x in region_dict[key]:
                    global_dict[key].append(x)
            if key in ["supply", "production costs","revenues", "flow capacities", "flow costs"]:
                if i == start:
                    global_dict[key]={}
                for key2 in region_dict[key]:
                    global_dict[key][key2] = region_dict[key][key2]

    def _extract_arc(a):
        source, target = a
        node_source = source[1]
        node_target = target[1]
        return (node_source, node_target)
    
    for a in inter_region_dict["arcs"]:
        global_dict["arcs"].append(_extract_arc(a))
    for a in inter_region_dict["costs"]:
            global_dict["flow costs"][_extract_arc(a)] = inter_region_dict["costs"][a]
    for a in inter_region_dict["capacities"]:
            global_dict["flow capacities"][_extract_arc(a)] = inter_region_dict["capacities"][a]
    return global_dict

def main():
    """
        do all the work
    """
    cfg = _parse_args()
    model = min_cost_distr_problem(global_dict_creator(num_scens=cfg.num_scens))

    solver_name = cfg.solver_name
    opt = pyo.SolverFactory(solver_name)
    results = opt.solve(model)  # solves and updates model
    pyo.assert_optimal_termination(results)
    model.pprint()

    # Grabs the objective function
    objectives = model.component_objects(pyo.Objective, active=True)
    count = 0
    for obj in objectives:
        objective_value = pyo.value(obj)
        count += 1   
    assert count == 1, f"only one objective function is authorized, there are {count}"
    print(f"Objective '{obj}' value: {objective_value}")


if __name__ == "__main__":
    main()
