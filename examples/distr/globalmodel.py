###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Distribution example without decomposition, this file is only written to execute the non-scalable example

### This code line can execute the script for a certain example
# python globalmodel.py --solver-name cplex_direct --num-scens 3


import pyomo.environ as pyo
import distr
import distr_data
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


def global_dict_creator(num_scens, start=0):
    """Merges the different region_dict thanks to the inter_region_dict in distr to create a global dictionary

    Args:
        num_scens (int): number of regions wanted
        start (int, optional): unuseful here. Defaults to 0.

    Returns:
        dict: dictionary with the information to create a min cost distribution problem
    """
    inter_region_dict = distr_data.inter_region_dict_creator(num_scens)
    global_dict={}
    for i in range(start,start+num_scens):
        scenario_name=f"Region{i+1}"
        region_dict = distr_data.region_dict_creator(scenario_name)
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
    assert cfg.scalable is False, "the global model example has not been adapted for the scalable example"
    model = distr.min_cost_distr_problem(global_dict_creator(num_scens=cfg.num_scens))

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
