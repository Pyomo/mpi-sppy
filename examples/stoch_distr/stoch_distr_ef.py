###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# general example driver for stoch_distr with cylinders

# This file can be executed thanks to python stoch_distr_ef.py --num-stoch-scens 2 --solver-name xpress --num-stoch-scens 2 --num-admm-subproblems 3 --scalable --mnpr 6

# Solves the stochastic distribution problem 
import stoch_distr
import stoch_distr_admm_cylinders
import examples.distr.distr_data as distr_data
import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
from mpisppy.utils import config
from mpisppy import MPI
global_rank = MPI.COMM_WORLD.Get_rank()


write_solution = True

def _parse_args():
    # create a config object and parse
    cfg = config.Config()
    stoch_distr.inparser_adder(cfg)
    cfg.add_to_config("solver_name",
                         description="Choice of the solver",
                         domain=str,
                         default=None)

    cfg.parse_command_line("stoch_distr_ef")
    return cfg


def solve_EF_directly(admm,solver_name):
    local_scenario_names = admm.local_admm_stoch_subproblem_scenarios_names
    scenario_creator = admm.admmWrapper_scenario_creator

    ef = sputils.create_EF(
        local_scenario_names,
        scenario_creator,
        nonant_for_fixed_vars=False,
    )
    solver = pyo.SolverFactory(solver_name)
    if 'persistent' in solver_name:
        solver.set_instance(ef, symbolic_solver_labels=True)
        solver.solve(tee=True)
    else:
        solver.solve(ef, tee=True, symbolic_solver_labels=True,)
    return ef

def main():

    cfg = _parse_args()

    if cfg.scalable:
        import json
        json_file_path = "../distr/data_params.json"

        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data_params = json.load(file)
            # In distr_data num_admm_subproblems is called num_scens
            cfg.add_to_config("num_scens",
                      description="num admm subproblems",
                      domain=int,
                      default=cfg.num_admm_subproblems)
            all_nodes_dict = distr_data.all_nodes_dict_creator(cfg, data_params)
            all_DC_nodes = [DC_node for region in all_nodes_dict for DC_node in all_nodes_dict[region]["distribution center nodes"]]
            inter_region_dict = distr_data.scalable_inter_region_dict_creator(all_DC_nodes, cfg, data_params)
    else:
        inter_region_dict = distr_data.inter_region_dict_creator(num_scens=cfg.num_admm_subproblems)
        all_nodes_dict = None
        data_params = {"max revenue": 1200}
    
    n_cylinders = 1 # There is no spoke so we only use one cylinder
    admm, _ = stoch_distr_admm_cylinders._make_admm(cfg, n_cylinders, all_nodes_dict, inter_region_dict, data_params) 

    solved_ef = solve_EF_directly(admm, cfg.solver_name)
    with open("ef.txt", "w") as f:
        solved_ef.pprint(f)
    print ("******* model written to ef.txt *******")
    solution_file_name = "solution_distr.txt"
    sputils.write_ef_first_stage_solution(solved_ef,
                                solution_file_name,)
    print(f"EF solution written to {solution_file_name}")
    print(f"EF objective: {pyo.value(solved_ef.EF_Obj)}")


if __name__ == "__main__":
    main()