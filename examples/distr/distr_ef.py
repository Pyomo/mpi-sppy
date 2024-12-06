###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# general example driver for distr with cylinders

# This file can be executed thanks to python distr_ef.py --num-scens 2 --solver-name cplex_direct

import mpisppy.utils.admmWrapper as admmWrapper
import distr
import distr_data
import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
from mpisppy.utils import config
from mpisppy import MPI
global_rank = MPI.COMM_WORLD.Get_rank()


write_solution = True

def _parse_args():
    # create a config object and parse
    cfg = config.Config()
    distr.inparser_adder(cfg)
    cfg.add_to_config("solver_name",
                         description="Choice of the solver",
                         domain=str,
                         default=None)

    cfg.parse_command_line("distr_ef")
    return cfg


def solve_EF_directly(admm,solver_name):
    scenario_names = admm.local_scenario_names
    scenario_creator = admm.admmWrapper_scenario_creator

    ef = sputils.create_EF(
        scenario_names,
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
        json_file_path = "data_params.json"

        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data_params = json.load(file)
            all_nodes_dict = distr_data.all_nodes_dict_creator(cfg, data_params)
            all_DC_nodes = [DC_node for region in all_nodes_dict for DC_node in all_nodes_dict[region]["distribution center nodes"]]
            inter_region_dict = distr_data.scalable_inter_region_dict_creator(all_DC_nodes, cfg, data_params)
    else:
        inter_region_dict = distr_data.inter_region_dict_creator(num_scens=cfg.num_scens)
        all_nodes_dict = None
        data_params = {"max revenue": 1200}

    options = {}
    all_scenario_names = distr.scenario_names_creator(num_scens=cfg.num_scens)
    scenario_creator = distr.scenario_creator
    scenario_creator_kwargs = distr.kw_creator(all_nodes_dict, cfg, inter_region_dict, data_params)  
    consensus_vars = distr.consensus_vars_creator(cfg.num_scens, inter_region_dict, all_scenario_names)

    n_cylinders = 1
    admm = admmWrapper.AdmmWrapper(options,
                           all_scenario_names, 
                           scenario_creator,
                           consensus_vars,
                           n_cylinders=n_cylinders,
                           mpicomm=MPI.COMM_WORLD,
                           scenario_creator_kwargs=scenario_creator_kwargs,
                           )

    solved_ef = solve_EF_directly(admm, cfg.solver_name)
    with open("ef.txt", "w") as f:
        solved_ef.pprint(f)
    print ("******* model written to ef.txt *******")
    solution_file_name = "solution_distr.txt"
    sputils.write_ef_first_stage_solution(solved_ef,
                                solution_file_name,)
    print(f"ef solution written to {solution_file_name}")
    print(f"EF objective: {pyo.value(solved_ef.EF_Obj)}")


if __name__ == "__main__":
    main()