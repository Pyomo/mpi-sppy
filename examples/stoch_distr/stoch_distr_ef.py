# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# general example driver for distr with cylinders

# Solves the stochastic distribution problem 
import mpisppy.utils.admm_ph as admm_ph
import distr
import mpisppy.cylinders
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

    cfg.parse_command_line("distr_ef2")
    return cfg


def solve_EF_directly(admm,solver_name):
    scenario_names = admm.local_scenario_names
    scenario_creator = admm.admm_ph_scenario_creator

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

    options = {}
    all_scenario_names = distr.scenario_names_creator(num_scens=cfg.num_scens)
    scenario_creator = distr.scenario_creator
    scenario_creator_kwargs = distr.kw_creator(cfg)  
    consensus_vars = distr.consensus_vars_creator(cfg.num_scens)

    n_cylinders = 1
    admm = admm_ph.ADMM_PH(options,
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