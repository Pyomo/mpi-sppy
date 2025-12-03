###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# This shows two ways to get parameters to the EF for solution; both are fairly short.

import sys
import farmer
import mpisppy.utils.sputils as sputils
import mpisppy.utils.solver_spec as solver_spec
from mpisppy.utils import config
import pyomo.environ as pyo


def main_no_cfg():
    # Some parameters from sys.argv and some hard-wired.
    if len(sys.argv) != 5:
        print("usage python farmer_ef.py {crops_multiplier} {scen_count} {use_integer} {solver_name}")
        print("e.g., python farmer_ef.py 1 3 1 gurobi")
        quit()

    scenario_creator = farmer.scenario_creator

    crops_multiplier = int(sys.argv[1])
    scen_count = int(sys.argv[2])
    use_integer = int(sys.argv[3])
    solver_name = sys.argv[4]
    
    scenario_creator_kwargs = {
        "use_integer": use_integer,
        "crops_multiplier": crops_multiplier,
    }

    scenario_names = ['Scenario' + str(i) for i in range(scen_count)]

    ef = sputils.create_EF(
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    solver = pyo.SolverFactory(solver_name)
    if 'persistent' in solver_name:
        solver.set_instance(ef, symbolic_solver_labels=True)
        solver.solve(tee=True)
    else:
        solver.solve(ef, tee=True, symbolic_solver_labels=True,)

    return ef


def main_with_cfg():
    # use a config objec and supporting utililities
    # python farmer_ef.py --help

    cfg = config.Config()
    cfg.EF2()
    farmer.inparser_adder(cfg)
    cfg.parse_command_line("farmer_ef")

    ef = sputils.create_EF(
        farmer.scenario_names_creator(cfg.num_scens),
        farmer.scenario_creator,
        scenario_creator_kwargs=farmer.kw_creator(cfg),
    )

    sroot, solver_name, solver_options = solver_spec.solver_specification(cfg, "EF")
   
    solver = pyo.SolverFactory(solver_name)
    if solver_options is not None:
        # We probably could just assign the dictionary in one line...
        for option_key,option_value in solver_options.items():
            solver.options[option_key] = option_value
    if 'persistent' in solver_name:
        solver.set_instance(ef, symbolic_solver_labels=True)
        results = solver.solve(tee=True)
    else:
        results = solver.solve(ef, tee=True, symbolic_solver_labels=True,)
    if not pyo.check_optimal_termination(results):
        print("Warning: solver reported non-optimal termination status")
        
    return ef


if __name__ == '__main__':
    # show two ways to get parameters
    use_cfg = False
    if not use_cfg:
        main_ef = main_no_cfg()
    else:
        main_ef = main_with_cfg()

    print(f"EF objective: {pyo.value(main_ef.EF_Obj)}")
    sputils.ef_ROOT_nonants_npy_serializer(main_ef, "farmer_root_nonants.npy")
