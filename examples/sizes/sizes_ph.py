###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# updated 23 April 2020
# Serial (not cylinders)

import sys
import mpisppy.phbase
import mpisppy.opt.ph
from mpisppy.convergers.primal_dual_converger import PrimalDualConverger
from mpisppy.extensions.xhatclosest import XhatClosest
from sizes import scenario_creator, \
                  scenario_denouement, \
                  _rho_setter


if __name__ == "__main__":

    if len(sys.argv) != 8:
        print("usage: python sizes_ph.py scenNum maxiter rho smooth_type pvalue_or_pratio beta proxlin")
        print("eg: python sizes_ph.py 3 100 1 0 0.0 1.0 1")
        quit()
    ScenCount = int(sys.argv[1])  # 3 or 10
    options = {}
    options["solver_name"] = "xpress"
    options["asynchronousPH"] = False
    options["PHIterLimit"] = int(sys.argv[2])
    options["defaultPHrho"] = float(sys.argv[3])
    options["convthresh"] = -0.001
    options["subsolvedirectives"] = None
    options["verbose"] = False
    options["display_timing"] = True
    options["display_progress"] = True
    options["linearize_proximal_terms"] = bool(int(sys.argv[7]))
    # one way to set up sub-problem solver options
    options["iter0_solver_options"] = {"mipgap": 0.01, "threads": 1}
    # another way
    options["iterk_solver_options"] = {"mipgap": 0.005, "threads": 1}
    
    options["xhat_closest_options"] =  {"xhat_solver_options": options["iterk_solver_options"], "keep_solution":True}
    options["primal_dual_converger_options"] = {"tol" : 1e-2}

    options["smoothed"] = int(sys.argv[4])
    options["defaultPHp"] = float(sys.argv[5])
    options["defaultPHbeta"] = float(sys.argv[6])

    options["rho_setter_kwargs"] = {"RF": options["defaultPHrho"]}

    all_scenario_names = list()
    for sn in range(ScenCount):
        all_scenario_names.append("Scenario"+str(sn+1))


    ph = mpisppy.opt.ph.PH(
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement,
        scenario_creator_kwargs={"scenario_count": ScenCount},
        rho_setter=_rho_setter, 
        ph_converger = PrimalDualConverger,
        extensions=XhatClosest,
    )
    
    ph.ph_main()
    variables = ph.gather_var_values_to_rank0()
    for (scenario_name, variable_name) in variables:
        variable_value = variables[scenario_name, variable_name]
        print(scenario_name, variable_name, variable_value)

    if ph.tree_solution_available:
        print(f"Final objective from XhatClosest: {ph.extobject._final_xhat_closest_obj}")
    else:
        print(f"Final objective from XhatClosest: {float('inf')}")

    

