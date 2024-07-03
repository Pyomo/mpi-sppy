from mpisppy.opt.ph import PH
from mpisppy.convergers.primal_dual_converger import PrimalDualConverger
import farmer
from mpisppy.extensions.xhatclosest import XhatClosest
import sys


def main():
    if len(sys.argv) != 6 and len(sys.argv) != 9:
        print("usage python farmer_ph_multi.py {crops_multiplier} {scen_count} {rho} {itermax} {solver_name}")
        print("e.g., python farmer_ph_multi.py 1 3 1 300 xpress")
        print("or python farmer_ph_multi.py {crops_multiplier} {scen_count} {rho} {itermax} {smooth_type} {pvalue_or_pratio} {beta} {solver_name}")
        print("e.g., python farmer_ph_multi.py 1 3 1 300 1 0.1 0.1 xpress")
        print("Your length:", len(sys.argv))
        quit()
    crops_multiplier = int(sys.argv[1])
    num_scen = int(sys.argv[2])
    rho = float(sys.argv[3])
    itermax = int(sys.argv[4])
    solver_name = sys.argv[-1]
    if len(sys.argv) == 9:
        smooth_type = int(sys.argv[5])
        pvalue = float(sys.argv[6])
        beta = float(sys.argv[7])
    elif len(sys.argv) == 6:
        smooth_type = 0
        pvalue = 0.0
        beta = 0.0
    options = {
        "solver_name": solver_name,
        "PHIterLimit": itermax,
        "defaultPHrho": rho,
        "convthresh": -1e-8,
        "verbose": True,
        "display_progress": True,
        "display_timing": True,
        "iter0_solver_options": dict(),
        "iterk_solver_options": dict(),
        "display_convergence_detail": True,
        "smoothed": smooth_type,
        "defaultPHp": pvalue,
        "defaultPHbeta": beta,
        "primal_dual_converger_options" : {"tol" : 1e-4},
        'xhat_closest_options': {'xhat_solver_options': {}, 'keep_solution':True}
    }
    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement
    all_scenario_names = ['scen{}'.format(sn) for sn in range(num_scen)]
    scenario_creator_kwargs = {
        'use_integer': False,
        "crops_multiplier": crops_multiplier,
    }
    ph = PH(
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement,
        scenario_creator_kwargs=scenario_creator_kwargs,
        extensions = XhatClosest,
        ph_converger = PrimalDualConverger
    )
    ph.ph_main()
    variables = ph.gather_var_values_to_rank0()
    for (scenario_name, variable_name) in variables:
        variable_value = variables[scenario_name, variable_name]
        print(scenario_name, variable_name, variable_value)
    
    if ph.tree_solution_available:
        print(f"Final objective from XhatClosest: {ph.extobject._final_xhat_closest_obj}")


if __name__=='__main__':
    main()