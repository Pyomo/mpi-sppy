from mpisppy.opt.ph import PH
from mpisppy.convergers.primal_dual_converger import PrimalDualConverger
import farmer
from mpisppy.extensions.xhatclosest import XhatClosest


def main():
    num_scen = 3
    crops_multiplier = 1
    options = {
        "solver_name": "xpress_direct",
        "PHIterLimit": 300,
        "defaultPHrho": 1,
        "convthresh": -1e-8,
        "verbose": False,
        "display_progress": True,
        "display_timing": True,
        "iter0_solver_options": dict(),
        "iterk_solver_options": dict(),
        "display_convergence_detail": True,
        "smoothed": True,
        "defaultPHp": .1,
        "defaultPHbeta": 0.1,
        "primal_dual_converger_options" : {"tol" : 1e-6},
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