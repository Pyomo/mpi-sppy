# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# general example driver for farmer with cylinders

import uc_funcs as uc

# Make it all go
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.cylinders.hub import LShapedHub
from mpisppy.opt.lshaped import LShapedMethod


def _parse_args():
    cfg = config.Config()
    cfg.popular_args()
    cfg.num_scens_required() 
    cfg.ph_args()
    cfg.two_sided_args()
    cfg.fwph_args()
    cfg.xhatlshaped_args()
    cfg.parse_command_line("uc_lshaped")
    return cfg


def main():
    cfg = _parse_args()

    # Need default_rho for FWPH, without you get 
    # uninitialized numeric value error
    if cfg.fwph and cfg.default_rho is None:
        print("Must specify a default_rho if using FWPH")
        quit()

    num_scen = cfg.num_scens
    fwph = cfg.fwph
    xhatlshaped = cfg.xhatlshaped

    scenario_creator = uc.scenario_creator
    scenario_denouement = uc.scenario_denouement
    scenario_creator_kwargs = {
        "path": f"./{num_scen}scenarios_r1/"
    }
    all_scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]

    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    # Options for the L-shaped method at the hub
    spo = None if cfg.max_solver_threads is None else {"threads": cfg.max_solver_threads}
    spo['mipgap'] = 0.005
    options = {
        "root_solver": cfg.solver_name,
        "sp_solver": cfg.solver_name,
        "sp_solver_options" : spo,
        "root_solver_options" : spo,
        #"valid_eta_lb": {n:0. for n in all_scenario_names},
        "max_iter": cfg.max_iterations,
        "verbose": False,
        "root_scenarios":[all_scenario_names[len(all_scenario_names)//2]],
   }
    
    # L-shaped hub
    hub_dict = {
        "hub_class": LShapedHub,
        "hub_kwargs": {
            "options": {
                "rel_gap": cfg.rel_gap,
                "abs_gap": cfg.abs_gap,
            },
        },
        "opt_class": LShapedMethod,
        "opt_kwargs": { # Args passed to LShapedMethod __init__
            "options": options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
        },
    }

    # FWPH spoke
    if fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
        fw_spoke["opt_kwargs"]["PH_options"]["abs_gap"] = 0.
        fw_spoke["opt_kwargs"]["PH_options"]["rel_gap"] = 1e-5
        fw_spoke["opt_kwargs"]["rho_setter"] = uc.scenario_rhos

    if xhatlshaped:
        xhatlshaped_spoke = vanilla.xhatlshaped_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    list_of_spoke_dict = list()
    if fwph:
        list_of_spoke_dict.append(fw_spoke)
    if xhatlshaped:
        list_of_spoke_dict.append(xhatlshaped_spoke)

    WheelSpinner(hub_dict, list_of_spoke_dict).spin()


if __name__ == "__main__":
    main()
