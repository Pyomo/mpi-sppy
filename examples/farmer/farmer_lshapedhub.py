# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# general example driver for farmer with cylinders and an l-shape hub
# NOTE: as of June 2020, it does not use the vanilla cylinders

import farmer

# Make it all go
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.cylinders.hub import LShapedHub
from mpisppy.opt.lshaped import LShapedMethod


def _parse_args():
    # create a Config object and parse into it
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.fwph_args()
    cfg.xhatlshaped_args()
    cfg.add_to_config("crops_mult",
                         description="There will be 3x this many crops (default 1)",
                         domain=int,
                         default=1)                

    cfg.parse_command_line("farmer_cylinders")
    return cfg


def main():
    cfg = _parse_args()

    # Need default_rho for FWPH, without you get 
    # uninitialized numeric value error
    if cfg.fwph and cfg.default_rho is None:
        print("Must specify a default_rho if using FWPH")
        quit()

    num_scen = cfg.num_scens
    crops_mult = cfg.crops_mult
    fwph = cfg.fwph
    xhatlshaped = cfg.xhatlshaped

    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement
    all_scenario_names = [f"scen{sn}" for sn in range(num_scen)]
    scenario_creator_kwargs = {
        "use_integer": False,
        "crops_multiplier": crops_mult,
    }

    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    # Options for the L-shaped method at the hub
    # Bounds only valid for 3 scenarios, I think? Need to ask Chris
    spo = None if cfg.max_solver_threads is None else {"threads": cfg.max_solver_threads}
    options = {
        "root_solver": cfg.solver_name,
        "sp_solver": cfg.solver_name,
        "sp_solver_options" : spo,
        #"valid_eta_lb": {i: -432000 for i in all_scenario_names},
        "max_iter": cfg.max_iterations,
        "verbose": False,
        "root_scenarios":[all_scenario_names[len(all_scenario_names)//2]]
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

    # xhat looper bound spoke -- any scenario will do for
    # lshaped (they're all the same)
    xhat_scenario_dict = {"ROOT": all_scenario_names[0]}
    
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
