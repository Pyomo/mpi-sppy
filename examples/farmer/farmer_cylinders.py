# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# general example driver for farmer with cylinders

import farmer
import mpisppy.cylinders

# Make it all go
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.sputils as sputils

from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla

from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
from mpisppy.convergers.norm_rho_converger import NormRhoConverger

write_solution = True

def _parse_args():
    # parse and update global config
    config.popular_args()
    config.two_sided_args()
    config.ph_args()    
    config.aph_args()    
    config.xhatlooper_args()
    config.fwph_args()
    config.lagrangian_args()
    config.lagranger_args()
    config.xhatshuffle_args()
    config.add_to_config("crops_mult",
                         description="There will be 3x this many crops (default 1)",
                         domain=int,
                         default=1)                
    config.add_to_config("use_norm_rho_updater",
                         description="Use the norm rho updater extension",
                         domain=bool,
                         default=False)
    config.add_to_config("use-norm-rho-converger",
                         description="Use the norm rho converger",
                         domain=bool,
                         default=False)
    config.add_to_config("run_async",
                         description="Run with async projective hedging instead of progressive hedging",
                         domain=bool,
                         default=False)
    config.add_to_config("use_norm_rho_converger",
                         description="Use the norm rho converger",
                         domain=bool,
                         default=False)
    # note that num_scens is special until Pyomo config supports positionals
    config.add_to_config("num_scens",
                         description="Number of Scenarios (required, positional)",
                         domain=int,
                         default=-1,
                         argparse=False)   # special
    
    parser = config.create_parser("farmer_cylinders")
    # more special treatment of num_scens
    parser.add_argument(
        "num_scens", help="Number of scenarios", type=int
    )
    
    args = parser.parse_args()  # from the command line
    args = config.global_config.import_argparse(args)

    # final special treatment of num_scens
    config.global_config.num_scens = args.num_scens

    
def main():
    
    _parse_args()
    cfg = config.global_config

    num_scen = cfg.num_scens
    crops_multiplier = cfg.crops_mult

    rho_setter = farmer._rho_setter if hasattr(farmer, '_rho_setter') else None
    if cfg.default_rho is None and rho_setter is None:
        raise RuntimeError("No rho_setter so a default must be specified via --default-rho")

    if cfg.use_norm_rho_converger:
        if not cfg.use_norm_rho_updater:
            raise RuntimeError("--use-norm-rho-converger requires --use-norm-rho-updater")
        else:
            ph_converger = NormRhoConverger
    else:
        ph_converger = None
    
    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement
    all_scenario_names = ['scen{}'.format(sn) for sn in range(num_scen)]
    scenario_creator_kwargs = {
        'use_integer': False,
        "crops_multiplier": crops_multiplier,
    }
    scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]

    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    if cfg.run_async:
        # Vanilla APH hub
        hub_dict = vanilla.aph_hub(*beans,
                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                   ph_extensions=None,
                                   rho_setter = rho_setter)
    else:
        # Vanilla PH hub
        hub_dict = vanilla.ph_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  ph_extensions=None,
                                  ph_converger=ph_converger,
                                  rho_setter = rho_setter)

    ## hack in adaptive rho
    if cfg.use_norm_rho_updater:
        hub_dict['opt_kwargs']['extensions'] = NormRhoUpdater
        hub_dict['opt_kwargs']['options']['norm_rho_options'] = {'verbose': True}

    # FWPH spoke
    if cfg.fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # Standard Lagrangian bound spoke
    if cfg.lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = rho_setter)

    # Special Lagranger bound spoke
    if cfg.lagranger:
        lagranger_spoke = vanilla.lagranger_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = rho_setter)

    # xhat looper bound spoke
    if cfg.xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # xhat shuffle bound spoke
    if cfg.xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
        
    list_of_spoke_dict = list()
    if cfg.fwph:
        list_of_spoke_dict.append(fw_spoke)
    if cfg.lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if cfg.lagranger:
        list_of_spoke_dict.append(lagranger_spoke)
    if cfg.xhatlooper:
        list_of_spoke_dict.append(xhatlooper_spoke)
    if cfg.xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    if write_solution:
        wheel.write_first_stage_solution('farmer_plant.csv')
        wheel.write_first_stage_solution('farmer_cyl_nonants.npy',
                first_stage_solution_writer=sputils.first_stage_nonant_npy_serializer)
        wheel.write_tree_solution('farmer_full_solution')

if __name__ == "__main__":
    main()
