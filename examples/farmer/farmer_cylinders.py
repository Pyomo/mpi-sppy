# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# general example driver for farmer with cylinders

import farmer
import mpisppy.cylinders

# Make it all go
import mpisppy.utils.sputils as sputils

from mpisppy.utils import baseparsers
from mpisppy.utils import vanilla

from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
from mpisppy.convergers.norm_rho_converger import NormRhoConverger

write_solution = True

def _parse_args():
    parser = baseparsers.make_parser(num_scens_reqd=True)
    parser = baseparsers.two_sided_args(parser)
    parser = baseparsers.aph_args(parser)    
    parser = baseparsers.xhatlooper_args(parser)
    parser = baseparsers.fwph_args(parser)
    parser = baseparsers.lagrangian_args(parser)
    parser = baseparsers.xhatshuffle_args(parser)
    parser.add_argument("--crops-mult",
                        help="There will be 3x this many crops (default 1)",
                        dest="crops_mult",
                        type=int,
                        default=1)                
    parser.add_argument("--use-norm-rho-updater",
                        help="Use the norm rho updater extension",
                        dest="use_norm_rho_updater",
                        action="store_true")
    parser.add_argument("--run-async",
                        help="Run with async projective hedging instead of progressive hedging",
                        dest="run_async",
                        action="store_true",
                        default=False)    
    args = parser.parse_args()
    return args


def main():
    
    args = _parse_args()

    num_scen = args.num_scens
    crops_multiplier = args.crops_mult

    rho_setter = farmer._rho_setter if hasattr(farmer, '_rho_setter') else None
    if args.default_rho is None and rho_setter is None:
        raise RuntimeError("No rho_setter so a default must be specified via --default-rho") 
    
    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement
    all_scenario_names = ['scen{}'.format(sn) for sn in range(num_scen)]
    scenario_creator_kwargs = {
        'use_integer': False,
        "crops_multiplier": crops_multiplier,
    }
    scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]

    # Things needed for vanilla cylinders
    beans = (args, scenario_creator, scenario_denouement, all_scenario_names)

    if args.run_async:
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
                                  rho_setter = rho_setter)

    ## hack in adaptive rho
    if args.use_norm_rho_updater:
        hub_dict['opt_kwargs']['extensions'] = NormRhoUpdater
        hub_dict['opt_kwargs']['options']['norm_rho_options'] = {'verbose': True}

    # FWPH spoke
    if args.with_fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # Standard Lagrangian bound spoke
    if args.with_lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = rho_setter)

    # xhat looper bound spoke
    if args.with_xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # xhat shuffle bound spoke
    if args.with_xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
        
    list_of_spoke_dict = list()
    if args.with_fwph:
        list_of_spoke_dict.append(fw_spoke)
    if args.with_lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if args.with_xhatlooper:
        list_of_spoke_dict.append(xhatlooper_spoke)
    if args.with_xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    mpisppy.cylinders.SPOKE_SLEEP_TIME = 0.1

    spcomm, opt_dict = sputils.spin_the_wheel(hub_dict, list_of_spoke_dict)

    if write_solution:
        sputils.write_spin_the_wheel_first_stage_solution(spcomm, opt_dict, 'farmer_plant.csv')
        sputils.write_spin_the_wheel_first_stage_solution(spcomm,
                                                          opt_dict,
                                                          'farmer_cyl_nonants.spy',
                                                          first_stage_solution_writer=\
                                                          sputils.first_stage_nonant_npy_serializer)
        sputils.write_spin_the_wheel_tree_solution(spcomm, opt_dict, 'farmer_full_solution')

if __name__ == "__main__":
    main()
