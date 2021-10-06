# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

# TBD: put in more options: threads, mipgaps for spokes

# There is  manipulation of the mip gap,
#  so we need modifications of the vanilla dicts.
# Notice also that this uses MutliExtensions
import sys
import json
import uc_funcs as uc

import mpisppy.utils.sputils as sputils

from mpisppy.extensions.extension import MultiExtension
from mpisppy.extensions.fixer import Fixer
from mpisppy.extensions.mipgapper import Gapper
from mpisppy.extensions.xhatclosest import XhatClosest
from mpisppy.utils import baseparsers
from mpisppy.utils import vanilla
from mpisppy.extensions.cross_scen_extension import CrossScenarioExtension


def _parse_args():
    parser = baseparsers.make_parser("uc_cylinders")
    parser = baseparsers.two_sided_args(parser)
    parser = baseparsers.fixer_args(parser)
    parser = baseparsers.fwph_args(parser)
    parser = baseparsers.lagrangian_args(parser)
    parser = baseparsers.xhatlooper_args(parser)
    parser = baseparsers.xhatshuffle_args(parser)
    parser = baseparsers.cross_scenario_cuts_args(parser)
    parser.add_argument("--ph-mipgaps-json",
                        help="json file with mipgap schedule (default None)",
                        dest="ph_mipgaps_json",
                        type=str,
                        default=None)
    parser.add_argument("--solution-dir",
                        help="writes a tree solution to the provided directory"
                             " (default None)",
                        dest="solution_dir",
                        type=str,
                        default=None)
    parser.add_argument("--xhat-closest-tree",
                        help="Uses XhatClosest to compute a tree solution after"
                             " PH termination (default False)",
                        action='store_true',
                        dest='xhat_closest_tree',
                        default=False)
    args = parser.parse_args()
    return args


def main():
    
    args = _parse_args()

    num_scen = args.num_scens

    with_fwph = args.with_fwph
    with_xhatlooper = args.with_xhatlooper
    with_xhatshuffle = args.with_xhatshuffle
    with_lagrangian = args.with_lagrangian
    with_fixer = args.with_fixer
    fixer_tol = args.fixer_tol
    with_cross_scenario_cuts = args.with_cross_scenario_cuts

    scensavail = [3,5,10,25,50,100]
    if num_scen not in scensavail:
        raise RuntimeError("num-scen was {}, but must be in {}".\
                           format(num_scen, scensavail))
    
    scenario_creator_kwargs = {
        "scenario_count": num_scen,
        "path": str(num_scen) + "scenarios_r1",
    }
    scenario_creator = uc.scenario_creator
    scenario_denouement = uc.scenario_denouement
    all_scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]
    rho_setter = uc._rho_setter
    
    # Things needed for vanilla cylinders
    beans = (args, scenario_creator, scenario_denouement, all_scenario_names)

    ### start ph spoke ###
    # Start with Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              scenario_creator_kwargs=scenario_creator_kwargs,
                              ph_extensions=MultiExtension,
                              rho_setter = rho_setter)

    # Extend and/or correct the vanilla dictionary
    ext_classes =  [Gapper]
    if with_fixer:
        ext_classes.append(Fixer)
    if with_cross_scenario_cuts:
        ext_classes.append(CrossScenarioExtension)
    if args.xhat_closest_tree:
        ext_classes.append(XhatClosest)
        
    hub_dict["opt_kwargs"]["extension_kwargs"] = {"ext_classes" : ext_classes}
    if with_cross_scenario_cuts:
        hub_dict["opt_kwargs"]["options"]["cross_scen_options"]\
            = {"check_bound_improve_iterations" : args.cross_scenario_iter_cnt}

    if with_fixer:
        hub_dict["opt_kwargs"]["options"]["fixeroptions"] = {
            "verbose": args.with_verbose,
            "boundtol": fixer_tol,
            "id_fix_list_fct": uc.id_fix_list_fct,
        }
    if args.xhat_closest_tree:
        hub_dict["opt_kwargs"]["options"]["xhat_closest_options"] = {
            "xhat_solver_options" : dict(),
            "keep_solution" : True
        }

    if args.ph_mipgaps_json is not None:
        with open(args.ph_mipgaps_json) as fin:
            din = json.load(fin)
        mipgapdict = {int(i): din[i] for i in din}
    else:
        mipgapdict = None
    hub_dict["opt_kwargs"]["options"]["gapperoptions"] = {
        "verbose": args.with_verbose,
        "mipgapdict": mipgapdict
        }
        
    if args.default_rho is None:
        # since we are using a rho_setter anyway
        hub_dict.opt_kwargs.options["defaultPHrho"] = 1  
    ### end ph spoke ###
    
    # FWPH spoke
    if with_fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # Standard Lagrangian bound spoke
    if with_lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = rho_setter)

    # xhat looper bound spoke
    if with_xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # xhat shuffle bound spoke
    if with_xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
       
    # cross scenario cut spoke
    if with_cross_scenario_cuts:
        cross_scenario_cuts_spoke = vanilla.cross_scenario_cuts_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    list_of_spoke_dict = list()
    if with_fwph:
        list_of_spoke_dict.append(fw_spoke)
    if with_lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if with_xhatlooper:
        list_of_spoke_dict.append(xhatlooper_spoke)
    if with_xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)
    if with_cross_scenario_cuts:
        list_of_spoke_dict.append(cross_scenario_cuts_spoke)

    spcomm, opt_dict = sputils.spin_the_wheel(hub_dict, list_of_spoke_dict)

    if args.solution_dir is not None:
        sputils.write_spin_the_wheel_tree_solution(
                spcomm, opt_dict, args.solution_dir, uc.scenario_tree_solution_writer )
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
        sputils.write_spin_the_wheel_first_stage_solution(spcomm, opt_dict, 'uc_cyl.csv')
        sputils.write_spin_the_wheel_first_stage_solution(spcomm,
                                                          opt_dict,
                                                          'uc_cyl_nonants.spy',
                                                          first_stage_solution_writer=\
                                                          sputils.first_stage_nonant_npy_serializer)


if __name__ == "__main__":
    main()
