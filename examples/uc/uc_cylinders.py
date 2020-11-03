# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

# TBD: put in more options: threads, mipgaps for spokes

# There is  manipulation of the mip gap,
#  so we need modifications of the vanilla dicts.
# Notice also that this uses MutliPHExtensions
import sys
import json
import mpisppy.examples.uc.uc_funcs as uc

from mpisppy.utils.sputils import spin_the_wheel
from mpisppy.extensions.extension import MultiPHExtension
from mpisppy.extensions.fixer import Fixer
from mpisppy.extensions.mipgapper import Gapper
from mpisppy.extensions.xhatclosest import XhatClosest
from mpisppy.examples import baseparsers
from mpisppy.examples import vanilla
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
    
    cb_data = {"scenario-count": num_scen,
               "path": str(num_scen)+"scenarios_r1"}
    scenario_creator = uc.scenario_creator
    scenario_denouement = uc.scenario_denouement
    all_scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]
    rho_setter = uc._rho_setter
    
    # Things needed for vanilla cylinders
    beans = (args, scenario_creator, scenario_denouement, all_scenario_names)

    ### start ph spoke ###
    # Start with Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              cb_data=cb_data,
                              ph_extensions=MultiPHExtension,
                              rho_setter = rho_setter)

    # Extend and/or correct the vanilla dictionary
    ext_classes =  [Gapper]
    if with_fixer:
        ext_classes.append(Fixer)
    if with_cross_scenario_cuts:
        ext_classes.append(CrossScenarioExtension)
    if args.xhat_closest_tree:
        ext_classes.append(XhatClosest)
        
    hub_dict["opt_kwargs"]["PH_extension_kwargs"] = {"ext_classes" : ext_classes}
    if with_cross_scenario_cuts:
        hub_dict["opt_kwargs"]["PHoptions"]["cross_scen_options"]\
            = {"check_bound_improve_iterations" : args.cross_scenario_iter_cnt}

    if with_fixer:
        hub_dict["opt_kwargs"]["PHoptions"]["fixeroptions"] = {
            "verbose": args.with_verbose,
            "boundtol": fixer_tol,
            "id_fix_list_fct": uc.id_fix_list_fct,
        }
    if args.xhat_closest_tree:
        hub_dict["opt_kwargs"]["PHoptions"]["xhat_closest_options"] = {
            "xhat_solver_options" : dict(),
            "keep_solution" : True
        }

    if args.ph_mipgaps_json is not None:
        with open(args.ph_mipgaps_json) as fin:
            din = json.load(fin)
        mipgapdict = {int(i): din[i] for i in din}
    else:
        mipgapdict = None
    hub_dict["opt_kwargs"]["PHoptions"]["gapperoptions"] = {
        "verbose": args.with_verbose,
        "mipgapdict": mipgapdict
        }
        
    if args.default_rho is None:
        # since we are using a rho_setter anyway
        hub_dict.opt_kwargs.PHoptions["defaultPHrho"] = 1  
    ### end ph spoke ###
    
    # FWPH spoke
    if with_fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, cb_data=cb_data)

    # Standard Lagrangian bound spoke
    if with_lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              cb_data=cb_data,
                                              rho_setter = rho_setter)

    # xhat looper bound spoke
    if with_xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(*beans, cb_data=cb_data)

    # xhat shuffle bound spoke
    if with_xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, cb_data=cb_data)
       
    # cross scenario cut spoke
    if with_cross_scenario_cuts:
        cross_scenario_cut_spoke = vanilla.cross_scenario_cut_spoke(*beans, cb_data=cb_data)

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
        list_of_spoke_dict.append(cross_scenario_cut_spoke)

    spcomm, opt_dict = spin_the_wheel(hub_dict, list_of_spoke_dict)

    if args.solution_dir is not None:
        uc.write_solution(spcomm, opt_dict, args.solution_dir)


if __name__ == "__main__":
    main()
