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
from mpisppy.spin_the_wheel import WheelSpinner

from mpisppy.extensions.extension import MultiExtension
from mpisppy.extensions.fixer import Fixer
from mpisppy.extensions.mipgapper import Gapper
from mpisppy.extensions.xhatclosest import XhatClosest
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.extensions.cross_scen_extension import CrossScenarioExtension

def _parse_args():
    cfg = config.Config()
    cfg.popular_args()
    cfg.num_scens_required() 
    cfg.ph_args()
    cfg.two_sided_args()
    cfg.aph_args()        
    cfg.fixer_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.xhatlooper_args()
    cfg.xhatshuffle_args()
    cfg.cross_scenario_cuts_args()
    cfg.add_to_config("ph_mipgaps_json",
                         description="json file with mipgap schedule (default None)",
                         domain=str,
                         default=None)
    cfg.add_to_config("solution_dir",
                         description="writes a tree solution to the provided directory"
                                      " (default None)",
                         domain=str,
                         default=None)
    cfg.add_to_config("xhat_closest_tree",
                         description="Uses XhatClosest to compute a tree solution after"
                                     " PH termination (default False)",
                         domain=bool,
                         default=False)
    cfg.add_to_config("run_aph",
                         description="Run with async projective hedging instead of progressive hedging",
                         domain=bool,
                         default=False)        
    cfg.parse_command_line("uc_cylinders")
    return cfg


def main():
    
    cfg = _parse_args()

    num_scen = cfg.num_scens

    fwph = cfg.fwph
    xhatlooper = cfg.xhatlooper
    xhatshuffle = cfg.xhatshuffle
    lagrangian = cfg.lagrangian
    fixer = cfg.fixer
    fixer_tol = cfg.fixer_tol
    cross_scenario_cuts = cfg.cross_scenario_cuts

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
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    ### start ph spoke ###
    if cfg.run_aph:
        hub_dict = vanilla.aph_hub(*beans,
                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                   ph_extensions=MultiExtension,
                                   rho_setter = rho_setter)
    else:
        hub_dict = vanilla.ph_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  ph_extensions=MultiExtension,
                                  rho_setter = rho_setter)
        
    # Extend and/or correct the vanilla dictionary
    ext_classes =  [Gapper]
    if fixer:
        ext_classes.append(Fixer)
    if cross_scenario_cuts:
        ext_classes.append(CrossScenarioExtension)
    if cfg.xhat_closest_tree:
        ext_classes.append(XhatClosest)
        
    hub_dict["opt_kwargs"]["extension_kwargs"] = {"ext_classes" : ext_classes}
    if cross_scenario_cuts:
        hub_dict["opt_kwargs"]["options"]["cross_scen_options"]\
            = {"check_bound_improve_iterations" : cfg.cross_scenario_iter_cnt}

    if fixer:
        hub_dict["opt_kwargs"]["options"]["fixeroptions"] = {
            "verbose": cfg.verbose,
            "boundtol": fixer_tol,
            "id_fix_list_fct": uc.id_fix_list_fct,
        }
    if cfg.xhat_closest_tree:
        hub_dict["opt_kwargs"]["options"]["xhat_closest_options"] = {
            "xhat_solver_options" : dict(),
            "keep_solution" : True
        }

    if cfg.ph_mipgaps_json is not None:
        with open(cfg.ph_mipgaps_json) as fin:
            din = json.load(fin)
        mipgapdict = {int(i): din[i] for i in din}
    else:
        mipgapdict = None
    hub_dict["opt_kwargs"]["options"]["gapperoptions"] = {
        "verbose": cfg.verbose,
        "mipgapdict": mipgapdict
        }
        
    if cfg.default_rho is None:
        # since we are using a rho_setter anyway
        hub_dict.opt_kwcfg.options["defaultPHrho"] = 1  
    ### end ph spoke ###
    
    # FWPH spoke
    if fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # Standard Lagrangian bound spoke
    if lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = rho_setter)

    # xhat looper bound spoke
    if xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # xhat shuffle bound spoke
    if xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
       
    # cross scenario cut spoke
    if cross_scenario_cuts:
        cross_scenario_cuts_spoke = vanilla.cross_scenario_cuts_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    list_of_spoke_dict = list()
    if fwph:
        list_of_spoke_dict.append(fw_spoke)
    if lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if xhatlooper:
        list_of_spoke_dict.append(xhatlooper_spoke)
    if xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)
    if cross_scenario_cuts:
        list_of_spoke_dict.append(cross_scenario_cuts_spoke)

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    if cfg.solution_dir is not None:
        wheel.write_tree_solution(cfg.solution_dir, uc.scenario_tree_solution_writer)

    wheel.write_first_stage_solution('uc_cyl_nonants.npy',
            first_stage_solution_writer=sputils.first_stage_nonant_npy_serializer)


if __name__ == "__main__":
    main()
