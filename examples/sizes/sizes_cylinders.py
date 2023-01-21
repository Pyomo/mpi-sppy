# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import sys
import copy
import sizes

from mpisppy.utils import config
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.extensions.fixer import Fixer
import mpisppy.utils.cfg_vanilla as vanilla

def _parse_args():
    cfg = config.Config()
    
    cfg.popular_args()
    cfg.num_scens_required()  # but not positional: you need --num-scens
    cfg.ph_args()
    cfg.two_sided_args()
    cfg.mip_options()
    cfg.fixer_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.xhatlooper_args()
    cfg.xhatshuffle_args()
    cfg.xhatxbar_args()

    cfg.parse_command_line("sizes_cylinders")
    return cfg

def main():
    
    cfg = _parse_args()

    num_scen = cfg.num_scens

    fwph = cfg.fwph
    xhatlooper = cfg.xhatlooper
    xhatshuffle = cfg.xhatshuffle
    xhatxbar = cfg.xhatxbar
    lagrangian = cfg.lagrangian
    fixer = cfg.fixer
    fixer_tol = cfg.fixer_tol

    if num_scen not in (3, 10):
        raise RuntimeError(f"num_scen must the 3 or 10; was {num_scen}")
    
    scenario_creator_kwargs = {"scenario_count": num_scen}
    scenario_creator = sizes.scenario_creator
    scenario_denouement = sizes.scenario_denouement
    all_scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]
    rho_setter = sizes._rho_setter
    
    if fixer:
        ph_ext = Fixer
    else:
        ph_ext = None

    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)        
    # Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              scenario_creator_kwargs=scenario_creator_kwargs,
                              ph_extensions=ph_ext,
                              rho_setter = rho_setter)

    if fixer:
        hub_dict["opt_kwargs"]["options"]["fixeroptions"] = {
            "verbose": False,
            "boundtol": fixer_tol,
            "id_fix_list_fct": sizes.id_fix_list_fct,
        }
    if cfg.default_rho is None:
        # since we are using a rho_setter anyway
        hub_dict.opt_kwargs.options["defaultPHrho"] = 1  
    
    # FWPH spoke
    if fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # Standard Lagrangian bound spoke
    if lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(
            *beans,
            scenario_creator_kwargs=scenario_creator_kwargs,
            rho_setter=rho_setter,
        )

    # xhat looper bound spoke
    if xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(
            *beans,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )

    # xhat shuffle bound spoke
    if xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(
            *beans,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )
       
    # xhat using xbar bound spoke
    if xhatxbar:
        xhatxbar_spoke = vanilla.xhatxbar_spoke(
            *beans,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )
       
    list_of_spoke_dict = list()
    if fwph:
        list_of_spoke_dict.append(fw_spoke)
    if lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if xhatlooper:
        list_of_spoke_dict.append(xhatlooper_spoke)
    if xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)
    if xhatxbar:
        list_of_spoke_dict.append(xhatxbar_spoke)

    WheelSpinner(hub_dict, list_of_spoke_dict).spin()


if __name__ == "__main__":
    main()
