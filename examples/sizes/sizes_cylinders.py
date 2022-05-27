# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import sys
import copy
import sizes

from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.extensions.fixer import Fixer
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla

def _parse_args():
    config.popular_args()
    config.num_scens_required()  # but not positional: you need --num-scens
    config.ph_args()
    config.two_sided_args()
    config.mip_options()
    config.fixer_args()
    config.fwph_args()
    config.lagrangian_args()
    config.xhatlooper_args()
    config.xhatshuffle_args()
    parser = config.create_parser("netdes")
    args = parser.parse_args()  # from the command line
    args = config.global_config.import_argparse(args)


def main():
    
    _parse_args()
    cfg = config.global_config  # typing aid

    num_scen = cfg.num_scens

    fwph = cfg.fwph
    xhatlooper = cfg.xhatlooper
    xhatshuffle = cfg.xhatshuffle
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
       
    list_of_spoke_dict = list()
    if fwph:
        list_of_spoke_dict.append(fw_spoke)
    if lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if xhatlooper:
        list_of_spoke_dict.append(xhatlooper_spoke)
    if xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    WheelSpinner(hub_dict, list_of_spoke_dict).spin()


if __name__ == "__main__":
    main()
