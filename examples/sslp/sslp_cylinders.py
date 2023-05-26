# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import sys
import os
import copy
import sslp

from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.extensions.fixer import Fixer
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla

def _parse_args():
    cfg = config.Config()
    cfg.popular_args()
    cfg.num_scens_optional() 
    cfg.ph_args()
    cfg.add_to_config("instance_name",
                         description="sslp instance name (e.g., sslp_15_45_10)",
                         domain=str,
                         default=None,
                         argparse_args = {"required": True})

    cfg.two_sided_args()
    cfg.fixer_args()
    cfg.xhatlooper_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.xhatshuffle_args()
    cfg.parse_command_line("sslp_cylinders")
    return cfg


def main():
    cfg = _parse_args()

    inst = cfg.instance_name
    num_scen = int(inst.split("_")[-1])
    if cfg.num_scens is not None and cfg.num_scens != num_scen:
        raise RuntimeError("Argument num-scens={} does not match the number "
                           "implied by instance name={} "
                           "\n(--num-scens is not needed for sslp)")

    fwph = cfg.fwph
    fixer = cfg.fixer
    fixer_tol = cfg.fixer_tol
    xhatlooper = cfg.xhatlooper
    xhatshuffle = cfg.xhatshuffle
    lagrangian = cfg.lagrangian

    if cfg.default_rho is None:
        raise RuntimeError("The --default-rho option must be specified")

    scenario_creator_kwargs = {"data_dir": f"{sslp.__file__[:-8]}/data/{inst}/scenariodata"}
    scenario_creator = sslp.scenario_creator
    scenario_denouement = sslp.scenario_denouement    
    all_scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]

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
                              rho_setter = None)

    if fixer:
        hub_dict["opt_kwargs"]["options"]["fixeroptions"] = {
            "verbose": False,
            "boundtol": fixer_tol,
            "id_fix_list_fct": sslp.id_fix_list_fct,
        }
        
    # FWPH spoke
    if fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # Standard Lagrangian bound spoke
    if lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = None)
        
    # xhat looper bound spoke
    if xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # xhat shuffle bound spoke
    if xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
       
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
