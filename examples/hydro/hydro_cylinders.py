# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# general example driver for the hydro example with cylinders
# Modfied April 2022 by DLW to illustrate config.py

import hydro

import mpisppy.utils.sputils as sputils

from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla

import mpisppy.cylinders as cylinders

write_solution = True

def _parse_args():
    # create a Config object, get values for it, and return it
    cfg = config.Config()
    cfg.multistage()
    cfg.ph_args()
    cfg.two_sided_args()
    cfg.xhatlooper_args()
    cfg.xhatshuffle_args()
    cfg.lagrangian_args()
    cfg.xhatspecific_args()

    cfg.add_to_config(name ="stage2EFsolvern",
                         description="Solver to use for xhatlooper stage2ef option (default None)",
                         domain = str,
                         default=None)

    cfg.parse_command_line("farmer_cylinders")
    return cfg


def main():
    cfg = _parse_args()

    _parse_args()  # updates cfg

    BFs = cfg["branching_factors"]
    if len(BFs) != 2:
        raise RuntimeError("Hydro is a three stage problem, so it needs 2 BFs")

    xhatshuffle = cfg["xhatshuffle"]
    lagrangian = cfg["lagrangian"]

    # This is multi-stage, so we need to supply node names
    all_nodenames = sputils.create_nodenames_from_branching_factors(BFs)

    ScenCount = BFs[0] * BFs[1]
    scenario_creator_kwargs = {"branching_factors": BFs}
    all_scenario_names = [f"Scen{i+1}" for i in range(ScenCount)]
    scenario_creator = hydro.scenario_creator
    scenario_denouement = hydro.scenario_denouement
    rho_setter = None
    
    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)
    
    # Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              scenario_creator_kwargs=scenario_creator_kwargs,
                              ph_extensions=None,
                              rho_setter = rho_setter,
                              all_nodenames = all_nodenames,
                             )

    # Standard Lagrangian bound spoke
    if lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                                    scenario_creator_kwargs=scenario_creator_kwargs,
                                                    rho_setter = rho_setter,
                                                    all_nodenames = all_nodenames,
                                                   )


    # xhat looper bound spoke
    
    if xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans,
                                                      all_nodenames=all_nodenames,
                                                      scenario_creator_kwargs=scenario_creator_kwargs,
                                                     )

    list_of_spoke_dict = list()
    if lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    if cfg.stage2EFsolvern is not None:
        xhatshuffle_spoke["opt_kwargs"]["options"]["stage2EFsolvern"] = cfg["stage2EFsolvern"]
        xhatshuffle_spoke["opt_kwargs"]["options"]["branching_factors"] = cfg["branching_factors"]

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    if wheel.global_rank == 0:  # we are the reporting hub rank
        print(f"BestInnerBound={wheel.BestInnerBound} and BestOuterBound={wheel.BestOuterBound}")
    
    if write_solution:
        wheel.write_first_stage_solution('hydro_first_stage.csv')
        wheel.write_tree_solution('hydro_full_solution')

if __name__ == "__main__":
    main()
