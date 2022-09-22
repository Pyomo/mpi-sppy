# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# general example driver for the hydro example with cylinders

import hydro
import hydro_cylinders

from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.utils.pysp_model import PySPModel

import mpisppy.cylinders as cylinders

write_solution = True

def main():

    cfg = hydro_cylinders._parse_args()  # we will ignore the branching factors

    xhatshuffle = cfg.xhatshuffle
    lagrangian = cfg.lagrangian

    # This is multi-stage, so we need to supply node names
    hydro = PySPModel("./PySP/models/", "./PySP/nodedata/")
    rho_setter = None
    
    # Things needed for vanilla cylinders
    beans = (cfg, hydro.scenario_creator, hydro.scenario_denouement, hydro.all_scenario_names)
    
    # Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              ph_extensions=None,
                              rho_setter = rho_setter,
                              all_nodenames=hydro.all_nodenames)

    # Standard Lagrangian bound spoke
    if lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              rho_setter = rho_setter,
                                              all_nodenames=hydro.all_nodenames)

    if xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans,
                                                      hydro.all_nodenames)

    list_of_spoke_dict = list()
    if lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    if wheel.global_rank == 0:  # we are the reporting hub rank
        print(f"BestInnerBound={wheel.BestInnerBound} and BestOuterBound={wheel.BestOuterBound}")
    
    if write_solution:
        wheel.write_first_stage_solution('hydro_first_stage.csv')
        wheel.write_tree_solution('hydro_full_solution')

    hydro.close()

if __name__ == "__main__":
    main()
