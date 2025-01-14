###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# This software is distributed under the 3-clause BSD License.

import farmer_gurobipy_agnostic
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.config as config
import mpisppy.agnostic.agnostic as agnostic
import mpisppy.utils.sputils as sputils

def _farmer_parse_args():
    # create a config object and parse JUST FOR TESTING
    cfg = config.Config()

    farmer_gurobipy_agnostic.inparser_adder(cfg)

    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()    
    cfg.aph_args()    
    cfg.xhatlooper_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.lagranger_args()
    cfg.xhatshuffle_args()

    cfg.parse_command_line("farmer_gurobipy_agnostic_cylinders")
    return cfg


if __name__ == "__main__":
    print("begin ad hoc main for agnostic.py")

    cfg = _farmer_parse_args()
    Ag = agnostic.Agnostic(farmer_gurobipy_agnostic, cfg)

    scenario_creator = Ag.scenario_creator
    scenario_denouement = farmer_gurobipy_agnostic.scenario_denouement   # should we go though Ag?
    all_scenario_names = ['scen{}'.format(sn) for sn in range(cfg.num_scens)]

    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    # Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              scenario_creator_kwargs=None,  # kwargs in Ag not here
                              ph_extensions=None,
                              ph_converger=None,
                              rho_setter = None)
    # pass the Ag object via options...
    hub_dict["opt_kwargs"]["options"]["Ag"] = Ag

    # xhat shuffle bound spoke
    if cfg.xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, scenario_creator_kwargs=None)
        xhatshuffle_spoke["opt_kwargs"]["options"]["Ag"] = Ag
    if cfg.lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans, scenario_creator_kwargs=None)       
        lagrangian_spoke["opt_kwargs"]["options"]["Ag"] = Ag

    list_of_spoke_dict = list()
    if cfg.xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)
    if cfg.lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
        
    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    write_solution = False
    if write_solution:
        wheel.write_first_stage_solution('farmer_plant.csv')
        wheel.write_first_stage_solution('farmer_cyl_nonants.npy',
                first_stage_solution_writer=sputils.first_stage_nonant_npy_serializer)
        wheel.write_tree_solution('farmer_full_solution')

