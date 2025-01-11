###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# This software is distributed under the 3-clause BSD License.
# Started by dlw Aug 2023

import farmer_pyomo_agnostic
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.config as config
import mpisppy.agnostic.agnostic as agnostic

def _farmer_parse_args():
    # create a config object and parse JUST FOR TESTING
    cfg = config.Config()

    farmer_pyomo_agnostic.inparser_adder(cfg)

    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()    
    cfg.aph_args()    
    cfg.xhatlooper_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.lagranger_args()
    cfg.xhatshuffle_args()

    cfg.parse_command_line("farmer_pyomo_agnostic_cylinders")
    return cfg


if __name__ == "__main__":
    print("begin ad hoc main for agnostic.py")

    cfg = _farmer_parse_args()
    Ag = agnostic.Agnostic(farmer_pyomo_agnostic, cfg)

    scenario_creator = Ag.scenario_creator
    scenario_denouement = farmer_pyomo_agnostic.scenario_denouement   # should we go though Ag?
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

    ph = hub_dict["opt_class"](**hub_dict["opt_kwargs"])

    conv, obj, triv = ph.ph_main()
    # Obj includes prox (only ok if we find a non-ant soln)
    if (conv < 1e-8):
        print(f'Objective value: {obj:.2f}')
    else:
        print('Did not find a non-anticipative solution '
             f'(conv = {conv:.1e})')
    
    ph.post_solve_bound(verbose=False)
