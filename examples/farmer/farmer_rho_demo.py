###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# This program can be used in two different ways:
# Compute gradient-based cost and rho for a given problem
# Use the gradient-based rho setter which sets adaptative gradient rho for PH.
# mpiexec -np 2 python -m mpi4py farmer_rho_demo.py  --num-scens 3 --bundles-per-rank=0 --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME} --xhatpath=./xhat.npy --rhopath= --rho-setter --order-stat=
# Edited by DLW Oct 2023
# Note: norm_rho_updater is the Gabe thing

import farmer

# Make it all go
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.sputils as sputils

from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla

from mpisppy.extensions.extension import MultiExtension

from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
from mpisppy.convergers.norm_rho_converger import NormRhoConverger
from mpisppy.extensions.gradient_extension import Gradient_extension

write_solution = False

def _parse_args():
    # create a config object and parse
    cfg = config.Config()
    
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()    
    cfg.aph_args()    
    cfg.xhatlooper_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.lagranger_args()
    cfg.ph_ob_args()
    cfg.xhatshuffle_args()
    cfg.dynamic_gradient_args() # gets gradient args for free
    cfg.add_to_config("crops_mult",
                         description="There will be 3x this many crops (default 1)",
                         domain=int,
                         default=1)                
    cfg.add_to_config("use_norm_rho_updater",
                         description="Use the norm rho updater extension",
                         domain=bool,
                         default=False)
    cfg.add_to_config("use-norm-rho-converger",
                         description="Use the norm rho converger",
                         domain=bool,
                         default=False)
    cfg.add_to_config("run_async",
                         description="Run with async projective hedging instead of progressive hedging",
                         domain=bool,
                         default=False)
    cfg.add_to_config("use_norm_rho_converger",
                         description="Use the norm rho converger",
                         domain=bool,
                         default=False)

    cfg.parse_command_line("farmer_demo")
    return cfg

    
def main():
    
    cfg = _parse_args()

    crops_multiplier = cfg.crops_mult
    rho_setter = None  # non-grad rho setter?

    if cfg.default_rho is None and rho_setter is None:
        raise RuntimeError("No rho_setter so a default must be specified via --default-rho")

    if cfg.use_norm_rho_converger:
        if not cfg.use_norm_rho_updater:
            raise RuntimeError("--use-norm-rho-converger requires --use-norm-rho-updater")
        elif cfg.grad_rho:
            raise RuntimeError("You cannot have--use-norm-rho-converger and --grad-rho-setter")            
        else:
            ph_converger = NormRhoConverger
    else:
        ph_converger = None
    
    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement
    all_scenario_names = farmer.scenario_names_creator(cfg.num_scens)
    scenario_creator_kwargs = {
        'use_integer': False,
        "crops_multiplier": crops_multiplier,
    }

    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    ext_classes = []
    if cfg.grad_rho:
        ext_classes.append(Gradient_extension)

    if cfg.run_async:
        raise RuntimeError("APH not supported in this example.")
    else:
        # Vanilla PH hub
        hub_dict = vanilla.ph_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  ph_extensions=MultiExtension,
                                  ph_converger=ph_converger,
                                  rho_setter=rho_setter)  # non-grad rho setter
    hub_dict["opt_kwargs"]["extension_kwargs"] = {"ext_classes" : ext_classes}
    hub_dict['opt_kwargs']['extensions'] = MultiExtension  # DLW: ???? (seems to not matter)

    #gradient extension kwargs
    if cfg.grad_rho:
        ext_classes.append(Gradient_extension)        
        hub_dict['opt_kwargs']['options']['gradient_extension_options'] = {'cfg': cfg}
    
    ## Gabe's (way pre-pandemic) adaptive rho
    if cfg.use_norm_rho_updater:
        ext_classes.append(NormRhoUpdater)                
        hub_dict['opt_kwargs']['options']['norm_rho_options'] = {'verbose': True}

    # FWPH spoke
    if cfg.fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # Standard Lagrangian bound spoke
    if cfg.lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = rho_setter)

    # Special Lagranger bound spoke
    if cfg.lagranger:
        lagranger_spoke = vanilla.lagranger_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = rho_setter)

    # ph outer bounder spoke
    if cfg.ph_ob:
        ph_ob_spoke = vanilla.ph_ob_spoke(*beans,
                                          scenario_creator_kwargs=scenario_creator_kwargs,
                                          rho_setter = rho_setter)        

    # xhat looper bound spoke
    if cfg.xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # xhat shuffle bound spoke
    if cfg.xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
        
    list_of_spoke_dict = list()
    if cfg.fwph:
        list_of_spoke_dict.append(fw_spoke)
    if cfg.lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if cfg.lagranger:
        list_of_spoke_dict.append(lagranger_spoke)
    if cfg.ph_ob:
        list_of_spoke_dict.append(ph_ob_spoke)        
    if cfg.xhatlooper:
        list_of_spoke_dict.append(xhatlooper_spoke)
    if cfg.xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    # If you want to write W and rho in a file :
    #grad.grad_cost_and_rho('farmer', cfg)

    if write_solution:
        wheel.write_first_stage_solution('farmer_plant.csv')
        wheel.write_first_stage_solution('farmer_cyl_nonants.npy',
                                         first_stage_solution_writer=sputils.first_stage_nonant_npy_serializer)
        wheel.write_tree_solution('farmer_full_solution')

if __name__ == "__main__":
    main()
