###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# general example driver for farmer with cylinders

import farmer

# Make it all go
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.sputils as sputils

from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla

from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
from mpisppy.convergers.norm_rho_converger import NormRhoConverger
from mpisppy.convergers.primal_dual_converger import PrimalDualConverger
from mpisppy.utils.cfg_vanilla import extension_adder

import pyomo.environ as pyo

write_solution = True

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
    cfg.converger_args()
    cfg.wxbar_read_write_args()
    cfg.tracking_args()
    cfg.reduced_costs_args()
    cfg.sep_rho_args()
    cfg.coeff_rho_args()
    cfg.dynamic_rho_args()
    cfg.add_to_config("crops_mult",
                         description="There will be 3x this many crops (default 1)",
                         domain=int,
                         default=1)
    cfg.add_to_config("use_norm_rho_updater",
                         description="Use the norm rho updater extension",
                         domain=bool,
                         default=False)
    cfg.add_to_config("run_async",
                         description="Run with async projective hedging instead of progressive hedging",
                         domain=bool,
                         default=False)

    cfg.parse_command_line("farmer_cylinders")
    return cfg


def main():

    cfg = _parse_args()

    num_scen = cfg.num_scens
    crops_multiplier = cfg.crops_mult

    rho_setter = farmer._rho_setter if hasattr(farmer, '_rho_setter') else None
    if cfg.default_rho is None and rho_setter is None:
        raise RuntimeError("No rho_setter so a default must be specified via --default-rho")

    if cfg.use_norm_rho_converger:
        if not cfg.use_norm_rho_updater:
            raise RuntimeError("--use-norm-rho-converger requires --use-norm-rho-updater")
        else:
            ph_converger = NormRhoConverger
    elif cfg.primal_dual_converger:
        ph_converger = PrimalDualConverger
    else:
        ph_converger = None

    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement
    all_scenario_names = ['scen{}'.format(sn) for sn in range(num_scen)]
    scenario_creator_kwargs = {
        'use_integer': False,
        "crops_multiplier": crops_multiplier,
        'sense': pyo.minimize
    }

    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    if cfg.run_async:
        # Vanilla APH hub
        hub_dict = vanilla.aph_hub(*beans,
                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                   ph_extensions=None,
                                   rho_setter = rho_setter)
    else:
        # Vanilla PH hub
        hub_dict = vanilla.ph_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  ph_extensions=None,
                                  ph_converger=ph_converger,
                                  rho_setter = rho_setter)

    if cfg.primal_dual_converger:
        hub_dict['opt_kwargs']['options']\
            ['primal_dual_converger_options'] = {
                'verbose': True,
                'tol': cfg.primal_dual_converger_tol,
                'tracking': True}

    ## hack in adaptive rho
    if cfg.use_norm_rho_updater:
        extension_adder(hub_dict, NormRhoUpdater)
        hub_dict['opt_kwargs']['options']['norm_rho_options'] = {'verbose': True}


    if cfg.reduced_costs:
        vanilla.add_reduced_costs_fixer(hub_dict, cfg)

    if cfg.sep_rho:
        vanilla.add_sep_rho(hub_dict, cfg)
    if cfg.coeff_rho:
        vanilla.add_coeff_rho(hub_dict, cfg)

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
        if cfg.sep_rho:
            vanilla.add_sep_rho(ph_ob_spoke, cfg)
        if cfg.coeff_rho:
            vanilla.add_coeff_rho(ph_ob_spoke, cfg)

    # xhat looper bound spoke
    if cfg.xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # xhat shuffle bound spoke
    if cfg.xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
    
    if cfg.reduced_costs:
        reduced_costs_spoke = vanilla.reduced_costs_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = None)

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
    if cfg.reduced_costs:
        list_of_spoke_dict.append(reduced_costs_spoke)

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    if write_solution:
        wheel.write_first_stage_solution('farmer_plant.csv')
        wheel.write_first_stage_solution('farmer_cyl_nonants.npy',
                first_stage_solution_writer=sputils.first_stage_nonant_npy_serializer)
        wheel.write_tree_solution('farmer_full_solution')

if __name__ == "__main__":
    main()
