# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# general example driver for farmer with cylinders

### Modified 2023 by U.Naepels: example to compute grad cost and rhos from gradient and use the corresponding rho setter
# mpiexec -np 2 python -m mpi4py farmer_demo.py  --num-scens 3 --bundles-per-rank=0 --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME} --xhatpath=./xhat.npy --rhopath= --grad-rho-setter --order-stat=


import farmer
import mpisppy.cylinders

# Make it all go
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.sputils as sputils

from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla

from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
from mpisppy.convergers.norm_rho_converger import NormRhoConverger
import mpisppy.utils.gradient as grad
import mpisppy.utils.find_rho as find_rho
from mpisppy.utils.wxbarwriter import WXBarWriter

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
    cfg.xhatshuffle_args()
    cfg.gradient_args() #required to use gradient
    cfg.rho_args()
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

    num_scen = cfg.num_scens
    crops_multiplier = cfg.crops_mult
    if cfg.rho_setter:
        print("Rhos are set from rho file")
        rho_setter = find_rho.Set_Rhos(cfg).rho_setter
    else:
        rho_setter = None

    if cfg.default_rho is None and rho_setter is None:
        raise RuntimeError("No rho_setter so a default must be specified via --default-rho")

    if cfg.use_norm_rho_converger:
        if not cfg.use_norm_rho_updater:
            raise RuntimeError("--use-norm-rho-converger requires --use-norm-rho-updater")
        else:
            ph_converger = NormRhoConverger
    else:
        ph_converger = None
    
    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement
    all_scenario_names = farmer.scenario_names_creator(cfg.num_scens)
    #all_scenario_names = ['scen{}'.format(sn) for sn in range(num_scen)]
    scenario_creator_kwargs = {
        'use_integer': False,
        "crops_multiplier": crops_multiplier,
    }
    #scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]

    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    if cfg.run_async:
        # Vanilla APH hub
        hub_dict = vanilla.aph_hub(*beans,
                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                   ph_extensions=None,
                                   rho_setter=rho_setter)
    else:
        # Vanilla PH hub
        hub_dict = vanilla.ph_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  ph_extensions=WXBarWriter,
                                  ph_converger=ph_converger,
                                  rho_setter=rho_setter)
        
    #write Ws in csv file
    hub_dict['opt_kwargs']['options']["W_and_xbar_writer"] =  {"Wcsvdir": "Wdir"}
    hub_dict['opt_kwargs']['options']["W_fname"] =  "w_fname.csv"
    
    ## hack in adaptive rho
    if cfg.use_norm_rho_updater:
        hub_dict['opt_kwargs']['extensions'] = NormRhoUpdater
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
    if cfg.xhatlooper:
        list_of_spoke_dict.append(xhatlooper_spoke)
    if cfg.xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()
    if wheel.strata_rank == 0:  # don't do this for bound ranks
        ph_object = wheel.spcomm.opt

    grad.grad_cost_and_rho('farmer', cfg)
    find_rho.get_rhos_from_Ws('farmer', cfg)

    if write_solution:
        wheel.write_first_stage_solution('farmer_plant.csv')
        wheel.write_first_stage_solution('farmer_cyl_nonants.npy',
                                         first_stage_solution_writer=sputils.first_stage_nonant_npy_serializer)
        wheel.write_tree_solution('farmer_full_solution')

if __name__ == "__main__":
    main()
