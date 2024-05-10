# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# general example driver for distr with cylinders

# Driver file for stochastic admm
import mpisppy.utils.admm_ph as admm_ph
import distr
import mpisppy.cylinders

from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.sputils as sputils
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy import MPI
global_rank = MPI.COMM_WORLD.Get_rank()


write_solution = True

def _parse_args():
    # create a config object and parse
    cfg = config.Config()
    distr.inparser_adder(cfg)
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.aph_args()
    cfg.xhatxbar_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.ph_ob_args()
    cfg.tracking_args()
    cfg.add_to_config("run_async",
                         description="Run with async projective hedging instead of progressive hedging",
                         domain=bool,
                         default=False)

    cfg.parse_command_line("distr_admm_cylinders")
    return cfg


def _count_cylinders(cfg):
    count = 1
    cfglist = ["xhatxbar", "lagrangian", "ph_ob", "fwph"] #all the cfg arguments that create a new cylinders
    for cylname in cfglist:
        if cfg[cylname]:
            count += 1
    return count


def main():

    cfg = _parse_args()

    if cfg.default_rho is None: # and rho_setter is None
        raise RuntimeError("No rho_setter so a default must be specified via --default-rho")

    ph_converger = None

    options = {}
    all_scenario_names = distr.scenario_names_creator(num_scens=cfg.num_scens)
    scenario_creator = distr.scenario_creator
    scenario_creator_kwargs = distr.kw_creator(cfg)  
    consensus_vars = distr.consensus_vars_creator(cfg.num_scens)
    n_cylinders = _count_cylinders(cfg)
    admm = admm_ph.ADMM_PH(options,
                           all_scenario_names, 
                           scenario_creator,
                           consensus_vars,
                           n_cylinders=n_cylinders,
                           mpicomm=MPI.COMM_WORLD,
                           scenario_creator_kwargs=scenario_creator_kwargs,
                           )

    # Things needed for vanilla cylinders
    scenario_creator = admm.admm_ph_scenario_creator ##change needed because of the wrapper
    scenario_creator_kwargs = None
    scenario_denouement = distr.scenario_denouement
    #note that the admm_ph scenario_creator wrapper doesn't take any arguments
    variable_probability = admm.var_prob_list_fct

    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    if cfg.run_async:
        # Vanilla APH hub
        hub_dict = vanilla.aph_hub(*beans,
                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                   ph_extensions=None,
                                   rho_setter = None,
                                   variable_probability=variable_probability)

    else:
        # Vanilla PH hub
        hub_dict = vanilla.ph_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  ph_extensions=None,
                                  ph_converger=ph_converger,
                                  rho_setter=None,
                                  variable_probability=variable_probability)


    # FWPH spoke
    if cfg.fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # Standard Lagrangian bound spoke
    if cfg.lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = None,
                                              variable_probability=variable_probability)


    # ph outer bounder spoke
    if cfg.ph_ob:
        ph_ob_spoke = vanilla.ph_ob_spoke(*beans,
                                          scenario_creator_kwargs=scenario_creator_kwargs,
                                          rho_setter = None,
                                          variable_probability=variable_probability)

    # xhat looper bound spoke
    if cfg.xhatxbar:
        xhatxbar_spoke = vanilla.xhatxbar_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    list_of_spoke_dict = list()
    if cfg.fwph:
        list_of_spoke_dict.append(fw_spoke)
    if cfg.lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if cfg.ph_ob:
        list_of_spoke_dict.append(ph_ob_spoke)
    if cfg.xhatxbar:
        list_of_spoke_dict.append(xhatxbar_spoke)

    assert n_cylinders == 1 + len(list_of_spoke_dict), f"n_cylinders = {n_cylinders}, len(list_of_spoke_dict) = {len(list_of_spoke_dict)}"

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    if write_solution:
        wheel.write_first_stage_solution('distr_soln.csv')
        wheel.write_first_stage_solution('distr_cyl_nonants.npy',
                first_stage_solution_writer=sputils.first_stage_nonant_npy_serializer)
        wheel.write_tree_solution('distr_full_solution')
    
    if global_rank == 0:
        best_objective = wheel.spcomm.BestInnerBound * len(all_scenario_names)
        print(f"{best_objective=}")


if __name__ == "__main__":
    main()