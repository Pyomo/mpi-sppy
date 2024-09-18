###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# general example driver for distr with cylinders
import mpisppy.utils.admmWrapper as admmWrapper
import distr_data
import distr

from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.sputils as sputils
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy import MPI
import time

global_rank = MPI.COMM_WORLD.Get_rank()

write_solution = False

def _parse_args():
    # create a config object and parse
    cfg = config.Config()
    distr.inparser_adder(cfg)
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.aph_args()
    cfg.xhatxbar_args()
    cfg.lagrangian_args()
    cfg.ph_ob_args()
    cfg.tracking_args()
    cfg.add_to_config("run_async",
                         description="Run with async projective hedging instead of progressive hedging",
                         domain=bool,
                         default=False)

    cfg.parse_command_line("distr_admm_cylinders")
    return cfg


# This need to be executed long before the cylinders are created
def _count_cylinders(cfg): 
    count = 1
    cfglist = ["xhatxbar", "lagrangian", "ph_ob"] # All the cfg arguments that create a new cylinders
    # Add to this list any cfg attribute that would create a spoke
    for cylname in cfglist:
        if cfg[cylname]:
            count += 1
    return count


def main():

    cfg = _parse_args()

    if cfg.default_rho is None: # and rho_setter is None
        raise RuntimeError("No rho_setter so a default must be specified via --default-rho")

    if cfg.scalable:
        import json
        json_file_path = "data_params.json"

        # Read the JSON file
        with open(json_file_path, 'r') as file:
            start_time_creating_data = time.time()

            data_params = json.load(file)
            all_nodes_dict = distr_data.all_nodes_dict_creator(cfg, data_params)
            all_DC_nodes = [DC_node for region in all_nodes_dict for DC_node in all_nodes_dict[region]["distribution center nodes"]]
            inter_region_dict = distr_data.scalable_inter_region_dict_creator(all_DC_nodes, cfg, data_params)
            end_time_creating_data = time.time()
            creating_data_time = end_time_creating_data - start_time_creating_data
            print(f"{creating_data_time=}")
    else:
        inter_region_dict = distr_data.inter_region_dict_creator(num_scens=cfg.num_scens)
        all_nodes_dict = None
        data_params = {"max revenue": 1200} # hard-coded because the model is hard-coded

    ph_converger = None

    options = {}
    all_scenario_names = distr.scenario_names_creator(num_scens=cfg.num_scens)
    scenario_creator = distr.scenario_creator
    scenario_creator_kwargs = distr.kw_creator(all_nodes_dict, cfg, inter_region_dict, data_params)  
    consensus_vars = distr.consensus_vars_creator(cfg.num_scens, inter_region_dict, all_scenario_names)
    n_cylinders = _count_cylinders(cfg)
    admm = admmWrapper.AdmmWrapper(options,
                           all_scenario_names, 
                           scenario_creator,
                           consensus_vars,
                           n_cylinders=n_cylinders,
                           mpicomm=MPI.COMM_WORLD,
                           scenario_creator_kwargs=scenario_creator_kwargs,
                           )

    # Things needed for vanilla cylinders
    scenario_creator = admm.admmWrapper_scenario_creator ##change needed because of the wrapper
    scenario_creator_kwargs = None
    scenario_denouement = distr.scenario_denouement
    #note that the admmWrapper scenario_creator wrapper doesn't take any arguments
    variable_probability = admm.var_prob_list

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

    # FWPH spoke DOES NOT WORK with variable probability

    # Standard Lagrangian bound spoke
    if cfg.lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = None)


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
        best_objective = wheel.spcomm.BestInnerBound
        print(f"{best_objective=}")


if __name__ == "__main__":
    main()
