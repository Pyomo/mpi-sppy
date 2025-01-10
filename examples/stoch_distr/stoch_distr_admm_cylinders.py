###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# general example driver for stoch_distr with cylinders

# Driver file for stochastic admm
import mpisppy.utils.stoch_admmWrapper as stoch_admmWrapper

import examples.distr.distr_data as distr_data
import stoch_distr

from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.sputils as sputils
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy import MPI
global_rank = MPI.COMM_WORLD.Get_rank()

write_solution = False


def _parse_args():
    # create a config object and parse
    cfg = config.Config()
    stoch_distr.inparser_adder(cfg)
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
    cfg.add_to_config("json_file_path",
                         description="JSON file with the data paramaters (default ../distr/data_params.json)",
                         domain=str,
                         default="../distr/data_params.json")

    cfg.parse_command_line("stoch_distr_admm_cylinders")
    return cfg


def _count_cylinders(cfg):
    count = 1
    cfglist = ["xhatxbar", "lagrangian", "ph_ob"] # All the cfg arguments that create a new cylinders
    # Add to this list any cfg attribute that would create a spoke
    for cylname in cfglist:
        if cfg[cylname]:
            count += 1
    return count


def _make_admm(cfg, n_cylinders, all_nodes_dict, inter_region_dict, data_params, verbose=None):
    options = {}

    admm_subproblem_names = stoch_distr.admm_subproblem_names_creator(cfg.num_admm_subproblems)
    stoch_scenario_names = stoch_distr.stoch_scenario_names_creator(num_stoch_scens=cfg.num_stoch_scens)
    all_admm_stoch_subproblem_scenario_names = stoch_distr.admm_stoch_subproblem_scenario_names_creator(admm_subproblem_names,stoch_scenario_names)
    
    split_admm_stoch_subproblem_scenario_name = stoch_distr.split_admm_stoch_subproblem_scenario_name
    
    scenario_creator = stoch_distr.scenario_creator
    scenario_creator_kwargs = stoch_distr.kw_creator(all_nodes_dict, cfg, inter_region_dict, data_params)
    stoch_scenario_name = stoch_scenario_names[0] # choice of any scenario
    consensus_vars = stoch_distr.consensus_vars_creator(admm_subproblem_names, stoch_scenario_name, **scenario_creator_kwargs)
    admm = stoch_admmWrapper.Stoch_AdmmWrapper(options,
                           all_admm_stoch_subproblem_scenario_names,
                           split_admm_stoch_subproblem_scenario_name,
                           admm_subproblem_names,
                           stoch_scenario_names,
                           scenario_creator,
                           consensus_vars,
                           n_cylinders=n_cylinders,
                           mpicomm=MPI.COMM_WORLD,
                           scenario_creator_kwargs=scenario_creator_kwargs,
                           verbose=verbose,
                           BFs=None,
                           )
    return admm, all_admm_stoch_subproblem_scenario_names
    

def _wheel_creator(cfg, n_cylinders, scenario_creator, variable_probability, all_nodenames, all_admm_stoch_subproblem_scenario_names, scenario_creator_kwargs=None): #the wrapper doesn't need any kwarg
    ph_converger = None
    #Things needed for vanilla cylinders
    scenario_denouement = stoch_distr.scenario_denouement

    beans = (cfg, scenario_creator, scenario_denouement, all_admm_stoch_subproblem_scenario_names)
    if cfg.run_async:
        # Vanilla APH hub
        hub_dict = vanilla.aph_hub(*beans,
                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                   ph_extensions=None,
                                   rho_setter = None,
                                   variable_probability=variable_probability,
                                   all_nodenames=all_nodenames) #needs to be modified, all_nodenames is None only for 2 stage problems

    else:
        # Vanilla PH hub
        hub_dict = vanilla.ph_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  ph_extensions=None,
                                  ph_converger=ph_converger,
                                  rho_setter=None,
                                  variable_probability=variable_probability,
                                  all_nodenames=all_nodenames)

    # FWPH spoke doesn't work with variable probability

    # Standard Lagrangian bound spoke
    if cfg.lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = None,
                                              all_nodenames=all_nodenames)

    # FWPH spoke : does not work here but may be called in global model.
    if cfg.fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # ph outer bounder spoke
    if cfg.ph_ob:
        ph_ob_spoke = vanilla.ph_ob_spoke(*beans,
                                          scenario_creator_kwargs=scenario_creator_kwargs,
                                          rho_setter = None,
                                          all_nodenames=all_nodenames,
                                          variable_probability=variable_probability)

    # xhat looper bound spoke
    if cfg.xhatxbar:
        xhatxbar_spoke = vanilla.xhatxbar_spoke(*beans,
                                                scenario_creator_kwargs=scenario_creator_kwargs,
                                                all_nodenames=all_nodenames,
                                                variable_probability=variable_probability)


    list_of_spoke_dict = list()
    if cfg.fwph:
        list_of_spoke_dict.append(fw_spoke)
    if cfg.lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if cfg.ph_ob:
        list_of_spoke_dict.append(ph_ob_spoke)
    if cfg.xhatxbar:
        list_of_spoke_dict.append(xhatxbar_spoke)

    assert n_cylinders == 1 + len(list_of_spoke_dict), f"{n_cylinders=},{len(list_of_spoke_dict)=}"

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)

    return wheel


def main(cfg):
    assert cfg.fwph_args is not None, "fwph does not support variable probability"

    if cfg.default_rho is None: # and rho_setter is None
        raise RuntimeError("No rho_setter so a default must be specified via --default-rho")

    if cfg.scalable:
        import json
        json_file_path = cfg.json_file_path

        # Read the JSON file
        with open(json_file_path, 'r') as file:

            data_params = json.load(file)
            # In distr_data num_admm_subproblems is called num_scens
            cfg.add_to_config("num_scens",
                      description="num admm subproblems",
                      domain=int,
                      default=cfg.num_admm_subproblems)
            
            all_nodes_dict = distr_data.all_nodes_dict_creator(cfg, data_params)
            all_DC_nodes = [DC_node for region in all_nodes_dict for DC_node in all_nodes_dict[region]["distribution center nodes"]]
            inter_region_dict = distr_data.scalable_inter_region_dict_creator(all_DC_nodes, cfg, data_params)

    else:
        inter_region_dict = distr_data.inter_region_dict_creator(num_scens=cfg.num_admm_subproblems)
        all_nodes_dict = None
        data_params = {"max revenue": 1200} # hard-coded because the model is hard-coded

    n_cylinders = _count_cylinders(cfg)
    admm, all_admm_stoch_subproblem_scenario_names = _make_admm(cfg, n_cylinders, all_nodes_dict, inter_region_dict, data_params)
    
    scenario_creator = admm.admmWrapper_scenario_creator # scenario_creator on a local scale
    #note that the stoch_admmWrapper scenario_creator wrapper doesn't take any arguments
    variable_probability = admm.var_prob_list
    all_nodenames = admm.all_nodenames
    wheel = _wheel_creator(cfg, n_cylinders, scenario_creator, variable_probability, all_nodenames, all_admm_stoch_subproblem_scenario_names)

    wheel.spin()
    if write_solution:
        wheel.write_first_stage_solution('stoch_distr_soln.csv')
        wheel.write_first_stage_solution('stoch_distr_cyl_nonants.npy',
                first_stage_solution_writer=sputils.first_stage_nonant_npy_serializer)
        wheel.write_tree_solution('stoch_distr_full_solution')
    if global_rank == 0:
        best_objective = wheel.spcomm.BestInnerBound #* len(all_admm_stoch_subproblem_scenario_names)
        print(f"{best_objective=}")


if __name__ == "__main__":
    cfg = _parse_args()
    main(cfg)
