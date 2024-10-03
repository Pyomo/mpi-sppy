# This software is distributed under the 3-clause BSD License.
# Started by dlw Aug 2023

import farmer_gams_gen_agnostic2 as farmer_gams_agnostic
#import farmer_gams_agnostic
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.config as config
import mpisppy.agnostic.agnostic as agnostic

from mpisppy import MPI
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

def _farmer_parse_args():
    # create a config object and parse JUST FOR TESTING
    cfg = config.Config()

    farmer_gams_agnostic.inparser_adder(cfg)

    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()    
    cfg.aph_args()    
    cfg.xhatlooper_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.lagranger_args()
    cfg.xhatshuffle_args()

    cfg.parse_command_line("farmer_gams_agnostic_cylinders")
    return cfg


if __name__ == "__main__":

    print("begin ad hoc main for agnostic.py")

    cfg = _farmer_parse_args()
    ### Creating the new gms file with ph included in it
    original_file = "GAMS/farmer_average.gms"
    nonants_support_set_name = "crop"
    nonant_variables_name = "x"
    nonants_name_pairs = [(nonants_support_set_name, nonant_variables_name)]


    if global_rank == 0:
        # Code for rank 0 to execute the task
        print("Global rank 0 is executing the task.")
        farmer_gams_agnostic.create_ph_model(original_file, nonants_name_pairs)
        print("Global rank 0 has completed the task.")

    # Broadcast a signal from rank 0 to all other ranks indicating the task is complete
    fullcomm.Barrier()
    Ag = agnostic.Agnostic(farmer_gams_agnostic, cfg)

    scenario_creator = Ag.scenario_creator
    scenario_denouement = farmer_gams_agnostic.scenario_denouement   # should we go though Ag?
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
