# This software is distributed under the 3-clause BSD License.
# Started by dlw April 2024: General agnostic cylinder driver.
"""
We need to get the module from the command line, then construct
the X_guest (e.g. Pyomo_guest) class to wrap the module
and feed it to the Agnostic object.

I think it can be a fully general cylinders program?
"""
import sys
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.config as config
import mpisppy.utils.agnostic as agnostic
import mpisppy.utils.sputils as sputils


def _parse_args(m):
    # m is the model file module
    cfg = config.Config()

    assert hasattr(m, "inparser_adder"), "The model file must have an inparser_adder function"
    m.inparser_adder(cfg)

    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()    
    cfg.aph_args()    
    cfg.xhatlooper_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.lagranger_args()
    cfg.xhatshuffle_args()

    cfg.parse_command_line("agnostic cylinders")
    return cfg


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("need the python model file name")
        print("usage, e.g.: python -m mpi4py agnostic_cylinders.py examples/farmer4agnostic.py" --help)
        quit()

    model_fname = sys.argv[1]

    module = sputils.module_name_to_module(model_fname)

    cfg = _parse_args(module)

    # xxxx now I need the pyomo_guest wrapper, then feed that to agnostic
    xxx
    Ag = agnostic.Agnostic(module, cfg)

    scenario_creator = Ag.scenario_creator
    assert hasattr(m, "scenario_denouement"), "The model file must have a scenario_denouement function"
    scenario_denouement = module.scenario_denouement   # should we go though Ag?
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
    
