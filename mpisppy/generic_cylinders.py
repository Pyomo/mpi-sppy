# This software is distributed under the 3-clause BSD License.
# Started by dlw August 2024: General cylinder driver.
# We get the module from the command line.

import sys
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.config as config
import mpisppy.agnostic.agnostic as agnostic
import mpisppy.utils.sputils as sputils


def _parse_args(m):
    # m is the model file module
    cfg = config.Config()
    cfg.add_to_config(name="module_name",
                      description="file name that has scenario creator, etc.",
                      domain=str,
                      default=None,
                      argparse=True)
    assert hasattr(m, "inparser_adder"), "The model file must have an inparser_adder function"
    cfg.add_to_config(name="write_solution",
                      description="verbose output",
                      domain=bool,
                      default=False)


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

    cfg.parse_command_line(f"mpi-sppy for {cfg.module_name}")
    return cfg


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("The python model file module name (no .py) must be given.")
        print("usage, e.g.: python -m mpi4py agnostic_cylinders.py --module-name farmer4agnostic" --help)
        quit()

    model_fname = sys.argv[2]

    module = sputils.module_name_to_module(model_fname)

    cfg = _parse_args(module)

    scenario_creator = module.scenario_creator
    assert hasattr(module, "scenario_denouement"), "The model file must have a scenario_denouement function"
    scenario_denouement = module.scenario_denouement

    all_scenario_names = module.scenario_names_creator(cfg.num_scens)

    xxxxx

    if cfg.write_solution:
        wheel.write_first_stage_solution(f'{model_fname}.csv')
        wheel.write_first_stage_solution(f'{model_fname}.npy',
                first_stage_solution_writer=sputils.first_stage_nonant_npy_serializer)
        wheel.write_tree_solution(f'{model_fname}_solution')
    
