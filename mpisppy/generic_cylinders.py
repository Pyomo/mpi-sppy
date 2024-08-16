# This software is distributed under the 3-clause BSD License.
# Started by dlw August 2024: General cylinder driver.
# We get the module from the command line.

import sys, os
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.config as config
import mpisppy.utils.sputils as sputils
from mpisppy.extensions.extension import MultiExtension
from mpisppy.extensions.fixer import Fixer

def _parse_args(m):
    # m is the model file module
    cfg = config.Config()
    cfg.add_to_config(name="module_name",
                      description="Name of the file that has the scenario creator, etc.",
                      domain=str,
                      default=None,
                      argparse=True)
    assert hasattr(m, "inparser_adder"), "The model file must have an inparser_adder function"
    cfg.add_to_config(name="solution_base_name",
                      description="The string used fo a directory of ouput along with a csv and an npv file (default None, which means no soltion output)",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="run_async",
                      description="Use APH instead of PH (default False)",
                      domain=bool,
                      default=False)
    

    m.inparser_adder(cfg)
    # many models, e.g., farmer, need num_scens_required
    #  in which case, it should go in the inparser_adder function
    # cfg.num_scens_required()

    # There are some arguments here that will not make sense for all models
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.aph_args()
    cfg.fixer_args()    
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.ph_ob_args()
    cfg.xhatshuffle_args()
    cfg.converger_args()
    cfg.wxbar_read_write_args()
    cfg.tracking_args()
    cfg.gradient_args()
    cfg.grad_rho_args()

    cfg.parse_command_line(f"mpi-sppy for {cfg.module_name}")
    return cfg


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("The python model file module name (no .py) must be given.")
        print("usage, e.g.: python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --help")
        quit()

    model_fname = sys.argv[2]

    # TBD: when agnostic is merged, use the function and delete the code lines
    # module = sputils.module_name_to_module(model_fname)
    # TBD: do the sys.path.append trick in sputils 
    import importlib, inspect
    if inspect.ismodule(model_fname):
        module = model_fname
    else:
        dpath = os.path.dirname(model_fname)
        fname = os.path.basename(model_fname)
        sys.path.append(dpath)
        module = importlib.import_module(fname)
    
    cfg = _parse_args(module)

    scenario_creator = module.scenario_creator
    scenario_creator_kwargs = module.kw_creator(cfg)    
    assert hasattr(module, "scenario_denouement"), "The model file must have a scenario_denouement function"
    scenario_denouement = module.scenario_denouement

    all_scenario_names = module.scenario_names_creator(cfg.num_scens)

    rho_setter = module._rho_setter if hasattr(module, '_rho_setter') else None
    if cfg.default_rho is None and rho_setter is None:
        raise RuntimeError("No rho_setter so a default must be specified via --default-rho")

    if False:   # maybe later... cfg.use_norm_rho_converger:
        if not cfg.use_norm_rho_updater:
            raise RuntimeError("--use-norm-rho-converger requires --use-norm-rho-updater")
        else:
            ph_converger = NormRhoConverger
    elif False:   # maybe later... cfg.primal_dual_converger:
        ph_converger = PrimalDualConverger
    else:
        ph_converger = None

    fwph = cfg.fwph
    fixer = cfg.fixer

    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)
        
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
    
    # Extend and/or correct the vanilla dictionary
    ext_classes = list()
    # TBD: add Gapper and get the mipgapdict from a json
    # TBD: add cross_scenario_cuts, which also needs a cylinder
    if fixer:  # vanilla takes care of the fixer_tol
        hub_dict['opt_kwargs']['extensions'] = MultiExtension  # TBD: move this  
        ext_classes.append(Fixer)
        hub_dict["opt_kwargs"]["options"]["fixeroptions"] = {
            "verbose": cfg.verbose,
            "boundtol": fixer_tol,
            "id_fix_list_fct": uc.id_fix_list_fct,
        }

    if hub_dict['opt_kwargs']['extensions'] == MultiExtension:
        hub_dict["opt_kwargs"]["extension_kwargs"] = {"ext_classes" : ext_classes}
    if cfg.primal_dual_converger:
        hub_dict['opt_kwargs']['options']\
            ['primal_dual_converger_options'] = {
                'verbose': True,
                'tol': cfg.primal_dual_converger_tol,
                'tracking': True}

    ## norm rho adaptive rho (not the gradient version)
    #if cfg.use_norm_rho_updater:
    #    extension_adder(hub_dict, NormRhoUpdater)
    #    hub_dict['opt_kwargs']['options']['norm_rho_options'] = {'verbose': True}

    # FWPH spoke
    if cfg.fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # Standard Lagrangian bound spoke
    if cfg.lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = rho_setter)

    # ph outer bounder spoke
    if cfg.ph_ob:
        ph_ob_spoke = vanilla.ph_ob_spoke(*beans,
                                          scenario_creator_kwargs=scenario_creator_kwargs,
                                          rho_setter = rho_setter)

    # xhat shuffle bound spoke
    if cfg.xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    list_of_spoke_dict = list()
    if cfg.fwph:
        list_of_spoke_dict.append(fw_spoke)
    if cfg.lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if cfg.ph_ob:
        list_of_spoke_dict.append(ph_ob_spoke)
    if cfg.xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    if cfg.solution_base_name is not None:
        wheel.write_first_stage_solution(f'{cfg.solution_base_name}.csv')
        wheel.write_first_stage_solution(f'{cfg.solution_base_name}.npy',
                first_stage_solution_writer=sputils.first_stage_nonant_npy_serializer)
        wheel.write_tree_solution(f'{cfg.solution_base_name}')    
    
