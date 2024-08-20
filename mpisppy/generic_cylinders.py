# This software is distributed under the 3-clause BSD License.
# Started by dlw August 2024: General cylinder driver.
# We get the module from the command line.

import sys
import os
import json
import numpy as np
import pyomo.environ as pyo
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.config as config
import mpisppy.utils.sputils as sputils
from mpisppy.extensions.extension import MultiExtension
from mpisppy.extensions.fixer import Fixer
from mpisppy.extensions.mipgapper import Gapper
import mpisppy.utils.solver_spec as solver_spec
from mpisppy import global_toc

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

    cfg.EF_base()  # If EF is slected, most other options will be moot
    # There are some arguments here that will not make sense for all models
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.aph_args()
    cfg.fixer_args()    
    cfg.gapper_args()    
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.ph_ob_args()
    cfg.xhatshuffle_args()
    cfg.xhatxbar_args()
    cfg.converger_args()
    cfg.wxbar_read_write_args()
    cfg.tracking_args()
    cfg.gradient_args()
    cfg.grad_rho_args()

    cfg.parse_command_line(f"mpi-sppy for {cfg.module_name}")
    return cfg

#==========
def _do_decomp(module, cfg, scenario_creator, scenario_creator_kwargs, scenario_denouement):
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

    # Note: high level code like this assumes there are branching factors
    # for multi-stage problems. For other trees, you will need lower-level code
    if cfg.get("branching_factors") is not None:
        all_nodenames = sputils.create_nodenames_from_branching_factors(\
                                    cfg.branching_factors)
        num_scens = np.prod(cfg.branching_factors)
        assert not cfg.xhatshuffle or cfg.get("stage2EFsolvern") is not None, "For now, stage2EFsolvern is required for multistage xhat"

    else:
        all_nodenames = None
        num_scens = cfg.num_scens
     
    all_scenario_names = module.scenario_names_creator(num_scens)
      
    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    if cfg.run_async:
        # Vanilla APH hub
        hub_dict = vanilla.aph_hub(*beans,
                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                   ph_extensions=None,
                                   rho_setter = rho_setter,
                                   all_nodenames = all_nodenames,
                                   )
    else:
        # Vanilla PH hub
        hub_dict = vanilla.ph_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  ph_extensions=None,
                                  ph_converger=ph_converger,
                                  rho_setter = rho_setter,
                                  all_nodenames = all_nodenames,
                                  )
    
    # Extend and/or correct the vanilla dictionary
    ext_classes = list()
    # TBD: add cross_scenario_cuts, which also needs a cylinder
    if cfg.mipgaps_json is not None:
        ext_classes.append(Gapper)
        with open(cfg.mipgaps_json) as fin:
            din = json.load(fin)
        mipgapdict = {int(i): din[i] for i in din}
        hub_dict["opt_kwargs"]["options"]["gapperoptions"] = {
            "verbose": cfg.verbose,
            "mipgapdict": mipgapdict
        }
        
    if cfg.fixer:  # cfg_vanilla takes care of the fixer_tol?
        ext_classes.append(Fixer)
        hub_dict["opt_kwargs"]["options"]["fixeroptions"] = {
            "verbose": cfg.verbose,
            "boundtol": cfg.fixer_tol,
            "id_fix_list_fct": uc.id_fix_list_fct,
        }
    if len(ext_classes) != 0:
        hub_dict['opt_kwargs']['extensions'] = MultiExtension
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
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs, all_nodenames=all_nodenames)

    # Standard Lagrangian bound spoke
    if cfg.lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                                rho_setter = rho_setter,
                                                all_nodenames = all_nodenames,
                                                )

    # ph outer bounder spoke
    if cfg.ph_ob:
        ph_ob_spoke = vanilla.ph_ob_spoke(*beans,
                                          scenario_creator_kwargs=scenario_creator_kwargs,
                                          rho_setter = rho_setter,
                                          all_nodenames = all_nodenames,
                                          )

    # xhat shuffle bound spoke
    if cfg.xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans,
                                                      scenario_creator_kwargs=scenario_creator_kwargs,
                                                      all_nodenames=all_nodenames)
 
   # special code for multi-stage (e.g., hydro)
    if cfg.get("stage2EFsolvern") is not None:
        print("debug foo")
        assert cfg.get("xhatshuffle"), "xhatshuffle is required for stage2EFsolvern"
        xhatshuffle_spoke["opt_kwargs"]["options"]["stage2EFsolvern"] = cfg["stage2EFsolvern"]
        xhatshuffle_spoke["opt_kwargs"]["options"]["branching_factors"] = cfg["branching_factors"]

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
        wheel.write_tree_solution(f'{cfg.solution_base_name}_soldir')    
        global_toc("Wrote solution data.")


#==========
def _do_EF(module, cfg, scenario_creator, scenario_creator_kwargs, scenario_denouement):
    ef = sputils.create_EF(
        module.scenario_names_creator(cfg.num_scens),
        module.scenario_creator,
        scenario_creator_kwargs=module.kw_creator(cfg),
    )

    sroot, solver_name, solver_options = solver_spec.solver_specification(cfg, "EF")

    solver = pyo.SolverFactory(solver_name)
    if solver_options is not None:
        # We probably could just assign the dictionary in one line...
        for option_key,option_value in solver_options.items():
            solver.options[option_key] = option_value
    if 'persistent' in solver_name:
        solver.set_instance(ef, symbolic_solver_labels=True)
        results = solver.solve(tee=cfg.tee_EF)
    else:
        results = solver.solve(ef, tee=cfg.tee_EF, symbolic_solver_labels=True,)

    global_toc(f"EF objective: {pyo.value(ef.EF_Obj)}")
    if cfg.solution_base_name is not None:
        sputils.ef_nonants_csv(ef, f'{cfg.solution_base_name}.csv')
        sputils.ef_ROOT_nonants_npy_serializer(ef, f'{cfg.solution_base_name}.csv')
        write_ef_tree_solution(ef,f'{cfg.solution_base_name}_soldir')
        global_toc("Wrote EF solution data.")
    

##########################################################################
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

    if cfg.EF:
        _do_EF(module, cfg, scenario_creator, scenario_creator_kwargs, scenario_denouement)
    else:
        _do_decomp(module, cfg, scenario_creator, scenario_creator_kwargs, scenario_denouement)
