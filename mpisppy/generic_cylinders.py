###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Generic_cylinder test driver. Adapted from run_all.py by dlw August 2024.

import sys
import os
import json
import shutil
import numpy as np
import pyomo.environ as pyo
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.config as config
import mpisppy.utils.sputils as sputils
from mpisppy.convergers.norm_rho_converger import NormRhoConverger
from mpisppy.convergers.primal_dual_converger import PrimalDualConverger
from mpisppy.extensions.extension import MultiExtension
from mpisppy.extensions.fixer import Fixer
from mpisppy.extensions.mipgapper import Gapper
from mpisppy.extensions.gradient_extension import Gradient_extension
from mpisppy.extensions.scenario_lpfiles import Scenario_lpfiles
import mpisppy.utils.solver_spec as solver_spec
from mpisppy import global_toc
from mpisppy import MPI


def _parse_args(m):
    # m is the model file module
    cfg = config.Config()
    cfg.proper_bundle_config()
    cfg.pickle_scenarios_config()
    
    cfg.add_to_config(name="module_name",
                      description="Name of the file that has the scenario creator, etc.",
                      domain=str,
                      default=None,
                      argparse=True)
    assert hasattr(m, "inparser_adder"), "The model file must have an inparser_adder function"
    cfg.add_to_config(name="solution_base_name",
                      description="The string used for a directory of ouput along with a csv and an npv file (default None, which means no soltion output)",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="scenario_lpfiles",
                      description="Invokes an extension that writes an model lp file and a nonants json file for each scenario before iteration 0",
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
    cfg.integer_relax_then_enforce_args()
    cfg.gapper_args()    
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.ph_ob_args()
    cfg.subgradient_args()
    cfg.xhatshuffle_args()
    cfg.xhatxbar_args()
    cfg.converger_args()
    cfg.wxbar_read_write_args()
    cfg.tracking_args()
    cfg.gradient_args()
    cfg.dynamic_rho_args()
    cfg.reduced_costs_args()
    cfg.sep_rho_args()
    cfg.coeff_rho_args()
    cfg.sensi_rho_args()
    cfg.reduced_costs_rho_args()
    cfg.parse_command_line(f"mpi-sppy for {cfg.module_name}")
    
    cfg.checker()  # looks for inconsistencies 
    return cfg

def _name_lists(module, cfg, bundle_wrapper=None):

    # Note: high level code like this assumes there are branching factors for
    # multi-stage problems. For other trees, you will need lower-level code
    if cfg.get("branching_factors") is not None:
        all_nodenames = sputils.create_nodenames_from_branching_factors(
                                    cfg.branching_factors)
        num_scens = np.prod(cfg.branching_factors)
        assert not cfg.xhatshuffle or cfg.get("stage2EFsolvern") is not None,\
            "For now, stage2EFsolvern is required for multistage xhat"
    else:
        all_nodenames = None
        num_scens = cfg.num_scens

    # proper bundles should be almost magic
    if cfg.unpickle_bundles_dir or cfg.scenarios_per_bundle is not None:
        num_buns = cfg.num_scens // cfg.scenarios_per_bundle
        all_scenario_names = bundle_wrapper.bundle_names_creator(num_buns, cfg=cfg)
        all_nodenames = None  # This is seldom used; also, proper bundles result in two stages
    else:
        all_scenario_names = module.scenario_names_creator(num_scens)

    return all_scenario_names, all_nodenames


#==========
def _do_decomp(module, cfg, scenario_creator, scenario_creator_kwargs, scenario_denouement, bundle_wrapper=None):
    rho_setter = module._rho_setter if hasattr(module, '_rho_setter') else None
    if cfg.default_rho is None and rho_setter is None:
        if cfg.sep_rho or cfg.coeff_rho or cfg.sensi_rho:
            cfg.default_rho = 1
        else:
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

    all_scenario_names, all_nodenames = _name_lists(module, cfg, bundle_wrapper=bundle_wrapper)    

    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    if cfg.APH:
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
        assert hasattr(module, "id_fix_list_fct"), "id_fix_list_fct required for --fixer"
        ext_classes.append(Fixer)
        hub_dict["opt_kwargs"]["options"]["fixeroptions"] = {
            "verbose": cfg.verbose,
            "boundtol": cfg.fixer_tol,
            "id_fix_list_fct": module.id_fix_list_fct,
        }
    if cfg.rc_fixer:
        vanilla.add_reduced_costs_fixer(hub_dict, cfg)

    if cfg.integer_relax_then_enforce:
        vanilla.add_integer_relax_then_enforce(hub_dict, cfg)

    if cfg.grad_rho:
        ext_classes.append(Gradient_extension)
        hub_dict['opt_kwargs']['options']['gradient_extension_options'] = {'cfg': cfg}        

    if cfg.scenario_lpfiles:
        ext_classes.append(Scenario_lpfiles)

    if cfg.sep_rho:
        vanilla.add_sep_rho(hub_dict, cfg)

    if cfg.coeff_rho:
        vanilla.add_coeff_rho(hub_dict, cfg)

    # these should be after sep rho and coeff rho
    # as they will use existing rho values if the
    # sensitivity is too small
    if cfg.sensi_rho:
        vanilla.add_sensi_rho(hub_dict, cfg)

    if cfg.reduced_costs_rho:
        vanilla.add_reduced_costs_rho(hub_dict, cfg)
 
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
        fw_spoke = vanilla.fwph_spoke(*beans,
                                      scenario_creator_kwargs=scenario_creator_kwargs,
                                      all_nodenames=all_nodenames,
                                      rho_setter=rho_setter,
                                      )
        # Need to fix FWPH to support extensions
        # if cfg.sep_rho:
        #     vanilla.add_sep_rho(fw_spoke, cfg)
        # if cfg.coeff_rho:
        #     vanilla.add_coeff_rho(fw_spoke, cfg)

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
        if cfg.sep_rho:
            vanilla.add_sep_rho(ph_ob_spoke, cfg)
        if cfg.coeff_rho:
            vanilla.add_coeff_rho(ph_ob_spoke, cfg)
        if cfg.sensi_rho:
            vanilla.add_sensi_rho(ph_ob_spoke, cfg)
 

    # subgradient outer bound spoke
    if cfg.subgradient:
        subgradient_spoke = vanilla.subgradient_spoke(*beans,
                                          scenario_creator_kwargs=scenario_creator_kwargs,
                                          rho_setter = rho_setter,
                                          all_nodenames = all_nodenames,
                                          )
        if cfg.sep_rho:
            vanilla.add_sep_rho(subgradient_spoke, cfg)
        if cfg.coeff_rho:
            vanilla.add_coeff_rho(subgradient_spoke, cfg)
        if cfg.sensi_rho:
            vanilla.add_sensi_rho(subgradient_spoke, cfg)

    # xhat shuffle bound spoke
    if cfg.xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans,
                                                      scenario_creator_kwargs=scenario_creator_kwargs,
                                                      all_nodenames=all_nodenames)

    if cfg.xhatxbar:
        xhatxbar_spoke = vanilla.xhatxbar_spoke(*beans,
                                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                                   all_nodenames=all_nodenames)

    # reduced cost fixer options setup
    if cfg.reduced_costs:
        vanilla.add_reduced_costs_fixer(hub_dict, cfg)

    # reduced cost fixer
    if cfg.reduced_costs:
        reduced_costs_spoke = vanilla.reduced_costs_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              all_nodenames=all_nodenames,
                                              rho_setter = None)

                
   # special code for multi-stage (e.g., hydro)
    if cfg.get("stage2EFsolvern") is not None:
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
    if cfg.subgradient:
        list_of_spoke_dict.append(subgradient_spoke)
    if cfg.xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)
    if cfg.xhatxbar:
        list_of_spoke_dict.append(xhatxbar_spoke)
    if cfg.reduced_costs:
        list_of_spoke_dict.append(reduced_costs_spoke)
        

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    if cfg.solution_base_name is not None:
        wheel.write_first_stage_solution(f'{cfg.solution_base_name}.csv')
        wheel.write_first_stage_solution(f'{cfg.solution_base_name}.npy',
                first_stage_solution_writer=sputils.first_stage_nonant_npy_serializer)
        wheel.write_tree_solution(f'{cfg.solution_base_name}_soldir')    
        global_toc("Wrote solution data.")


#==========
def _write_scenarios(module,
                     cfg,
                     scenario_creator,
                     scenario_creator_kwargs,
                     scenario_denouement,
                     comm):
    assert hasattr(cfg, "num_scens")
    ScenCount = cfg.num_scens
    
    n_proc = comm.Get_size()
    my_rank = comm.Get_rank()
    avg = ScenCount / n_proc
    slices = [list(range(int(i * avg), int((i + 1) * avg))) for i in range(n_proc)]

    local_slice = slices[my_rank]
    my_start = local_slice[0]   # zero based
    inum = sputils.extract_num(module.scenario_names_creator(1)[0])
    
    local_scenario_names = module.scenario_names_creator(len(local_slice),
                                                         start=inum + my_start)
    if my_rank == 0:
        if os.path.exists(cfg.pickle_scenarios_dir):
            shutil.rmtree(cfg.pickle_scenarios_dir)
        os.makedirs(cfg.pickle_scenarios_dir)
    comm.Barrier()
    for sname in local_scenario_names:
        scen = scenario_creator(sname, **scenario_creator_kwargs)
        fname = os.path.join(cfg.pickle_scenarios_dir, sname+".pkl")
        pickle_bundle.dill_pickle(scen, fname)
        # scenario_denouement(my_rank, sname, scen)  # see farmer.py
    global_toc(f"Pickled Scenarios written to {cfg.pickle_scenarios_dir}")


#==========
def _read_pickled_scenario(sname, cfg):
    fname = os.path.join(cfg.unpickle_scenarios_dir, sname+".pkl")
    scen = pickle_bundle.dill_unpickle(fname)
    return scen
    
        
#==========
def _write_bundles(module,
                   cfg,
                   scenario_creator,
                   scenario_creator_kwargs,
                   comm):
    assert hasattr(cfg, "num_scens")
    ScenCount = cfg.num_scens
    bsize = int(cfg.scenarios_per_bundle)
    numbuns = ScenCount // bsize
    n_proc = comm.Get_size()
    my_rank = comm.Get_rank()

    if numbuns < n_proc:
        raise RuntimeError(
            "More MPI ranks (%d) supplied than needed given the number of bundles (%d) "
            % (n_proc, numbuns)
        )

    avg = numbuns / n_proc
    slices = [list(range(int(i * avg), int((i + 1) * avg))) for i in range(n_proc)]

    local_slice = slices[my_rank]
    # We need to know if scenarios (not bundles) are one-based.
    inum = sputils.extract_num(module.scenario_names_creator(1)[0])
    
    local_bundle_names = [f"Bundle_{bn*bsize+inum}_{(bn+1)*bsize-1+inum}" for bn in local_slice]

    if my_rank == 0:
        if os.path.exists(cfg.pickle_bundles_dir):
            shutil.rmtree(cfg.pickle_bundles_dir)
        os.makedirs(cfg.pickle_bundles_dir)
    comm.Barrier()
    for bname in local_bundle_names:
        bundle = scenario_creator(bname, **scenario_creator_kwargs)
        fname = os.path.join(cfg.pickle_bundles_dir, bname+".pkl")
        pickle_bundle.dill_pickle(bundle, fname)
    global_toc(f"Bundles written to {cfg.pickle_bundles_dir}")
        
#==========
def _do_EF(module, cfg, scenario_creator, scenario_creator_kwargs, scenario_denouement, bundle_wrapper=None):

    all_scenario_names, _ = _name_lists(module, cfg, bundle_wrapper=bundle_wrapper)
    ef = sputils.create_EF(
        all_scenario_names,
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
    if not pyo.check_optimal_termination(results):
        print("Warning: non-optimal solver termination")

    global_toc(f"EF objective: {pyo.value(ef.EF_Obj)}")
    if cfg.solution_base_name is not None:
        sputils.ef_nonants_csv(ef, f'{cfg.solution_base_name}.csv')
        sputils.ef_ROOT_nonants_npy_serializer(ef, f'{cfg.solution_base_name}.npy')
        sputils.write_ef_tree_solution(ef,f'{cfg.solution_base_name}_soldir')
        global_toc("Wrote EF solution data.")

def _model_fname():
    def _bad_news():
        raise RuntimeError("Unable to parse module name from first argument"
                           " (for module foo.py, we want, e.g.\n"
                           "--module-name foo\n"
                           "or\n"
                           "--module-name=foo")
    def _len_check(needed_length):
        if len(sys.argv) <= needed_length:
            _bad_news()
        else:
            return True

    _len_check(1)
    assert sys.argv[1][:13] == "--module-name", f"The first command argument must start with'--module-name' but you gave {sys.argv[1]}"
    if sys.argv[1] == "--module-name":
        _len_check(2)
        return sys.argv[2]
    else:
        parts = sys.argv[1].split("=")
        if len(parts) != 2:
            _bad_news()
        return parts[1]
    

def _proper_bundles(cfg):
    return cfg.get("pickle_bundles_dir", ifmissing=False)\
        or cfg.get("unpickle_bundles_dir", ifmissing=False)\
        or cfg.get("scenarios_per_bundle", ifmissing=False)

##########################################################################
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("The python model file module name (no .py) must be given.")
        print("usage, e.g.: python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --help")
        quit()
    model_fname = _model_fname()

    # TBD: when agnostic is merged, use the function and delete the code lines
    # module = sputils.module_name_to_module(model_fname)
    # TBD: do the sys.path.append trick in sputils 
    import importlib
    import inspect
    if inspect.ismodule(model_fname):
        module = model_fname
    else:
        dpath = os.path.dirname(model_fname)
        fname = os.path.basename(model_fname)
        sys.path.append(dpath)
        module = importlib.import_module(fname)
    
    cfg = _parse_args(module)

    bundle_wrapper = None  # the default
    if _proper_bundles(cfg):
        # TBD: remove the need for dill if you are not reading or writing
        import mpisppy.utils.pickle_bundle as pickle_bundle
        import mpisppy.utils.proper_bundler as proper_bundler
    
        bundle_wrapper = proper_bundler.ProperBundler(module)
        scenario_creator = bundle_wrapper.scenario_creator
        # The scenario creator is wrapped, so these kw_args will not go the original
        # creator (the kw_creator will keep the original args)
        scenario_creator_kwargs = bundle_wrapper.kw_creator(cfg)
    elif cfg.unpickle_scenarios_dir is not None:
        # So reading pickled scenarios cannot be composed with proper bundles
        import mpisppy.utils.pickle_bundle as pickle_bundle
        scenario_creator = _read_pickled_scenario
        scenario_creator_kwargs = {"cfg": cfg}
    else:  # the most common case
        scenario_creator = module.scenario_creator
        scenario_creator_kwargs = module.kw_creator(cfg)
        
    assert hasattr(module, "scenario_denouement"), "The model file must have a scenario_denouement function"
    scenario_denouement = module.scenario_denouement

    if cfg.pickle_bundles_dir is not None:
        global_comm = MPI.COMM_WORLD
        _write_bundles(module,
                       cfg,
                       scenario_creator,
                       scenario_creator_kwargs,
                       global_comm)
    elif cfg.pickle_scenarios_dir is not None:
        import mpisppy.utils.pickle_bundle as pickle_bundle
        global_comm = MPI.COMM_WORLD
        _write_scenarios(module,
                         cfg,
                         scenario_creator,
                         scenario_creator_kwargs,
                         scenario_denouement,
                         global_comm)
    elif cfg.EF:
        _do_EF(module, cfg, scenario_creator, scenario_creator_kwargs, scenario_denouement, bundle_wrapper=bundle_wrapper)
    else:
        _do_decomp(module, cfg, scenario_creator, scenario_creator_kwargs, scenario_denouement, bundle_wrapper=bundle_wrapper)
