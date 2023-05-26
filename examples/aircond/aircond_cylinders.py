# This software is distributed under the 3-clause BSD License.
# NOTE: as of March 2022, consider aircondB.py as an alternative to aircond.py
# Use bundle_pickler.py to create bundle pickles
# NOTE: As of 3 March 2022, you can't compare pickle bundle problems with non-pickled. See _demands_creator in aircondB.py for more discusion.

import sys
import os
import copy
import numpy as np
import itertools
from mpisppy import global_toc
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils.sputils import first_stage_nonant_npy_serializer, option_string_to_dict
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.tests.examples.aircond as aircond
import mpisppy.tests.examples.aircondB as aircondB  # pickle bundle version
from mpisppy.utils import pickle_bundle
from mpisppy.utils import amalgamator

# construct a node-scenario dictionary a priori for xhatspecific_spoke,
# according to naming convention for this problem

def make_node_scenario_dict_balanced(BFs,leaf_nodes=True,start=1):
    """ creates a node-scenario dictionary for aircond problem
    Args: 
        BFs (list of ints): branching factors for each stage 
        leaf_nodes (bool): include/exclude leaf nodes (default True)
        start (int): starting value for scenario number (default 1)

    Returns:
        node_scenario_dict (dict): dictionary of node-scenario key-value pairs

    """
    nodenames = make_nodenames_balanced(BFs, leaf_nodes)
    num_scens = np.prod(BFs)
    scenario_names = aircond.scenario_names_creator(num_scens, start)
    
    num_stages = len(BFs)+1
    node_scenario_dict ={"ROOT": scenario_names[0]}
    
    for i in range(BFs[0]):
        node_scenario_dict[nodenames[i]] = scenario_names[i]
        if num_stages >= 3:
            for stage in range(2,num_stages+leaf_nodes):
                node_scenario_dict = _recursive_node_dict(node_scenario_dict,
                                                          nodenames,
                                                          scenario_names,
                                                          BFs,
                                                          stage)

    return node_scenario_dict

def _recursive_node_dict(current_dict, nodenames, scenario_names, BFs, stage):
    """ recursive call for make_node_scenario_dict for 3+ stage problems
    Args: 
        current_dict (dict): current dictionary for scenarios
        nodenames (list): distinct non-leaf nodes (see make_nodenames_balanced)
        scenario_names (list): scenario names low to high
        BFs(list): branching factors for each stage 
        stage (int): current problem stage

    Returns:
        current_dict.copy() (dict): updated dictionary for stages 1-stage

    """
    # How far along in the node_idx are we?
    node_offset = int(np.sum([np.prod(BFs[:i]) for i in range(1,stage-1)]))
    #
    scen_offset = 1
    
    # Check if we're handling leaf nodes:
    if stage == len(BFs)+1:
        scen_offset = 0
        
    for i in range(np.prod(BFs[:(stage-1)])):
        node_idx = i + node_offset
        scen_idx = 0 if i == 0 else int(np.prod(BFs[(stage-1):]))*i - scen_offset

        current_dict[nodenames[node_idx]] = scenario_names[scen_idx]
        
    return current_dict.copy()


def make_nodenames_balanced(BFs, leaf_nodes=False, root = True):
    """ creates a list of node names as in aircond example
    Args:
        BFs (list): branching factor for each stage
        leaf_nodes (bool): if False, exclude leaf node names
        root (bool): if False, no "ROOT_" string on non-root nodes
    
    Returns: 
        nodenames (list): list of all nodenames, "ROOT" is last, ordered as:
        0, 1, 2, ..., BFs[0], 0_0, 0_1, ..., 0_BFs[1], 1_0, 1_1, ... ,
        1_BFs[1], ... , BFs[0]_BFs[1], ... , BFs[0]...BFs[-2]
    """
    if leaf_nodes == False:
        BFs = BFs[:-1] # exclude leaf nodes

    # Constructs all nodenames
    # 0, 1, 2, ..., BFs[0], 00, 01, ..., 0BFs[1], 10, 11, ... ,
    # 1BFs[1], ... , BFs[0]BFs[1], ... , BFs[0]\cdotsBFs[-2]

    max_str_len = len(BFs)
    lists = [[str(k) for k in range(BFs[0])]]
    for i in range(max_str_len-1):
        the_list = []
        for j in range(len(lists[i])):
            the_list.append([lists[i][j]+"_"+str(a) for a in range(BFs[i+1])])
        lists.append(list(itertools.chain(*the_list)))

    strings = list(itertools.chain(*lists))

    nodenames = []

    if root:
        for string in strings:
            rt_string = "ROOT_" + string
            nodenames.append(rt_string)
    else: 
        nodenames = strings
    # add root node (it is nicer to keep it at the end)
    nodenames.append("ROOT")
        
    return nodenames
    
def _parse_args():
    # create and return a Config object with values from the command line
    cfg = config.Config()
    
    cfg.multistage()
    cfg.ph_args()
    cfg.two_sided_args()
    cfg.xhatlooper_args()
    cfg.xhatshuffle_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.lagranger_args()
    cfg.xhatspecific_args()
    cfg.mip_options()
    cfg.aph_args()    
    aircond.inparser_adder(cfg)  # not aircondB
    cfg.add_to_config("run_async",
                         description="Run with async projective hedging instead of progressive hedging",
                         domain=bool,
                         default=False)    
    cfg.add_to_config("write_solution",
                         description="Write various solution files (default False)",
                         domain=bool,
                         default=False)    
    # special "proper" bundle arguments
    parser = pickle_bundle.pickle_bundle_parser(cfg)

    cfg.add_to_config("EF_directly",
                         description="Solve the EF directly instead of using cylinders (default False)",
                         domain=bool,
                         default=False)
    # DLW March 2022: this should be in baseparsers and vanilla some day
    cfg.add_to_config("solver_options",
                         description="Space separarated string with = for arguments (default None)",
                         domain=str,
                         default=None)

    cfg.parse_command_line("aircond_cylinders")
    return cfg

def main():
    cfg = _parse_args()

    cmdargs = _parse_args()

    BFs = cfg.branching_factors

    if BFs is None:
        raise RuntimeError("Branching factors must be specified")
    proper_bundles = pickle_bundle.have_proper_bundles(cmdargs)
    if proper_bundles:
        pickle_bundle.check_args(cmdargs)
        assert cfg.scenarios_per_bundle is not None
        if cfg.pickle_bundles_dir is not None:
            raise ValueError("Do not give a --pickle-bundles-dir to this program. "
                             "This program only takes --unpickle-bundles-dir. "
                             "Use bundle_pickler.py to create bundles.")

    with_xhatspecific = cfg.xhatspecific
    with_lagrangian = cfg.lagrangian
    with_lagranger = cfg.lagranger
    with_fwph = cfg.fwph
    with_xhatshuffle = cfg.xhatshuffle

    # This is multi-stage, so we need to supply node names
    #all_nodenames = ["ROOT"] # all trees must have this node
    # The rest is a naming convention invented for this problem.
    # Note that mpisppy does not have nodes at the leaves,
    # and node names must end in a serial number.

    if proper_bundles:
        # All the scenarios will happen to be bundles, but mpisppy does not need to know that.
        # This code needs to know the correct scenarios per bundle and the number of
        # bundles in --branching-factors (one element in the list).
        bsize = int(cfg.scenarios_per_bundle)
        assert len(BFs) == 1, "for proper bundles, --branching-factors should be the number of bundles"
        numbuns = BFs[0]
        all_scenario_names = [f"Bundle_{bn*bsize}_{(bn+1)*bsize-1}" for bn in range(numbuns)]
        refmodule = aircondB
        primal_rho_setter = None
        dual_rho_setter = None
        global_toc("WARNING: not using rho setters with proper bundles")
        
    else:
        ScenCount = np.prod(BFs)
        all_scenario_names = [f"scen{i}" for i in range(ScenCount)] #Scens are 0-based
        refmodule = aircond
        primal_rho_setter = refmodule.primal_rho_setter
        dual_rho_setter = refmodule.dual_rho_setter

    xhat_scenario_dict = make_node_scenario_dict_balanced(BFs)
    all_nodenames = list(xhat_scenario_dict.keys())

    scenario_creator_kwargs = refmodule.kw_creator(cfg)  # take everything from config
    scenario_creator = refmodule.scenario_creator
    # TBD: consider using aircond instead of refmodule and pare down aircondB.py
    scenario_denouement = refmodule.scenario_denouement
    
    if cfg.EF_directly:
        """
        ama_options = {"EF-mstage": not proper_bundles,
                       "EF-2stage": proper_bundles,
                       "EF_solver_name": cfg.solver_name,
                       "branching_factors": cfg.branching_factors,
                       "num_scens": ScenCount,  # is this needed?
                       "_mpisppy_probability": 1/ScenCount,  # is this needed?
                       "tee_ef_solves":False,
                       }
        """
        ama = amalgamator.from_module(refmodule,
                                      cfg, use_command_line=False)
        ama.run()
        print(f"EF inner bound=", ama.best_inner_bound)
        print(f"EF outer bound=", ama.best_outer_bound)
        quit()

    # if we are still here, we are running cylinders
    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)
    
    if cfg.run_async:
        # Vanilla APH hub
        hub_dict = vanilla.aph_hub(*beans,
                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                   ph_extensions=None,
                                   rho_setter = None,
                                   all_nodenames=all_nodenames)
    else:
        # Vanilla PH hub
        hub_dict = vanilla.ph_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  ph_extensions=None,
                                  rho_setter = primal_rho_setter,
                                  all_nodenames=all_nodenames)

    # Standard Lagrangian bound spoke
    if with_lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                                    scenario_creator_kwargs=scenario_creator_kwargs,
                                                    rho_setter = primal_rho_setter,
                                                    all_nodenames = all_nodenames)

    # Indepdent Lagranger bound spoke
    if with_lagranger:
        lagranger_spoke = vanilla.lagranger_spoke(*beans,
                                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                                  rho_setter = primal_rho_setter,
                                                  all_nodenames = all_nodenames)
    # Indepdent FWPH bound spoke
    if with_fwph:
        fwph_spoke = vanilla.fwph_spoke(*beans,
                                        scenario_creator_kwargs=scenario_creator_kwargs,
                                        all_nodenames = all_nodenames)

    # xhat specific bound spoke
    if with_xhatspecific:
        xhatspecific_spoke = vanilla.xhatspecific_spoke(*beans,
                                                        xhat_scenario_dict,
                                                        all_nodenames=all_nodenames,
                                                        scenario_creator_kwargs=scenario_creator_kwargs)
    
    #xhat shuffle looper bound spoke
    
    if with_xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, 
                                                      all_nodenames=all_nodenames,
                                                      scenario_creator_kwargs=scenario_creator_kwargs)
    list_of_spoke_dict = list()
    if with_lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if with_lagranger:
        list_of_spoke_dict.append(lagranger_spoke)
    if with_fwph:
        list_of_spoke_dict.append(fwph_spoke)
    if with_xhatspecific:
        list_of_spoke_dict.append(xhatspecific_spoke)
    if with_xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    # DLW (March 2022): This should be generalized
    if cfg.solver_options is not None:
        soptions = option_string_to_dict(cfg.solver_options)
        hub_dict["opt_kwargs"]["options"]["iter0_solver_options"].update(soptions)
        hub_dict["opt_kwargs"]["options"]["iterk_solver_options"].update(soptions)
        for sd in list_of_spoke_dict:
            sd["opt_kwargs"]["options"]["iter0_solver_options"].update(soptions)
            sd["opt_kwargs"]["options"]["iterk_solver_options"].update(soptions)

        if with_xhatspecific:
            xhatspecific_spoke["opt_kwargs"]["options"]["xhat_looper_options"]["xhat_solver_options"].update(soptions)
        if with_xhatshuffle:
            xhatshuffle_spoke["opt_kwargs"]["options"]["xhat_looper_options"]["xhat_solver_options"].update(soptions)
    # special code to get a trace for xhatshuffle
    if with_xhatshuffle and cfg.trace_prefix is not None:
        xhatshuffle_spoke["opt_kwargs"]["options"]['shuffle_running_trace_prefix']  = cfg.trace_prefix

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    fname = 'aircond_cyl_nonants.npy'
    if wheel.global_rank == 0:
        print("BestInnerBound={} and BestOuterBound={}".\
              format(wheel.BestInnerBound, wheel.BestOuterBound))
        if cfg.write_solution:
            print(f"Writing first stage solution to {fname}")
    # all ranks need to participate because only the winner will write
    if cfg.write_solution:
        wheel.write_first_stage_solution('aircond_first_stage.csv')
        wheel.write_tree_solution('aircond_full_solution')
        wheel.write_first_stage_solution(fname,
                    first_stage_solution_writer=first_stage_nonant_npy_serializer)
        global_toc("Solutions written.")

if __name__ == "__main__":
    main()
    
