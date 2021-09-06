import sys
import os
import copy
import mpisppy.tests.examples.aircond_submodels as aircond_submodels
import numpy as np
import itertools
import mpisppy.utils.sputils as sputils
from mpisppy.utils import baseparsers
from mpisppy.utils import vanilla

write_solution = True

# construct a node-scenario dictionary a priori for xhatspecific_spoke,
# according to naming convention for this problem

def make_node_scenario_dict_balanced(BFs,leaf_nodes=True,start=1):
    """ creates a node-scenario dictionary for aircond problem
    Args: 
        BFs(list): branching factors for each stage 
        leaf_nodes (bool): include/disclue leaf nodes (default True)
        start (int): starting value for scenario number (default 1)

    Returns:
        node_scenario_dict (dict): dictionary of node-scenario key-value pairs

    """
    nodenames = make_nodenames_balanced(BFs, leaf_nodes)
    num_scens = np.prod(BFs)
    scenario_names = aircond_submodels.scenario_names_creator(num_scens, start)
    
    stages = len(BFs)
    node_scenario_dict ={"ROOT": scenario_names[0]}
    
    for i in range(BFs[0]):
        if i == 0:
            node_scenario_dict[nodenames[i]] = scenario_names[i]
        else: # 2
            if stages >= 2:
                for stage in range(2,stages+1+leaf_nodes):
                    node_scenario_dict = _recursive_node_dict(node_scenario_dict,
                     nodenames, scenario_names, BFs, stage)


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
        leaf_nodes (bool): if False, disclude leaf node names
        root (bool): if False, no "ROOT_" string on non-root nodes
    
    Returns: 
        nodenames (list): list of all nodenames, "ROOT" is last, ordered as:
        0, 1, 2, ..., BFs[0], 0_0, 0_1, ..., 0_BFs[1], 1_0, 1_1, ... ,
        1_BFs[1], ... , BFs[0]_BFs[1], ... , BFs[0]...BFs[-2]
    """
    if leaf_nodes == False:
        BFs = BFs[:-1] # disclude leaf nodes

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
    parser = baseparsers.make_multistage_parser()
    parser = baseparsers.two_sided_args(parser)
    parser = baseparsers.xhatlooper_args(parser)
    parser = baseparsers.xhatshuffle_args(parser)
    parser = baseparsers.fwph_args(parser)
    parser = baseparsers.lagrangian_args(parser)
    parser = baseparsers.xhatspecific_args(parser)
    parser = baseparsers.mip_options(parser)    
    args = parser.parse_args()
    return args

def main():

    args = _parse_args()

    BFs = args.branching_factors

    xhat_scenario_dict = make_node_scenario_dict_balanced(BFs)
    all_nodenames = list(xhat_scenario_dict.keys())

    with_xhatspecific = args.with_xhatspecific
    with_lagrangian = args.with_lagrangian
    with_xhatshuffle = args.with_xhatshuffle

    # This is multi-stage, so we need to supply node names
    #all_nodenames = ["ROOT"] # all trees must have this node
    # The rest is a naming convention invented for this problem.
    # Note that mpisppy does not have nodes at the leaves,
    # and node names must end in a serial number.

    ScenCount = np.prod(BFs)
    #ScenCount = _get_num_leaves(BFs)
    scenario_creator_kwargs = {"BFs": BFs}
    all_scenario_names = [f"scen{i}" for i in range(ScenCount)] #Scens are 0-based
    # print(all_scenario_names)
    scenario_creator = aircond_submodels.scenario_creator
    scenario_denouement = aircond_submodels.scenario_denouement
    rho_setter = None
    
    # Things needed for vanilla cylinders
    beans = (args, scenario_creator, scenario_denouement, all_scenario_names)
    
    # Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              scenario_creator_kwargs=scenario_creator_kwargs,
                              ph_extensions=None,
                              rho_setter = rho_setter,
                              all_nodenames=all_nodenames)

    # Standard Lagrangian bound spoke
    if with_lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = rho_setter,
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
    if with_xhatspecific:
        list_of_spoke_dict.append(xhatspecific_spoke)
    if with_xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    spcomm, opt_dict = sputils.spin_the_wheel(hub_dict, list_of_spoke_dict)

    if "hub_class" in opt_dict:  # we are a hub rank
        if spcomm.opt.cylinder_rank == 0:  # we are the reporting hub rank
            print("BestInnerBound={} and BestOuterBound={}".\
                  format(spcomm.BestInnerBound, spcomm.BestOuterBound))
    
    if write_solution:
        sputils.write_spin_the_wheel_first_stage_solution(spcomm, opt_dict, 'aircond_first_stage.csv')
        sputils.write_spin_the_wheel_tree_solution(spcomm, opt_dict, 'aircond_full_solution')

if __name__ == "__main__":
    main()
    