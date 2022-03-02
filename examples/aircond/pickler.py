# This software is distributed under the 3-clause BSD License.
# HACK to create proper bundles for aircond; DLW march 2022
# python pickler.py --branching-factors 2 2 2 --pickle-bundles-dir="." --scenarios-per-bundle=2
import sys
import os
import copy
import numpy as np
import itertools
import mpisppy.tests.examples.aircondB as aircondB
from mpisppy.utils import baseparsers
from mpisppy.utils import pickle_bundle

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
    scenario_names = aircondB.scenario_names_creator(num_scens, start)
    
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
    parser = baseparsers._basic_multistage()
    parser = pickle_bundle.pickle_bundle_parser(parser)
    args = parser.parse_args()

    return args

def main():

    args = _parse_args()
    assert args.pickle_bundles_dir is not None
    assert args.scenarios_per_bundle is not None
    assert args.unpickle_bundles_dir is None

    BFs = args.branching_factors

    if BFs is None:
        raise RuntimeError("Branching factors must be specified")

    ##xhat_scenario_dict = make_node_scenario_dict_balanced(BFs)
    ##all_nodenames = list(xhat_scenario_dict.keys())

    # This is multi-stage, so we need to supply node names
    #all_nodenames = ["ROOT"] # all trees must have this node
    # The rest is a naming convention invented for this problem.
    # Note that mpisppy does not have nodes at the leaves,
    # and node names must end in a serial number.

    ScenCount = np.prod(BFs)
    #ScenCount = _get_num_leaves(BFs)
    sc_options = {"args": args}
    kwargs = aircondB.kw_creator(sc_options)

    bsize = int(args.scenarios_per_bundle)
    numbuns = ScenCount // bsize
    all_bundle_names = [f"Bundle_{bn*bsize}_{(bn+1)*bsize-1}" for bn in range(numbuns)]

    for bname in all_bundle_names:
        aircondB.scenario_creator(bname, **kwargs)
    

if __name__ == "__main__":
    main()
    
