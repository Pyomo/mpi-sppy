# This software is distributed under the 3-clause BSD License.
# HACK to create proper bundles for aircond; DLW march 2022
# python pickler.py --branching-factors 2 2 2 --pickle-bundles-dir="." --scenarios-per-bundle=4 --Capacity 200 --QuadShortCoeff 0.3  --BeginInventory 50 --mu-dev 0 --sigma-dev 40 --start-seed 0 
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

def _parse_args():
    parser = baseparsers._basic_multistage()
    parser = pickle_bundle.pickle_bundle_parser(parser)
    parser = aircondB.inparser_adder(parser)
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
    
