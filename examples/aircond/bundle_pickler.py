# This software is distributed under the 3-clause BSD License.
# Program to create proper bundles for aircond; DLW march 2022
"""
python bundle_pickler.py --branching-factors 4 4 4 --pickle-bundles-dir="." --scenarios-per-bundle=16 --Capacity 200 --QuadShortCoeff 0.3  --BeginInventory 50 --mu-dev 0 --sigma-dev 40 --start-seed 0 

# It is entirely up to the user to make sure that the scenario count and scenarios per bundle match between creating the pickles and using them (but the second set of args might not matter)
# branching factors is 64 because there are a total of 64 scenarios when pickled; scenarios per bundle is 16 because that's what is was when pickled (the problem is now two-stage)
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors 8 --Capacity 200 --QuadShortCoeff 0.3  --BeginInventory 50 --rel-gap 0.01 --mu-dev 0 --sigma-dev 40 --max-solver-threads 2 --start-seed 0 --no-fwph --no-lagranger --start-ups --bundles-per-rank=0 --unpickle-bundles-dir="." --scenarios-per-bundle=4
"""
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
    
