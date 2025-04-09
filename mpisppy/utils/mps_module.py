##############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2025, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# WARNING: the scenario_creator is very dependent on how the mps_reader works.
# This is designed for use with generic_cylinders.py
"""You pass in a path to a special mps_files_directory in the config and this
module will become a valid mpi-sppy module.
The directory has to have file pairs for each scenario where one file
is the mps file and other is a json file with a scenario tree dictionary
for the scenario.

Scenario names must have a consistent base (e.g. "scenario" or "scen") and
must end in a serial number (unless you can't, you should start with 0; 
otherwise, start with 1).

Parenthesis in variable names in the json file must become underscores

note to dlw from dlw:
  You could offer the option to split up the objective by stages in the lp file
  You should also offer other types of nonants in the json

"""
import os
import re
import glob
import json
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import mpisppy.utils.mps_reader as mps_reader
# assume you can get the path from config, set in kw_creator as a side-effect
mps_files_directory = None

def scenario_creator(sname, cfg=None):
    """ Load the model from an mps file

    Args:
        scenario_name (str):
            Name of the scenario to construct.
        cfg (Pyomo config object): options
    """

    sharedPath = os.path.join(cfg.mps_files_directory, sname)
    mpsPath = sharedPath + ".mps"
    model = mps_reader.read_mps_and_create_pyomo_model(mpsPath)

    # now read the JSON file and attach the tree information.
    jsonPath = sharedPath + "_nonants.json"
    with open(jsonPath) as f:
        nonantDict = json.load(f)
    try:
        scenProb = nonantDict["scenarioData"]["scenProb"]
    except Exception as e:
        raise RuntimeError(f'Error getting scenProb from {jsonPath}: {e}')
    assert "ROOT" in nonantDict, f'"ROOT" must be top node in {jsonPath}'
    treeNodes = list()
    parent_ndn = None   # counting on the json file to have ordered nodes
    stage = 1
    treeDict = nonantDict["treeData"]
    for ndn in treeDict:
        cp = treeDict[ndn]["condProb"]
        nonants = [model.\
                   find_component(var_name.replace('(','_').replace(')','_'))
                   for var_name in treeDict[ndn]["nonAnts"]]
        assert parent_ndn == sputils.parent_ndn(ndn),\
            f"bad node names or parent order in {jsonPath} detected at {ndn}"
        treeNodes.append(scenario_tree.\
                         ScenarioNode(name=ndn,
                                      cond_prob=cp,
                                      stage=stage,
                                      cost_expression=0.0,
                                      nonant_list=nonants,
                                      scen_model=model,
                                      nonant_ef_suppl_list = None,
                                      parent_name = parent_ndn
                                      )
                         )
        parent_ndn = ndn
        stage += 1
        
    model._mpisppy_probability = scenProb
    model._mpisppy_node_list = treeNodes
    return model


#=========
def scenario_names_creator(num_scens, start=None):
    # validate the directory and use it to get names (that have to be numbered)
    # IMPORTANT: start is zero-based even if the names are one-based!
    mps_files = [os.path.basename(f)
                for f in glob.glob(os.path.join(mps_files_directory, "*.mps"))]
    mps_files.sort()
    if start is None:
        start = 0
    if num_scens is None:
        num_scens = len(mps_files) - start
    first = re.search(r"\d+$",mps_files[0][:-4])  # first scenario number
    try:
        first = int(first.group())
    except Exception as e:
        raise RuntimeError(f'mps files in {mps_files_directory} must end with an integer'
                           f'found file {mps_files[0]} (error was: {e})')
    
    print("WARNING: one-based senario names might cause trouble"
          f" found {first} for dir {mps_files_directory}")
    assert start+num_scens <= len(mps_files),\
        f"Trying to create scenarios names with {start=}, {num_scens=} but {len(mps_files)=}"
    retval = [fn[:-4] for fn in mps_files[start:start+num_scens]]
    return retval

#=========
def inparser_adder(cfg):
    # verify that that the mps_files_directory is there, or add it
    if "mps_files_directory" not in cfg:
        cfg.add_to_config("mps_files_directory",
                          "Directory with mps, json pairs for scenarios",
                          domain=str,
                          default=None,
                          argparse=True)

#=========
def kw_creator(cfg):
    # creates keywords for scenario creator
    # SIDE EFFECT: A bit of hack to get the directory path
    global mps_files_directory
    mps_files_directory = cfg.mps_files_directory
    return {"cfg": cfg}


# This is only needed for sampling
def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    # assert two stage, then do the usual for two stages?
    pass

    
#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass

