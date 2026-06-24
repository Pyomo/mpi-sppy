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
is the model file (.lp or .mps) and the other is a json file with a scenario
tree dictionary for the scenario. An optional {scenario}_rho.csv file supplies
per-nonant rho values.

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
import mpisppy.problem_io.mps_reader as mps_reader
import mpisppy.utils.rho_utils as rho_utils
# assume you can get the path from config, set in kw_creator as a side-effect
mps_files_directory = None

# Scenario model files may be MPS or LP; .lp is preferred when both are present.
_SCENARIO_EXTS = (".lp", ".mps")


def _scenario_model_path(directory, sname):
    """ Return the path to the scenario model file, preferring .lp over .mps. """
    for ext in _SCENARIO_EXTS:
        p = os.path.join(directory, sname + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No scenario model file for '{sname}' in {directory} "
        f"(looked for {', '.join(sname + e for e in _SCENARIO_EXTS)})")

def scenario_creator(sname, cfg=None):
    """ Load the model from an mps file

    Args:
        scenario_name (str):
            Name of the scenario to construct.
        cfg (Pyomo config object): options
    """

    sharedPath = os.path.join(cfg.mps_files_directory, sname)
    modelPath = _scenario_model_path(cfg.mps_files_directory, sname)
    model = mps_reader.read_mps_and_create_pyomo_model(modelPath)

    # now read the JSON file and attach the tree information.
    jsonPath = sharedPath + "_nonants.json"
    with open(jsonPath) as f:
        jsonDict = json.load(f)
    try:
        scenProb = jsonDict["scenarioData"]["scenProb"]
    except Exception as e:
        raise RuntimeError(f'Error getting scenProb from {jsonPath}: {e}')
    treeNodes = list()
    parent_ndn = None   # counting on the json file to have ordered nodes
    stage = 1
    treeDict = jsonDict["treeData"]["nodes"]
    assert "ROOT" in treeDict, f'"ROOT" must be top node in {jsonPath}'
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

    # Optional per-nonant rho file; picked up by _rho_setter (None if absent).
    rho_path = sharedPath + "_rho.csv"
    model._mps_rho_csv = rho_path if os.path.exists(rho_path) else None

    return model


#=========
def scenario_names_creator(num_scens, start=None):
    # validate the directory and use it to get names (that have to be numbered)
    # IMPORTANT: start is zero-based even if the names are one-based!
    model_files = []
    for ext in _SCENARIO_EXTS:
        model_files += glob.glob(os.path.join(mps_files_directory, "*" + ext))
    # dedup base names so a dir with both {s}.lp and {s}.mps yields one name
    model_names = sorted({os.path.splitext(os.path.basename(f))[0]
                          for f in model_files})
    if start is None:
        start = 0
    if num_scens is None:
        num_scens = len(model_names) - start
    first = re.search(r"\d+$", model_names[0])  # first scenario number
    try:
        first = int(first.group())
    except Exception as e:
        raise RuntimeError(f'scenario model files in {mps_files_directory} must end'
                           f' with an integer found file {model_names[0]}'
                           f' (error was: {e})')
    if first != 0:
        print("WARNING: non-zero-based senario names might cause trouble"
              f" found {first=} for dir {mps_files_directory}")
    assert start+num_scens <= len(model_names),\
        f"Trying to create scenarios names with {start=}, {num_scens=} but {len(model_names)=}"
    retval = model_names[start:start+num_scens]
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


#=========
def _rho_setter(scenario):
    """ Per-nonant rho from the scenario's {s}_rho.csv, if present.

    Auto-discovered by the generic driver (generic/decomp.py:_get_rho_setter)
    and threaded to the hub/spokes. Returns an empty list when no rho file is
    present, so the --default-rho path is preserved. When a rho file is present
    it should cover every nonant (with --default-rho passed as a backstop),
    because defining _rho_setter bypasses the driver's --default-rho check.
    """
    path = getattr(scenario, "_mps_rho_csv", None)
    if not path:
        return []
    return rho_utils.rho_list_from_csv(scenario, path)


# This is only needed for sampling
def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    # assert two stage, then do the usual for two stages?
    pass

    
#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass

