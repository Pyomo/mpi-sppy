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

Rho consistency is the file writer's responsibility: PH requires the SAME rho
for a given nonant at a tree node across every scenario that passes through
that node. mpi-sppy applies these rhos per scenario and does NOT check
cross-scenario consistency (a full check would need a collective across ranks),
so an inconsistent set of {scenario}_rho.csv files silently yields an
ill-defined PH run.

Under proper bundling a bundle is one PH subproblem with a single rho per bundle
nonant, so its rho is assembled from the sub-scenarios' {scenario}_rho.csv files
by _rho_setter / _bundle_rho_list (the bundle itself has no rho file). PH uses one
rho per (node, nonant) across every scenario through the node, so the
sub-scenarios' values for a bundle nonant must agree. A bundle's sub-scenarios are
all local, so there mpi-sppy DOES check and raises an error on a mismatch (the
unbundled per-scenario path, lacking a cheap cross-scenario view, applies each
value unchecked).

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
import math
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

    # Optional {scenario}_rho.csv: stash its path for this module's _rho_setter
    # to read later (a private handshake within mps_module). Like
    # _mpisppy_node_list above, this is a plain creation-time attribute;
    # _mpisppy_data does not exist until SPBase sets the scenario up.
    rho_path = sharedPath + "_rho.csv"
    model._rho_csv_path = rho_path if os.path.exists(rho_path) else None

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

    Reads the csv path that scenario_creator stashed on the scenario as
    _rho_csv_path. Auto-discovered by the generic driver
    (generic/decomp.py:_get_rho_setter) and threaded to the hub/spokes. Returns
    an empty list when no rho file is present, so the --default-rho path is
    preserved. When a rho file is present it should cover every nonant (with
    --default-rho passed as a backstop), because defining _rho_setter bypasses
    the driver's --default-rho check.

    With proper bundling the object handed to a rho setter is a bundle EF, not
    an individual scenario, and the bundle carries no _rho_csv_path of its own.
    We detect that case and assemble the bundle's rho from its sub-scenarios'
    {s}_rho.csv files (see _bundle_rho_list); without this the per-scenario rho
    files would be silently ignored under bundling and every nonant would fall
    back to --default-rho.

    The csv writer must give identical rho to every scenario sharing a node for
    a given nonant (see the module docstring). The unbundled path applies each
    scenario's value without checking; within a bundle _bundle_rho_list checks
    (the sub-scenarios are local) and errors on a mismatch.
    """
    sub_models = _bundle_sub_models(scenario)
    if sub_models:
        return _bundle_rho_list(scenario, sub_models)
    path = getattr(scenario, "_rho_csv_path", None)
    if not path:
        return []
    return rho_utils.rho_list_from_csv(scenario, path)


def _bundle_sub_models(scenario):
    """Return the per-sub-scenario Pyomo models nested in a proper bundle.

    A proper bundle is an EF model that sputils.create_EF tags with
    _ef_scenario_names and to which it adds each sub-scenario as a named
    component. Returns [] when scenario is not such a bundle (a plain scenario,
    or the degenerate single-scenario EF where the scenario *is* the EF and so
    is not a component of itself), so callers fall back to the plain path.
    """
    names = getattr(scenario, "_ef_scenario_names", None)
    if not names:
        return []
    subs = [getattr(scenario, n, None) for n in names]
    return [s for s in subs if s is not None and s is not scenario]


def _bundle_rho_list(bundle, sub_models):
    """Assemble per-nonant rho for a proper bundle from its sub-scenarios.

    A proper bundle is a single PH subproblem with one rho per bundle nonant,
    yet each sub-scenario carries its own {s}_rho.csv. PH uses one rho per
    (node, nonant) shared across every scenario through that node, so the
    sub-scenarios' csv values for a given bundle nonant must agree; the bundle
    takes that shared value. A bundle's sub-scenarios are all local, so -- unlike
    the cross-scenario unbundled case, which would need an MPI collective -- we
    can and do check: a nonant whose sub-scenarios disagree on rho is a
    RuntimeError.

    bundle.consensus_groups maps each bundle nonant index to the per-sub-scenario
    Vars at that position. Returns [] when no sub-scenario carries a rho file,
    preserving the --default-rho fallback exactly as the plain-scenario path does.
    """
    rho_by_id = {}
    sub_by_id = {}   # id(Var) -> sub-scenario name, for error messages
    for sub in sub_models:
        path = getattr(sub, "_rho_csv_path", None)
        if not path or not os.path.exists(path):
            continue
        for vid, rho in rho_utils.rho_list_from_csv(sub, path):
            rho_by_id[vid] = rho
            sub_by_id[vid] = sub.name
    if not rho_by_id:
        return []
    consensus_groups = bundle.consensus_groups
    retlist = []
    for ndn_i, ref_var in bundle._mpisppy_data.nonant_indices.items():
        covered = [v for v in consensus_groups.get(ndn_i, (ref_var,))
                   if id(v) in rho_by_id]
        if not covered:
            continue
        rho0 = rho_by_id[id(covered[0])]
        if any(not math.isclose(rho_by_id[id(v)], rho0, rel_tol=1e-12, abs_tol=0.0)
               for v in covered):
            detail = ", ".join(f"{sub_by_id[id(v)]}={rho_by_id[id(v)]}"
                               for v in covered)
            raise RuntimeError(
                f"{bundle.name}: scenarios in this bundle give different rho for "
                f"nonant {ref_var.name} ({detail}). PH requires one rho per nonant "
                f"at a node; make the per-scenario {{s}}_rho.csv files agree."
            )
        retlist.append((id(ref_var), rho0))
    return retlist


# This is only needed for sampling
def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    # assert two stage, then do the usual for two stages?
    pass

    
#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass

