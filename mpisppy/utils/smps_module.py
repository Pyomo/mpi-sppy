###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Module interface for SMPS format problems, for use with generic_cylinders.py.

Usage:
    python -m mpisppy.generic_cylinders --module-name mpisppy.utils.smps_module \
        --smps-dir examples/sizes/SMPS --solver-name cplex --EF

The --smps-dir should point to a directory containing .cor, .tim, and .sto files.
Only SCENARIOS DISCRETE format is supported.
"""
import os
import shutil
import tempfile
import pyomo.environ as pyo
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.smps_reader as smps_reader
import mpisppy.utils.mps_reader as mps_reader

# Module-level state set by kw_creator
_smps_dir = None
_parsed = None  # cached parse results


def _ensure_parsed(smps_dir):
    """Parse SMPS files once and cache the results."""
    global _parsed
    if _parsed is not None and _parsed["smps_dir"] == smps_dir:
        return _parsed

    cor_path, tim_path, sto_path = smps_reader._find_smps_files(smps_dir)
    stages = smps_reader.parse_tim(tim_path)
    scenarios = smps_reader.parse_sto_discrete(sto_path)
    var_order = smps_reader.get_var_order_from_mps(cor_path)
    vars_by_stage = smps_reader.partition_vars_by_stage(var_order, stages)
    rhs_name = smps_reader.get_rhs_name_from_mps(cor_path)

    _parsed = {
        "smps_dir": smps_dir,
        "cor_path": cor_path,
        "stages": stages,
        "scenarios": scenarios,
        "var_order": var_order,
        "vars_by_stage": vars_by_stage,
        "rhs_name": rhs_name,
        "scen_by_name": {s["name"]: s for s in scenarios},
    }
    return _parsed


def _apply_rhs_modification(model, constr_name, value):
    """Modify the RHS of a constraint in the Pyomo model.

    Args:
        model: Pyomo ConcreteModel
        constr_name (str): constraint name
        value (float): new RHS value
    """
    constr = model.find_component(constr_name)
    if constr is None:
        raise RuntimeError(
            f"Constraint {constr_name} not found in model")
    # Determine constraint sense and update accordingly
    lb = constr.lower
    ub = constr.upper
    if lb is not None and ub is not None:
        if float(lb) == float(ub):
            # equality constraint
            constr.set_value((value, constr.body, value))
        else:
            raise RuntimeError(
                f"Range constraint {constr_name} not supported for RHS modification")
    elif lb is not None:
        # >= constraint: modify lower bound
        constr.set_value((value, constr.body, None))
    elif ub is not None:
        # <= constraint: modify upper bound
        constr.set_value((None, constr.body, value))
    else:
        raise RuntimeError(
            f"Free constraint {constr_name} has no RHS to modify")


def scenario_creator(scenario_name, cfg=None):
    """Create a scenario model from SMPS files.

    Args:
        scenario_name (str): name of the scenario (e.g., "SCEN01")
        cfg: Config object with smps_dir attribute

    Returns:
        Pyomo ConcreteModel with _mpisppy annotations
    """
    parsed = _ensure_parsed(cfg.smps_dir)
    scen_data = parsed["scen_by_name"].get(scenario_name)
    if scen_data is None:
        raise RuntimeError(
            f"Scenario {scenario_name} not found in .sto file. "
            f"Available: {list(parsed['scen_by_name'].keys())}")

    # Read the core model fresh (each scenario gets its own copy)
    # The mip library requires .mps extension, so symlink if needed
    cor_path = parsed["cor_path"]
    if not cor_path.lower().endswith(".mps"):
        mps_link = parsed.get("_mps_link")
        if mps_link is None or not os.path.exists(mps_link):
            tmpdir = tempfile.mkdtemp()
            mps_link = os.path.join(tmpdir, "core.mps")
            os.symlink(os.path.abspath(cor_path), mps_link)
            parsed["_mps_link"] = mps_link
        cor_path = mps_link
    model = mps_reader.read_mps_and_create_pyomo_model(cor_path)

    # Apply scenario modifications
    rhs_name = parsed["rhs_name"]
    for col_name, row_name, value in scen_data["modifications"]:
        if col_name == rhs_name:
            _apply_rhs_modification(model, row_name, value)
        else:
            raise RuntimeError(
                f"Only RHS modifications are supported, got column={col_name} "
                f"row={row_name}. Expected column to be '{rhs_name}'.")

    # Build scenario tree (2-stage: only ROOT node needed)
    stages = parsed["stages"]
    vars_by_stage = parsed["vars_by_stage"]

    # The nonant variables are those in the first stage (ROOT)
    root_stage_name = stages[0][0]
    nonant_var_names = vars_by_stage[root_stage_name]

    # Look up the Pyomo variables (mps_reader replaces parens with underscores)
    nonant_vars = []
    for vname in nonant_var_names:
        pyo_name = vname.replace("(", "_").replace(")", "_")
        v = model.find_component(pyo_name)
        if v is None:
            raise RuntimeError(
                f"Nonant variable {vname} (as {pyo_name}) not found in model")
        nonant_vars.append(v)

    # For 2-stage problems, only the ROOT node goes in the node list
    model._mpisppy_node_list = [
        scenario_tree.ScenarioNode(
            name="ROOT",
            cond_prob=1.0,
            stage=1,
            cost_expression=0.0,
            nonant_list=nonant_vars,
            scen_model=model,
            nonant_ef_suppl_list=None,
            parent_name=None,
        )
    ]
    model._mpisppy_probability = scen_data["probability"]

    return model


def scenario_names_creator(num_scens, start=None):
    """Return scenario names from the .sto file.

    Args:
        num_scens (int or None): number of scenarios (None = all)
        start (int or None): starting index (0-based)

    Returns:
        list of str: scenario names
    """
    parsed = _ensure_parsed(_smps_dir)
    all_names = [s["name"] for s in parsed["scenarios"]]

    if start is None:
        start = 0
    if num_scens is None:
        num_scens = len(all_names) - start

    assert start + num_scens <= len(all_names), \
        f"Requested {start=}, {num_scens=} but only {len(all_names)} scenarios available"

    return all_names[start:start + num_scens]


def inparser_adder(cfg):
    """Add --smps-dir to the config."""
    if "smps_dir" not in cfg:
        cfg.add_to_config("smps_dir",
                          "Directory containing .cor, .tim, .sto files",
                          domain=str,
                          default=None,
                          argparse=True)


def kw_creator(cfg):
    """Create keywords for scenario_creator. Side effect: set module-level _smps_dir."""
    global _smps_dir
    _smps_dir = cfg.smps_dir
    # Pre-parse so scenario_names_creator can work
    _ensure_parsed(_smps_dir)
    return {"cfg": cfg}


def scenario_denouement(rank, scenario_name, scenario):
    pass
