###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Reader for SMPS format stochastic programming problems.

Supports the SCENARIOS DISCRETE format only (for now).
The three SMPS files are:
  .cor  — core deterministic model in MPS format
  .tim  — time/stage structure (PERIODS section)
  .sto  — stochastic data (SCENARIOS DISCRETE section)

The .cor file is read by mps_reader.read_mps_and_create_pyomo_model.
This module parses .tim and .sto, and provides functions to create
scenario Pyomo models with the appropriate modifications.
"""
import os
import glob


def _find_smps_files(smps_dir):
    """Find the .cor, .tim, and .sto files in the given directory.

    Args:
        smps_dir (str): path to the SMPS directory

    Returns:
        tuple: (cor_path, tim_path, sto_path)
    """
    def _find_one(ext):
        matches = glob.glob(os.path.join(smps_dir, f"*{ext}"))
        if len(matches) == 0:
            raise FileNotFoundError(
                f"No {ext} file found in {smps_dir}")
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple {ext} files found in {smps_dir}: {matches}")
        return matches[0]

    return _find_one(".cor"), _find_one(".tim"), _find_one(".sto")


def parse_tim(tim_path):
    """Parse a .tim file to get stage boundary information.

    Args:
        tim_path (str): path to the .tim file

    Returns:
        list of tuples: [(stage_name, first_var, first_constraint), ...]
            ordered by appearance in the file (i.e., by stage)
    """
    stages = []
    in_periods = False
    with open(tim_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*"):
                continue
            parts = line.split()
            if parts[0] == "TIME":
                continue
            if parts[0] == "PERIODS":
                in_periods = True
                continue
            if parts[0] == "ENDATA":
                break
            if in_periods:
                # format: first_var  first_constraint  stage_name
                first_var = parts[0]
                first_constr = parts[1]
                stage_name = parts[2]
                stages.append((stage_name, first_var, first_constr))
    if not stages:
        raise RuntimeError(f"No stage information found in {tim_path}")
    return stages


def parse_sto_discrete(sto_path):
    """Parse a .sto file with SCENARIOS DISCRETE format.

    Args:
        sto_path (str): path to the .sto file

    Returns:
        list of dicts, each with keys:
            "name": scenario name (str)
            "parent": parent node name (str)
            "probability": scenario probability (float)
            "stage": stage name (str)
            "modifications": list of (col_name, row_name, value) tuples
    """
    scenarios = []
    current_scen = None
    found_discrete = False

    with open(sto_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*"):
                continue
            parts = line.split()
            if parts[0] == "STOCH":
                continue
            if parts[0] == "SCENARIOS":
                if parts[1] != "DISCRETE":
                    raise RuntimeError(
                        f"Only SCENARIOS DISCRETE is supported, got: {line}")
                found_discrete = True
                continue
            if parts[0] == "ENDATA":
                break

            if not found_discrete:
                continue

            if parts[0] == "SC":
                # SC <name> <parent> <probability> <stage>
                if current_scen is not None:
                    scenarios.append(current_scen)
                current_scen = {
                    "name": parts[1],
                    "parent": parts[2],
                    "probability": float(parts[3]),
                    "stage": parts[4],
                    "modifications": [],
                }
            else:
                # modification line: col_name row_name value
                if current_scen is None:
                    raise RuntimeError(
                        f"Data line before any SC declaration: {line}")
                col_name = parts[0]
                row_name = parts[1]
                value = float(parts[2])
                current_scen["modifications"].append(
                    (col_name, row_name, value))

    if current_scen is not None:
        scenarios.append(current_scen)

    if not found_discrete:
        raise RuntimeError(
            f"No SCENARIOS DISCRETE section found in {sto_path}")

    return scenarios


def get_var_order_from_mps(mps_path):
    """Extract ordered list of unique variable names from MPS COLUMNS section.

    Args:
        mps_path (str): path to the MPS (.cor) file

    Returns:
        list of str: variable names in MPS column order
    """
    var_names = []
    seen = set()
    section = None
    with open(mps_path, "r", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*"):
                continue
            parts = line.split()
            tok0 = parts[0]
            if tok0 in ("NAME", "ROWS"):
                section = tok0
                continue
            if tok0 == "COLUMNS":
                section = "COLUMNS"
                continue
            if tok0 in ("RHS", "RANGES", "BOUNDS", "ENDATA"):
                if section == "COLUMNS":
                    break
                section = tok0
                continue
            if section == "COLUMNS":
                if "'MARKER'" in line:
                    continue
                vname = parts[0]
                if vname not in seen:
                    var_names.append(vname)
                    seen.add(vname)
    return var_names


def get_rhs_name_from_mps(mps_path):
    """Extract the RHS name from the MPS file.

    Args:
        mps_path (str): path to the MPS (.cor) file

    Returns:
        str: the RHS name (e.g., "RHS1")
    """
    section = None
    with open(mps_path, "r", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*"):
                continue
            parts = line.split()
            tok0 = parts[0]
            if tok0 == "RHS":
                section = "RHS"
                continue
            if section == "RHS":
                if tok0 in ("RANGES", "BOUNDS", "ENDATA"):
                    break
                return parts[0]
    return None


def get_bounds_name_from_mps(mps_path):
    """Extract the BOUNDS name from the MPS file.

    Args:
        mps_path (str): path to the MPS (.cor) file

    Returns:
        str or None: the bounds name (e.g., "BND1"), or None if no BOUNDS section
    """
    section = None
    with open(mps_path, "r", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*"):
                continue
            parts = line.split()
            tok0 = parts[0]
            if tok0 == "BOUNDS":
                section = "BOUNDS"
                continue
            if section == "BOUNDS":
                if tok0 == "ENDATA":
                    break
                # Bounds lines: type name var_name [value]
                # The name is in parts[1]
                return parts[1]
    return None


def get_bound_types_from_mps(mps_path, bounds_name):
    """Extract bound types for each variable from the MPS BOUNDS section.

    In SMPS SCENARIOS DISCRETE, bounds modifications replace the bound value
    but don't specify the bound type; that comes from the .cor file.

    Args:
        mps_path (str): path to the MPS (.cor) file
        bounds_name (str): the bounds vector name (e.g., "BND1")

    Returns:
        dict: var_name -> bound_type (e.g., "LO", "UP", "FX")
              If a variable has multiple bound entries, the last one wins.
    """
    bound_types = {}
    section = None
    with open(mps_path, "r", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*"):
                continue
            parts = line.split()
            tok0 = parts[0]
            if tok0 == "BOUNDS":
                section = "BOUNDS"
                continue
            if section == "BOUNDS":
                if tok0 == "ENDATA":
                    break
                # Bounds lines: type name var_name [value]
                btype = parts[0]
                bname = parts[1]
                vname = parts[2]
                if bname == bounds_name:
                    bound_types[vname] = btype
    return bound_types


def partition_vars_by_stage(var_order, stages):
    """Partition variables into stages based on .tim boundaries.

    Args:
        var_order (list of str): variable names in MPS column order
        stages (list of tuples): from parse_tim, each (stage_name, first_var, first_constr)

    Returns:
        dict: stage_name -> list of variable names
    """
    # Build a map from variable name to its index in the ordering
    var_idx = {v: i for i, v in enumerate(var_order)}

    # Get the starting index for each stage
    stage_boundaries = []
    for stage_name, first_var, _first_constr in stages:
        if first_var not in var_idx:
            raise RuntimeError(
                f"Variable {first_var} from .tim file not found in .cor file")
        stage_boundaries.append((var_idx[first_var], stage_name))
    stage_boundaries.sort()

    # Partition
    result = {}
    for i, (start_idx, stage_name) in enumerate(stage_boundaries):
        if i + 1 < len(stage_boundaries):
            end_idx = stage_boundaries[i + 1][0]
        else:
            end_idx = len(var_order)
        result[stage_name] = var_order[start_idx:end_idx]

    return result
