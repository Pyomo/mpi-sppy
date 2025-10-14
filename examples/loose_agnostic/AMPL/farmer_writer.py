###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# In this example, AMPL is the guest language.
# This is the python model file for AMPL farmer.
# It will work with farmer.mod and slight deviations.

import re
import shutil
from pathlib import Path

from amplpy import AMPL, add_to_path
add_to_path(r"full path to the AMPL installation directory")
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import mpisppy.agnostic.examples.farmer as farmer
import numpy as np
from mpisppy import MPI  # for debugging
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

# If you need random numbers, use this random stream:
farmerstream = np.random.RandomState()  # pylint: disable=no-member

def _sanitize_name(name: str, limit: int = 8):
    """
    Sanitize a symbolic name for MPS:
      - Replace '[' with '(' and ']' with ')'
      - Allow letters, digits, underscore, and parentheses
      - Replace any other char with '_', collapse repeats
      - If first char isn't a letter, prefix with 'N'
      - Truncate to 'limit' chars (0 = no truncation)
    """
    # 1) Brackets to parentheses
    s = name.replace("[", "(").replace("]", ")")

    # 2) Keep only A-Z a-z 0-9 _ ( )
    s = re.sub(r"[^A-Za-z0-9_()]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    # 3) Ensure first char is a letter
    if not s:
        s = "N"
    if not re.match(r"[A-Za-z]", s[0]):
        s = "N" + s

    # 4) Truncate (classic MPS = 8 chars; free MPS = 0 for unlimited)
    return s[:limit] if limit else s


def _make_unique(names):
    """
    Ensure list of names are unique by appending _1, _2, ... when needed.
    Returns a new list with unique names.
    """
    seen = {}
    out = []
    for n in names:
        base = n
        k = seen.get(base, 0)
        if k == 0 and base not in seen:
            out.append(base)
            seen[base] = 1
        else:
            # bump until unique
            while True:
                k += 1
                cand = f"{base}_{k}"
                if cand not in seen:
                    out.append(cand)
                    seen[base] = k
                    seen[cand] = 1
                    break
    return out

def _read_name_list(path: Path):
    """Read a one-name-per-line file; strip whitespace; ignore blank lines."""
    names = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s != "":
                names.append(s)
    return names

def rewrite_mps_with_meaningful_names(
    mps_path: str,
    row_map_path: str,
    col_map_path: str,
    out_path: str | None = None,
    free_names: bool = True,
):
    """
    Replace R000i / C000j names in an AMPL-written MPS using .row / .col.

    Parameters
    ----------
    mps_path : str
        Path to original MPS (e.g., 'scen0.mps').
    row_map_path : str
        Path to .row file (e.g., 'scen0.row') listing row names in R0001..order.
    col_map_path : str
        Path to .col file (e.g., 'scen0.col') listing col names in C0001..order.
    out_path : str | None
        Output path. If None, overwrite the input MPS.
    free_names : bool
        If True, allow longer names (typical “free” MPS parsers like Gurobi/CPLEX accept).
        If False, enforce 8-char classic MPS names.
    """
    mps_path = Path(mps_path)
    row_path = Path(row_map_path)
    col_path = Path(col_map_path)
    out_path = Path(out_path) if out_path else mps_path

    # Read mapping lists
    row_names_raw = _read_name_list(row_path)
    col_names_raw = _read_name_list(col_path)

    # Sanitize + enforce uniqueness
    limit = 0 if free_names else 8
    row_names_san = [_sanitize_name(n, limit=limit) for n in row_names_raw]
    col_names_san = [_sanitize_name(n, limit=limit) for n in col_names_raw]
    row_names = _make_unique(row_names_san)
    col_names = _make_unique(col_names_san)

    # Build R000i/C000j -> meaningful name maps
    row_map = {f"R{i:04d}": row_names[i-1] for i in range(1, len(row_names)+1)}
    col_map = {f"C{i:04d}": col_names[i-1] for i in range(1, len(col_names)+1)}

    # Parse and rewrite the MPS
    lines_out = []
    section = None  # None | 'ROWS' | 'COLUMNS' | 'RHS' | 'BOUNDS' | 'RANGES'
    with mps_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # Section tracking
            u = line.strip().upper()
            if u == "ROWS":
                section = "ROWS"
                lines_out.append(line)
                continue
            elif u == "COLUMNS":
                section = "COLUMNS"
                lines_out.append(line)
                continue
            elif u == "RHS":
                section = "RHS"
                lines_out.append(line)
                continue
            elif u == "BOUNDS":
                section = "BOUNDS"
                lines_out.append(line)
                continue
            elif u == "RANGES":
                section = "RANGES"
                lines_out.append(line)
                continue
            elif u == "ENDATA":
                section = None
                lines_out.append(line)
                continue
            elif u == "NAME" or u.startswith("NAME "):
                # Keep the NAME line as-is
                lines_out.append(line)
                continue

            # Rewrite based on section
            if section == "ROWS":
                # Example: " L  R0001"
                toks = line.split()
                if len(toks) >= 2:
                    # toks[0] is row type (N, L, G, E)
                    # toks[1] is row name
                    rname = toks[1]
                    toks[1] = row_map.get(rname, rname)
                    lines_out.append("  ".join(toks))
                else:
                    lines_out.append(line)

            elif section == "COLUMNS":
                # Examples:
                # "    C0001     R0001     1"
                # "    C0001     R0002     2    R0006     150"
                toks = line.split()
                if not toks:
                    lines_out.append(line)
                    continue
                # First token is column name
                col = toks[0]
                toks[0] = col_map.get(col, col)
                # Remaining tokens come in pairs: row value [row value]
                for i in range(1, len(toks), 2):
                    # Guard in case of odd token count
                    if i < len(toks):
                        name_or_value = toks[i]
                        # If it's a row token, replace; values will be numbers and left alone
                        # We can safely check if starts with 'R' digit; otherwise look up map
                        toks[i] = row_map.get(name_or_value, name_or_value)
                lines_out.append("  ".join(toks))

            elif section == "RHS":
                # Example: "    B         R0001     500"
                # tokens: rhs_name row_name value [row_name value]
                toks = line.split()
                if len(toks) >= 3:
                    # Replace row names at positions 1,3,...
                    for i in range(1, len(toks), 2):
                        toks[i] = row_map.get(toks[i], toks[i])
                    lines_out.append("  ".join(toks))
                else:
                    lines_out.append(line)

            elif section == "BOUNDS":
                # Example: " UP BOUND     C0006     6000"
                # tokens: btype bnd_name col_name [value]
                toks = line.split()
                if len(toks) >= 3:
                    # Column name is at index 2
                    toks[2] = col_map.get(toks[2], toks[2])
                    lines_out.append("  ".join(toks))
                else:
                    lines_out.append(line)

            elif section == "RANGES":
                # Similar structure to RHS: name, row, value pairs
                toks = line.split()
                if len(toks) >= 3:
                    for i in range(1, len(toks), 2):
                        toks[i] = row_map.get(toks[i], toks[i])
                    lines_out.append("  ".join(toks))
                else:
                    lines_out.append(line)

            else:
                # Outside sections, copy through
                lines_out.append(line)

    # Write result
    with out_path.open("w", encoding="utf-8") as g:
        for l in lines_out:
            g.write(l + "\n")

    return out_path

# --- Example usage for your files ---
# rewrite_mps_with_meaningful_names("scen0.mps", "scen0.row", "scen0.col", out_path="scen0_named.mps", free_names=True)
# If you truly need classic 8-char names, set free_names=False and it will truncate safely.


def scenario_creator(scenario_name, ampl_file_name,
                     use_integer=False, sense=pyo.minimize, crops_multiplier=1,
                     num_scens=None, seedoffset=0
                     ):
    """ Create a scenario for the (scalable) farmer example
    
    Args:
        scenario_name (str):
            Name of the scenario to construct.
        ampl_file_name (str):
            The name of the ampl model file (with AMPL in it)
            (This adds flexibility that maybe we don't need; it could be hardwired)
        use_integer (bool, optional):
            If True, restricts variables to be integer. Default is False.
        sense (int, optional):
            Model sense (minimization or maximization). Must be either
            pyo.minimize or pyo.maximize. Default is pyo.minimize.
        crops_multiplier (int, optional):
            Factor to control scaling. There will be three times this many
            crops. Default is 1.
        num_scens (int, optional):
            Number of scenarios. We use it to compute _mpisppy_probability. 
            Default is None.
        seedoffset (int): used by confidence interval code

    NOTE: for ampl, the names will be tuples name, index
    
    Returns:
        ampl_model (AMPL object): the AMPL model
        prob (float or "uniform"): the scenario probability
        nonant_var_data_list (list of AMPL variables): the nonants
        obj_fct (AMPL Objective function): the objective function
    """

    assert crops_multiplier == 1, "for AMPL, just getting started with 3 crops"

    ampl = AMPL()

    ampl.read(ampl_file_name)

    # scenario specific data applied
    scennum = sputils.extract_num(scenario_name)
    assert scennum < 3, "three scenarios hardwired for now"
    y = ampl.get_parameter("RandomYield")
    if scennum == 0:  # below
        y.set_values({"wheat": 2.0, "corn": 2.4, "beets": 16.0})
    elif scennum == 2: # above
        y.set_values({"wheat": 3.0, "corn": 3.6, "beets": 24.0})

    areaVarDatas = list(ampl.get_variable("area").instances())

    try:
        obj_fct = ampl.get_objective("minus_profit")
    except:
        print("big troubles!!; we can't find the objective function")
        raise
    return ampl, "uniform", areaVarDatas, obj_fct
    

def write_mps_file(ampl, s, name_maps=True):
    """Write scen{s}.mps using AMPL's 'write m<stub>' syntax.
       If name_maps=True, also writes scen{s}.row/.col mapping files."""
    ampl.eval(f'option auxfiles {"rc" if name_maps else ""};')
    ampl.eval(f'write mscen{s};')   # produces scen{s}.mps (and .row/.col if auxfiles rc)
    return f"scen{s}.mps"

if __name__ == "__main__":
    num_scens = 3
    ampl_file_name = "farmer.mod"
    for s in range(num_scens):
        ampl, prob, nonants, obj_fct = scenario_creator(
            f"scen{s}", ampl_file_name, num_scens=num_scens
        )
        print(f"we have the ampl model for scenario {s}")

        mps_file = write_mps_file(ampl, s, name_maps=True)
        assert mps_file == f"scen{s}.mps"
        print(f"wrote {mps_file}, but now re-writing with better names")
        rewrite_mps_with_meaningful_names(f"scen{s}.mps",
                                          f"scen{s}.row",
                                          f"scen{s}.col",
                                          out_path=f"scen{s}_named.mps",
                                          free_names=True)
        shutil.copyfile(f"scen{s}.mps", f"scen{s}_densenames.mps")
        shutil.copyfile(f"scen{s}_named.mps", f"scen{s}.mps")
        print(f"  wrote {mps_file}, with better names.")
        
