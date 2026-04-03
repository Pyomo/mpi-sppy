###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2025, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# example to write files from AMPL that allow loose coupling with mpi-sppy
# This is a fixed-format MPS file example
# Note that AMPL provides col and row files to get back to nice names.
#  (See farmer_free_writer for a free format writer that can't be read as of Oct 2025)

import os
import sys
import re
import json
from pathlib import Path
from typing import Iterable

from mpisppy.utils import config
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import numpy as np
from mpisppy import MPI  # for debugging

from amplpy import AMPL, add_to_path
add_to_path(r"full path to the AMPL installation directory")


fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

# If you need random numbers, use this random stream:
farmerstream = np.random.RandomState()  # pylint: disable=no-member

def scenario_creator(
    scenario_name,
    ampl_file_name,
    use_integer=False,
    sense=pyo.minimize,
    crops_multiplier=1,
    num_scens=None,
    seedoffset=0,
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
    elif scennum == 2:  # above
        y.set_values({"wheat": 3.0, "corn": 3.6, "beets": 24.0})

    areaVarDatas = list(ampl.get_variable("area").instances())

    try:
        obj_fct = ampl.get_objective("minus_profit")
    except Exception:
        print("big troubles!!; we can't find the objective function")
        raise
    return ampl, "uniform", areaVarDatas, obj_fct

# this function is fairly general
def _nonant_names_from_mps(
    mps_path: str,
    nonants: Iterable,
    col_map_path: str | None = None,
):
    """
    Given an MPS path and a list of nonants (AMPL var instances or strings),
    return the list of MPS column ids (C0001, C0002, ...) corresponding to them,
    by matching each nonant's *index tuple* to the AMPL-generated .col lines.

    We do NOT require knowing the base variable name; we match by the bracketed
    index (e.g., ['wheat'] or ['a','b']) that appears in the .col file.
    """

    def _read_name_list(path: Path):
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    out.append(s)
        return out

    # Normalize a bracketed index string like:
    #   "['wheat']"  -> "'wheat'"
    #   "[ 'a' , 'b' ]" -> "'a','b'"
    # We compare *only* the inner content (without the surrounding []), with:
    #   - single quotes
    #   - no spaces
    def _normalize_bracket_inner(bracket_str: str) -> str:
        s = bracket_str.replace('"', "'")
        # extract inside [...]
        m = re.search(r"\[([^\]]+)\]", s)
        if not m:
            return ""
        inner = m.group(1)
        # remove spaces around commas
        inner = re.sub(r"\s*,\s*", ",", inner)
        # remove stray spaces
        inner = re.sub(r"\s+", "", inner)
        return inner

    # Try to get "area['wheat']" etc. from an amplpy instance; fall back to str(x)
    def _extract_index_key_from_nonant(x) -> str | None:
        # 1) If the object has a name() or name attribute with brackets, use it
        try:
            nm = getattr(x, "name", None)
            if callable(nm):
                nm = nm()
            if isinstance(nm, str) and "[" in nm and "]" in nm:
                key = _normalize_bracket_inner(nm)
                if key:
                    return key
        except Exception:
            pass

        # 2) Parse from str(x). For amplpy VariableInstance, str(x) often looks like:
        #    "(('wheat',), <amplpy.ampl.Variable object at 0x...>)"
        sx = str(x).replace('"', "'")
        # Grab the first parenthesized tuple of indices inside the leading "( ... , <amplpy..."
        m = re.search(r"^\(\((.*?)\),\s*<amplpy\.ampl\.Variable", sx)
        if m:
            inner = m.group(1)         # e.g., "'wheat',"   or   "'a','b'"
            # normalize commas/spaces and ensure single quotes, no spaces
            inner = re.sub(r"\s*,\s*", ",", inner)
            inner = re.sub(r"\s+", "", inner)
            # strip trailing comma for singleton tuples like "'wheat',"
            inner = inner.rstrip(",")
            return inner

        # 3) As a last resort, if there is any [...] in str(x), use it
        m2 = re.search(r"\[([^\]]+)\]", sx)
        if m2:
            inner = re.sub(r"\s*,\s*", ",", m2.group(1))
            inner = re.sub(r"\s+", "", inner.replace('"', "'"))
            return inner

        return None

    # Build normalized index keys for the requested nonants
    target_keys = []
    for n in nonants:
        key = _extract_index_key_from_nonant(n)
        if key:
            target_keys.append(key)
    if not target_keys:
        raise ValueError(
            "Could not extract index tuples from the provided nonants. "
            f"Examples seen: {[str(n) for n in list(nonants)[:3]]}"
        )

    # Read .col and map each line's index to Cxxxx
    mps_p = Path(mps_path)
    col_p = Path(col_map_path) if col_map_path else mps_p.with_suffix(".col")
    if not col_p.exists():
        raise FileNotFoundError(f"Column map file not found: {col_p}")

    col_lines = _read_name_list(col_p)

    index_to_cid: dict[str, str] = {}
    for idx, raw in enumerate(col_lines, start=1):  # 1-based -> C0001
        key = _normalize_bracket_inner(raw)
        if key:  # only store lines that actually have brackets
            # If duplicates existed (rare), keep the first occurrence
            index_to_cid.setdefault(key, f"C{idx:04d}")

    # Map requested index keys to Cxxxx
    mps_ids: list[str] = []
    missing = []
    for key in target_keys:
        cid = index_to_cid.get(key)
        if cid:
            mps_ids.append(cid)
        else:
            missing.append(key)

    if missing:
        # Helpful debug: show a preview of what we saw
        preview = [ell for ell in col_lines[:10]]
        raise ValueError(
            "Some nonants were not found in .col. "
            f"Missing keys (normalized inside []): {missing}. "
            f"First .col lines: {preview}"
        )

    return mps_ids


def write_mps_file(ampl: AMPL, stub: str, name_maps: bool = True):
    """Write <stub>.mps (and <stub>.row/.col if name_maps)."""
    if name_maps:
        ampl.eval('option auxfiles rc;')
    # AMPL requires: write m<stub>;  (no space, no quotes)
    ampl.eval(f'write m{stub};')


def check_empty_dir(dirname: str) -> bool:
    """Require that dirname exists and is an empty directory."""
    if not os.path.isdir(dirname):
        print(f"Error: '{dirname}' is not a valid directory path.", file=sys.stderr)
        return False
    if os.listdir(dirname):
        print(f"Error: Directory '{dirname}' is not empty.", file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    num_scens = 3
    ampl_file_name = "farmer.mod"

    cfg = config.Config()
    cfg.add_to_config(
        "output_directory",
        description="The directory where scenario files will be written",
        domain=str,
        default=None,
        argparse_args={"required": True},
    )
    cfg.parse_command_line("farmer_writer.py")

    dirname = cfg.output_directory
    if not check_empty_dir(dirname):
        raise RuntimeError(f"{dirname} must exist and be empty")

    namebase = os.path.join(dirname, "scen")

    for s in range(num_scens):
        # scenario_name should contain the scenario number for extract_num();
        # we keep the simple "scen{s}" (digits at the end are what matters).
        scenario_name = f"scen{s}"
        ampl, prob, nonants, obj_fct = scenario_creator(
            scenario_name, ampl_file_name, num_scens=num_scens
        )
        print(f"we have the ampl model for scenario {s}")

        # Use a path STUB (no extension) so AMPL writes .mps/.row/.col correctly
        stub = f"{namebase}{s}"
        write_mps_file(ampl, stub, name_maps=True)

        mps = f"{stub}.mps"
        row = f"{stub}.row"
        col = f"{stub}.col"

        print(f"  wrote {mps}, {row}, and {col}.")

        # --- Write {stub}_nonants.json ---
        # Scenario probability
        if prob == "uniform":
            scenProb = 1.0 / num_scens
        else:
            scenProb = float(prob)

        nonant_names = _nonant_names_from_mps(mps, nonants, col)

        data = {
            "scenarioData": {
                "name": f"scen{s}",
                "scenProb": scenProb,
            },
            "treeData": {
                "globalNodeCount": 1,
                "nodes": {
                    "ROOT": {
                        "serialNumber": 0,
                        "condProb": 1.0,
                        "nonAnts": nonant_names,
                    }
                },
            },
        }

        with open(f"{stub}_nonants.json", "w", encoding="utf-8") as jf:
            json.dump(data, jf, indent=2)
        print(f"  wrote {stub}_nonants.json")

        # --- Write {stub}_rho.csv ---
        default_rho = 1.0  # or whatever value you want to use globally
        rho_filename = f"{stub}_rho.csv"
        with open(rho_filename, "w", encoding="utf-8") as csvf:
            csvf.write("varname,rho\n")
            for name in nonant_names:
                csvf.write(f"{name},{default_rho}\n")
        print(f"  wrote {rho_filename}")
