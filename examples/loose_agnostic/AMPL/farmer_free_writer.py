# example to write files from AMPL that allow loose coupling with mpi-sppy
# NOTE: There is a lot of code here to create a nice free-format MPS file,
#   but as of October 2025, the software we are using read mps cannot
#   really handle it. Use lp files or use fixed format as in farmer_writer.py.

import os
import sys
import re
import shutil
import json
from pathlib import Path

from mpisppy.utils import config
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
    row_map = {f"R{i:04d}": row_names[i - 1] for i in range(1, len(row_names) + 1)}
    col_map = {f"C{i:04d}": col_names[i - 1] for i in range(1, len(col_names) + 1)}

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
                    if i < len(toks):
                        name_or_value = toks[i]
                        toks[i] = row_map.get(name_or_value, name_or_value)
                lines_out.append("  ".join(toks))

            elif section == "RHS":
                # Example: "    B         R0001     500"
                # tokens: rhs_name row_name value [row_name value]
                toks = line.split()
                if len(toks) >= 3:
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


def write_mps_file(ampl: AMPL, stub: str, name_maps: bool = True):
    """Write <stub>.mps (and <stub>.row/.col if name_maps)."""
    if name_maps:
        ampl.eval('option auxfiles rc;')
    # AMPL requires: write m<stub>;  (no space, no quotes)
    ampl.eval(f'write m{stub};')


def _nonant_names_from_mps(mps_path, nonant_var_base="area"):
    """
    Parse the MPS file and extract the nonant variable names
    (e.g., area(_wheat_), area(_corn_), area(_beets_)).
    Only keeps names starting with `nonant_var_base`.
    """
    names = []
    with open(mps_path, "r", encoding="utf-8") as f:
        in_columns = False
        for line in f:
            u = line.strip().upper()
            if u == "COLUMNS":
                in_columns = True
                continue
            if u in {"RHS", "BOUNDS", "RANGES", "ENDATA"}:
                in_columns = False
            if not in_columns:
                continue

            tokens = line.split()
            if tokens:
                var = tokens[0]
                if var.startswith(nonant_var_base):
                    if var not in names:
                        names.append(var)
    return names


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

        print(f"wrote {mps}, but now re-writing with better names")
        rewrite_mps_with_meaningful_names(
            mps,
            row,
            col,
            out_path=f"{stub}_named.mps",
            free_names=True,
        )

        # Keep a copy of the dense-name original and then replace .mps with the named one
        shutil.copyfile(mps, f"{stub}_densenames.mps")
        shutil.copyfile(f"{stub}_named.mps", mps)
        print(f"  wrote {mps}, with better names.")

        # --- Write {stub}_nonants.json ---
        # Scenario probability
        if prob == "uniform":
            scenProb = 1.0 / num_scens
        else:
            scenProb = float(prob)

        nonant_names = _nonant_names_from_mps(mps, nonant_var_base="area")

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
