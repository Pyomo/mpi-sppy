#!/usr/bin/env python3
###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
###############################################################################

import os
import sys
import json
import re
import subprocess
from pathlib import Path
from shutil import which

import numpy as np  # optional
import mpisppy.utils.sputils as sputils  # optional
from mpisppy.utils import config

# ---------------- Helpers ----------------
def check_empty_dir(dirname: str) -> bool:
    if not os.path.isdir(dirname):
        print(f"Error: '{dirname}' is not a valid directory path.", file=sys.stderr)
        return False
    if os.listdir(dirname):
        print(f"Error: Directory '{dirname}' is not empty.", file=sys.stderr)
        return False
    return True

def _find_case_insensitive(sdir: Path, *names: str) -> Path | None:
    """Search top-level, then recursively (case-insensitive) for any of the given names."""
    lower_top = {p.name.lower(): p for p in sdir.iterdir() if p.is_file()}
    for nm in names:
        p = lower_top.get(nm.lower())
        if p:
            return p
    # recursive fallback
    cands = []
    wanted = {n.lower() for n in names}
    for p in sdir.rglob("*"):
        if p.is_file() and p.name.lower() in wanted:
            cands.append(p)
    return sorted(cands)[0] if cands else None


def _pick_from_filelist(scen_dir: Path) -> Path | None:
    """
    If CONVERT produced a file list, parse it and return the first plausible
    model artifact (*.mps or *.lp). Search recursively for listed basenames.
    """
    # file list names we accept (documented default is 'files.txt')
    for name in ("files.txt", "filelist.txt", "file.txt"):
        fl = next((p for p in scen_dir.rglob("*") if p.is_file() and p.name.lower() == name), None)
        if fl:
            break
    else:
        return None

    try:
        lines = [ln.strip() for ln in fl.read_text(encoding="utf-8", errors="ignore").splitlines()]
    except Exception:
        return None

    prio_exts  = (".mps", ".lp")
    prio_names = ("fixed.mps", "cplex.mps", "model.mps", "cplex.lp", "model.lp")

    files = []
    for ln in lines:
        if ln and not ln.startswith("*"):
            files.append(Path(ln).name)  # only the basename

    # name preference
    for want in prio_names:
        for f in files:
            if f.lower() == want:
                hits = list(scen_dir.rglob(f)) or list(scen_dir.rglob(f.upper())) or list(scen_dir.rglob(f.capitalize()))
                if hits:
                    return sorted(hits)[0]

    # extension preference
    for ext in prio_exts:
        for f in files:
            if f.lower().endswith(ext):
                hits = list(scen_dir.rglob(f)) or list(scen_dir.rglob(f.upper())) or list(scen_dir.rglob(f.capitalize()))
                if hits:
                    return sorted(hits)[0]

    # single-entry fallback
    uniq = sorted(set(files))
    if len(uniq) == 1:
        hits = list(scen_dir.rglob(uniq[0])) or list(scen_dir.rglob(uniq[0].upper())) or list(scen_dir.rglob(uniq[0].capitalize()))
        if hits:
            return sorted(hits)[0]

    return None


def _parse_gams_dict_for_nonants(dict_path: Path,
                                 nonant_prefixes=("area(", "area[")):
    mps_cols, orig_names = [], []
    with dict_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r"\s*([exbi]\d+)\s+(.*\S)\s*$", line)
            if not m:
                continue
            scalar_name, original = m.group(1), m.group(2)
            if any(original.startswith(p) for p in nonant_prefixes):
                mps_cols.append(scalar_name)
                orig_names.append(original)
    if not mps_cols:
        raise RuntimeError(
            f"No nonant variables found in {dict_path}. "
            f"Looked for prefixes: {nonant_prefixes}."
        )
    return mps_cols, orig_names


def _write_convert_opt(where: Path):
    """Minimal options compatible with your CONVERT build."""
    (where / "convert.opt").write_text(
        "FixedMPS 1\n"
        "Dict dict.txt\n",
        encoding="utf-8",
    )


def _resolve_gams_bin(cli_value: str | None) -> str:
    if cli_value:
        p = Path(cli_value).expanduser().resolve()
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
        raise RuntimeError(f"--gams_bin points to a non-executable: {p}")

    w = which("gams")
    if w:
        return w

    for envvar in ("GAMS", "GAMS_SYS_DIR"):
        base = os.environ.get(envvar)
        if base:
            cand = Path(base) / "gams"
            if cand.exists() and os.access(cand, os.X_OK):
                return str(cand)

    raise RuntimeError(
        "Could not locate the GAMS executable.\n"
        "Install GAMS and ensure `gams` is on PATH, or pass --gams_bin /full/path/to/gams."
    )


def _run_gams_convert(gams_bin: str, gms_file: Path, workdir: Path):
    """
    Run GAMS in workdir; the scenario .gms sets `option lp=convert;`.
    We pass `optfile=1` so the solver reads convert.opt.
    """
    gms_arg = gms_file.name  # just 'scenX.gms'
    cmd = [gams_bin, gms_arg, "lo=3", "o=convert.log", "optfile=1"]
    res = subprocess.run(cmd, cwd=str(workdir),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        log_path = workdir / "convert.log"
        log_tail = ""
        if log_path.exists():
            try:
                with log_path.open("r", encoding="utf-8", errors="ignore") as lf:
                    lines = lf.readlines()
                    log_tail = "\n--- convert.log (last 200 lines) ---\n" + "".join(lines[-200:])
            except Exception:
                pass
        raise RuntimeError(
            f"GAMS CONVERT failed (rc={res.returncode}).\n"
            f"Command: {' '.join(cmd)}\nCWD: {workdir}\n"
            f"--- stdout/stderr ---\n{res.stdout}{log_tail}"
        )


def _detect_model_name(gms_path_or_text) -> str | None:
    if isinstance(gms_path_or_text, Path):
        txt = gms_path_or_text.read_text(encoding="utf-8", errors="ignore")
    else:
        txt = gms_path_or_text
    m = re.search(r"(?im)^\s*Model\s+([A-Za-z_]\w*)\s*/", txt)
    return m.group(1) if m else None


def _inject_before_solve(gms_text: str, to_insert: str) -> str:
    m = re.search(r"(?im)^\s*solve\b", gms_text)
    if not m:
        return gms_text.rstrip() + "\n" + to_insert + "\n"
    idx = m.start()
    return gms_text[:idx] + to_insert + "\n" + gms_text[idx:]


def _find_dict_file(scen_dir: Path) -> Path | None:
    # top-level, case-insensitive
    top = {p.name.lower(): p for p in scen_dir.iterdir() if p.is_file()}
    for name in ("dict.txt", "gamsdict.txt"):
        if name in top:
            return top[name]
    # recursive fallback, case-insensitive
    candidates = []
    for p in scen_dir.rglob("*"):
        if p.is_file() and p.name.lower() in ("dict.txt", "gamsdict.txt"):
            candidates.append(p)
    return sorted(candidates)[0] if candidates else None


def _pick_converted_file(scen_dir: Path, model_name: str) -> tuple[Path | None, list[str]]:
    """
    Case-insensitive, recursive artifact search.
    Preference:
      1) fixed.mps
      2) <model_name>.mps
      3) model.mps
      4) single *.mps anywhere
      5) <model_name>.lp
      6) model.lp
      7) single *.lp anywhere
    """
    present = [str(p.relative_to(scen_dir)) for p in sorted(scen_dir.rglob("*"))]

    def find_exact_ci(relname: str) -> Path | None:
        target_lower = relname.lower()
        for p in scen_dir.iterdir():
            if p.name.lower() == target_lower:
                return p
        return None

    for name in ("fixed.mps", f"{model_name}.mps", "model.mps"):
        p = find_exact_ci(name)
        if p:
            return p, present

    mps_list = [p for p in scen_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".mps"]
    if len(mps_list) == 1:
        return mps_list[0], present

    for name in (f"{model_name}.lp", "model.lp"):
        p = find_exact_ci(name)
        if p:
            return p, present

    lp_list = [p for p in scen_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".lp"]
    if len(lp_list) == 1:
        return lp_list[0], present

    return None, present


def _convert_write_mps_and_dict_from_gms(gams_bin: str, src_gms: Path, outdir: Path, stub: str):
    """
    Run GAMS/CONVERT and copy the produced model artifact to <stub>.mps.
    Works with older CONVERT that writes FixedMPS to a file literally named '1'.
    """
    scen_dir = src_gms.parent
    scen_dir.mkdir(parents=True, exist_ok=True)

    # Minimal, compatible convert.opt
    _write_convert_opt(scen_dir)

    # Run GAMS
    _run_gams_convert(gams_bin, src_gms, scen_dir)

    # Find dict.txt (case-insensitive, recursive)
    def _find_dict_anywhere(root: Path) -> Path | None:
        top = {p.name.lower(): p for p in root.iterdir() if p.is_file()}
        for nm in ("dict.txt", "gamsdict.txt"):
            if nm in top:
                return top[nm]
        hits = [p for p in root.rglob("*") if p.is_file() and p.name.lower() in ("dict.txt", "gamsdict.txt")]
        return sorted(hits)[0] if hits else None

    dict_txt = _find_dict_anywhere(scen_dir)
    if dict_txt is None:
        present_top = [p.name for p in sorted(scen_dir.iterdir())]
        raise RuntimeError(
            "Dictionary file not found in scenario directory.\n"
            f"Looked for dict.txt/gamsdict.txt under: {scen_dir}\n"
            f"Top-level contents: {present_top}"
        )

    # Primary search: *.mps / *.lp anywhere under scen_dir (case-insensitive)
    def _pick_any_artifact(root: Path) -> Path | None:
        preferred = ("fixed.mps", "model.mps", "cplex.mps", "model.lp", "cplex.lp")
        # preferred names at top-level
        for p in root.glob("*"):
            if p.is_file() and p.name.lower() in preferred:
                return p
        # any single .mps / .lp recursively
        mps = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".mps"]
        if len(mps) == 1:
            return mps[0]
        lp  = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".lp"]
        if len(lp) == 1:
            return lp[0]
        # if multiple, pick by preferred names
        for nm in preferred:
            cand = [p for p in root.rglob("*") if p.is_file() and p.name.lower() == nm]
            if cand:
                return sorted(cand)[0]
        return None

    chosen = _pick_any_artifact(scen_dir)

    # Fallback for older CONVERT: it writes the FixedMPS to a file named '1'
    if chosen is None:
        # top-level numeric file
        num_files = [p for p in scen_dir.iterdir() if p.is_file() and p.name.isdigit()]
        if not num_files:
            # recursive numeric file (rare, but try)
            num_files = [p for p in scen_dir.rglob("*") if p.is_file() and p.name.isdigit()]
        if num_files:
            # choose the largest non-empty numeric file as the artifact
            num_files = [p for p in num_files if p.stat().st_size > 0]
            if num_files:
                chosen = max(num_files, key=lambda p: p.stat().st_size)

    if chosen is None:
        # show helpful diagnostics (tail of convert.log and listing)
        log_tail = ""
        log_path = scen_dir / "convert.log"
        if log_path.exists():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                log_tail = "\n".join(lines[-200:])
            except Exception:
                pass
        listing = [str(p.relative_to(scen_dir)) for p in sorted(scen_dir.rglob("*"))]
        raise RuntimeError(
            "No MPS/LP produced by CONVERT (including numeric-file fallback).\n"
            f"Scenario dir listing ({scen_dir}): {listing}\n"
            + ("--- convert.log (last 200 lines) ---\n" + log_tail if log_tail else "")
        )

    # Copy to <stub>.mps (even if source has no extension)
    target_mps = outdir / f"{stub}.mps"
    with chosen.open("r", encoding="utf-8", errors="ignore") as fin, \
         target_mps.open("w", encoding="utf-8") as fout:
        for line in fin:
            if line.strip():
                fout.write(line)

    return target_mps, dict_txt


def _patch_yield_block(gms_text: str, wheat: float, corn: float, beets: float) -> str:
    pattern = re.compile(r"(yield\s*\(\s*crop\s*\)[^/]*?/)([\s\S]*?)(/)", re.IGNORECASE)
    records = (
        f" wheat {wheat}\n"
        f"                                                     corn {corn}\n"
        f"                                                     sugarbeets {beets}   "
    )
    def repl(m):
        return m.group(1) + records + m.group(3)
    new_text, nsub = pattern.subn(repl, gms_text, count=1)
    if nsub != 1:
        raise RuntimeError("Could not locate/replace yield(crop) record list in the .gms file.")
    return new_text


def _make_scenario_gms(base_gms: Path, dest_gms: Path, scennum: int):
    """
    Build a per-scenario .gms:
      - Patch yields
      - Embed convert.opt via $onecho
      - Inject <model>.optfile = 1; right before the first 'solve'
      - Prepend 'option lp=convert;'
    """
    if scennum == 0:
        data = (2.0, 2.4, 16.0)
    elif scennum == 1:
        data = (2.5, 3.0, 20.0)
    else:
        data = (3.0, 3.6, 24.0)

    original = base_gms.read_text(encoding="utf-8")
    patched = _patch_yield_block(original, *data)

    model_name = _detect_model_name(patched) or "simple"

    # Embed convert.opt *and* set optfile=1 right before SOLVE
    embedded_opt = (
        "$onecho > convert.opt\n"
        "KeepNames 1\n"
        "MPS 1\n"
        "FixedMPS 1\n"
        "MPSName fixed.mps\n"
        "Dict dict.txt\n"
        "GamsDict dict.txt\n"
        "$offecho\n"
        f"{model_name}.optfile = 1;\n"
    )
    patched = _inject_before_solve(patched, embedded_opt)

    # Use CONVERT as the solver; optfile is set per-model below
    header = "option lp=convert;\n"
    dest_gms.write_text(header + patched, encoding="utf-8")


# ---------------- Main driver ----------------
def main():
    num_scens = 3

    cfg = config.Config()
    cfg.add_to_config(
        "gms_file",
        description="Path to the source GAMS model file (e.g., farmer_average.gms)",
        domain=str,
        default=None,
        argparse_args={"required": True},
    )
    cfg.add_to_config(
        "output_directory",
        description="Directory where scenario files will be written",
        domain=str,
        default=None,
        argparse_args={"required": True},
    )
    cfg.add_to_config(
        "nonant_prefix",
        description="Prefix of nonant variables (default area())",
        domain=str,
        default="area(",
    )
    cfg.add_to_config(
        "gams_bin",
        description="Name/path of GAMS executable",
        domain=str,
        default=None,
    )
    cfg.parse_command_line("farmer_gams_writer_from_gms.py")

    gms_file = Path(cfg.gms_file).resolve()
    if not gms_file.exists():
        raise RuntimeError(f"GAMS file not found: {gms_file}")

    dirname = cfg.output_directory
    if not check_empty_dir(dirname):
        raise RuntimeError(f"{dirname} must exist and be empty")
    outdir = Path(dirname)

    gams_bin = _resolve_gams_bin(cfg.gams_bin)

    default_rho = 1.0
    NONANT_PREFIXES = (cfg.nonant_prefix, cfg.nonant_prefix.replace("(", "["))

    for s in range(num_scens):
        scenario_name = f"scen{s}"
        print(f"preparing scenario {s}")

        scen_dir = outdir / f"{scenario_name}_files"
        scen_dir.mkdir(parents=True, exist_ok=True)
        scen_gms = scen_dir / f"{scenario_name}.gms"
        _make_scenario_gms(gms_file, scen_gms, s)

        mps_path, dict_path = _convert_write_mps_and_dict_from_gms(
            gams_bin=gams_bin, src_gms=scen_gms, outdir=outdir, stub=scenario_name
        )
        print(f"  wrote {mps_path} and {dict_path}")

        nonant_cols, _ = _parse_gams_dict_for_nonants(dict_path, NONANT_PREFIXES)

        scenProb = 1.0 / num_scens
        data = {
            "scenarioData": {"name": scenario_name, "scenProb": scenProb},
            "treeData": {
                "globalNodeCount": 1,
                "nodes": {
                    "ROOT": {
                        "serialNumber": 0,
                        "condProb": 1.0,
                        "nonAnts": nonant_cols,
                    }
                },
            },
        }
        (outdir / f"{scenario_name}_nonants.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  wrote {outdir / f'{scenario_name}_nonants.json'}")

        rho_path = outdir / f"{scenario_name}_rho.csv"
        with rho_path.open("w", encoding="utf-8") as csvf:
            csvf.write("varname,rho\n")
            for name in nonant_cols:
                csvf.write(f"{name},{default_rho}\n")
        print(f"  wrote {rho_path}")


if __name__ == "__main__":
    main()
