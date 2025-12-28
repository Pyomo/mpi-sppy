#!/usr/bin/env python3
"""
make_scalene_latex_table.py

Generate a LaTeX summary table from Scalene MPI rank profiles.

Inputs:
  scalene_rank_0.json, scalene_rank_1.json, ...

Outputs:
  A LaTeX file containing:
    - System information (CPU + memory) collected from standard Unix tools/files
    - Run parameters extracted from JSON argv (best-effort)
    - A per-rank table with:
        Wall (s): elapsed_time_sec from JSON
        Python (s), Native (s), System (s): derived by parsing `scalene view --cli --reduced`
          and summing per-line percentage columns (Time Python / native / system), then
          multiplying by wall time.

Totals row:
  - Job wall time = max wall time across ranks
  - Sum(Python seconds), Sum(Native seconds), Sum(System seconds) across ranks (if available)

Why this approach:
  With Scalene 2.0.1, `scalene run` emits JSON profiles whose CPU summary fields may be null.
  The `scalene view --cli` output contains per-line "% of time" columns; summing those
  percentages yields overall percent-of-wall-time totals.

Usage:
  python make_scalene_latex_table.py --glob "scalene_rank_*.json" --out scalene_summary.tex

Options:
  --cache-cli     Cache CLI output as <json>.cli.txt (reused if present)
  --no-view       Don't run scalene view (wall time only)
  --reduced       Pass --reduced to scalene view (recommended)
  --columns N     Set terminal width (COLUMNS) for scalene view output (default 200)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RankRow:
    filename: str
    rank: Optional[int]
    wall_time: Optional[float]
    python_time: Optional[float]
    native_time: Optional[float]
    system_time: Optional[float]


# ---------------------------
# JSON + misc helpers
# ---------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _parse_rank_from_filename(fname: str) -> Optional[int]:
    m = re.search(r"rank[_\-]?(\d+)", os.path.basename(fname))
    if m:
        return int(m.group(1))
    return None


def _latex_escape(s: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def _get(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _first_present(d: Dict[str, Any], candidate_paths: List[List[str]]) -> Any:
    for p in candidate_paths:
        v = _get(d, p)
        if v is not None:
            return v
    return None


# ---------------------------
# CLI arg extraction (from JSON)
# ---------------------------

def _extract_argv(run: Dict[str, Any]) -> Optional[List[str]]:
    candidates = [
        ["argv"],
        ["commandline"],
        ["command_line"],
        ["cmdline"],
        ["cmd_line"],
        ["command"],
        ["args"],
        ["metadata", "argv"],
        ["metadata", "commandline"],
        ["metadata", "command_line"],
        ["meta", "argv"],
        ["meta", "commandline"],
        ["meta", "command_line"],
        ["header", "argv"],
        ["header", "commandline"],
        ["header", "command_line"],
    ]
    val = _first_present(run, candidates)
    if isinstance(val, list) and all(isinstance(x, (str, int, float)) for x in val):
        return [str(x) for x in val]
    if isinstance(val, str):
        return val.split()
    return None


def _extract_params_from_argv(argv: List[str]) -> Dict[str, str]:
    params: Dict[str, str] = {}

    def take_value(i: int) -> Optional[str]:
        return argv[i + 1] if i + 1 < len(argv) else None

    target = next((tok for tok in argv if isinstance(tok, str) and tok.endswith(".py")), None)
    if target:
        params["target"] = target

    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in (
            "--module-name",
            "--num-scens",
            "--solver-name",
            "--max-iterations",
            "--max-solver-threads",
            "--default-rho",
            "--rel-gap",
            "--outfile",
        ):
            v = take_value(i)
            if v is not None:
                params[tok.lstrip("-")] = v
                i += 2
                continue
        if tok in ("--lagrangian", "--xhatshuffle"):
            params[tok.lstrip("-")] = "true"
            i += 1
            continue
        i += 1

    return params


# ---------------------------
# System info collection (Unix/bash assumptions)
# ---------------------------

def _run_cmd(cmd: List[str]) -> Optional[str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
        out = p.stdout.strip()
        return out if out else None
    except Exception:
        return None


def _read_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            continue
    return None


def _parse_meminfo_kib(meminfo_text: str) -> Dict[str, int]:
    """
    Return selected values from /proc/meminfo in KiB
    """
    out: Dict[str, int] = {}
    for line in meminfo_text.splitlines():
        m = re.match(r"^(\w+):\s+(\d+)\s+kB\s*$", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def _format_bytes(n: Optional[int]) -> str:
    if n is None:
        return "unknown"
    # n is bytes
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    i = 0
    while x >= 1024.0 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    if i == 0:
        return f"{int(x)} {units[i]}"
    return f"{x:.2f} {units[i]}"


def _collect_system_info() -> Dict[str, str]:
    """
    Best-effort system inventory on Unix:
      - OS/kernel
      - CPU count (logical + physical if available)
      - CPU model
      - CPU max MHz (or base)
      - Total memory and (approx) available memory at report time

    Works on Linux best; degrades gracefully on macOS/others.
    """
    info: Dict[str, str] = {}

    # OS / kernel
    info["os"] = f"{platform.system()} {platform.release()} ({platform.machine()})"

    # CPU counts
    logical = os.cpu_count()
    info["cpu_logical"] = str(logical) if logical is not None else "unknown"

    # lscpu (Linux)
    lscpu = _run_cmd(["bash", "-lc", "lscpu"])
    cpu_model = None
    cpu_mhz = None
    cpu_max_mhz = None
    cpu_sockets = None
    cores_per_socket = None
    threads_per_core = None

    if lscpu:
        for line in lscpu.splitlines():
            if ":" not in line:
                continue
            k, v = [x.strip() for x in line.split(":", 1)]
            kl = k.lower()
            if kl == "model name":
                cpu_model = v
            elif kl in ("cpu mhz",):
                cpu_mhz = v
            elif kl in ("cpu max mhz",):
                cpu_max_mhz = v
            elif kl == "socket(s)":
                cpu_sockets = v
            elif kl == "core(s) per socket":
                cores_per_socket = v
            elif kl == "thread(s) per core":
                threads_per_core = v

    # sysctl (macOS / BSD)
    if cpu_model is None:
        cpu_model = _run_cmd(["bash", "-lc", "sysctl -n machdep.cpu.brand_string 2>/dev/null"]) or None

    # Physical cores (best-effort)
    physical_cores = None
    if cpu_sockets and cores_per_socket:
        try:
            physical_cores = int(cpu_sockets) * int(cores_per_socket)
        except Exception:
            physical_cores = None
    if physical_cores is None:
        # macOS
        pc = _run_cmd(["bash", "-lc", "sysctl -n hw.physicalcpu 2>/dev/null"])
        if pc and pc.isdigit():
            physical_cores = int(pc)

    if physical_cores is not None:
        info["cpu_physical_cores"] = str(physical_cores)

    if cpu_model:
        info["cpu_model"] = cpu_model

    # Frequency
    # Prefer max MHz if available
    freq = cpu_max_mhz or cpu_mhz
    if freq:
        info["cpu_mhz"] = freq

    if threads_per_core:
        info["threads_per_core"] = threads_per_core
    if cpu_sockets:
        info["cpu_sockets"] = cpu_sockets
    if cores_per_socket:
        info["cores_per_socket"] = cores_per_socket

    # Memory: Linux /proc/meminfo or macOS sysctl/vm_stat
    meminfo = _read_first_existing(["/proc/meminfo"])
    if meminfo:
        m = _parse_meminfo_kib(meminfo)
        mem_total_bytes = m.get("MemTotal", 0) * 1024 if "MemTotal" in m else None
        mem_avail_bytes = m.get("MemAvailable", 0) * 1024 if "MemAvailable" in m else None
        info["mem_total"] = _format_bytes(mem_total_bytes)
        info["mem_available"] = _format_bytes(mem_avail_bytes)
    else:
        # macOS total
        mt = _run_cmd(["bash", "-lc", "sysctl -n hw.memsize 2>/dev/null"])
        if mt and mt.isdigit():
            info["mem_total"] = _format_bytes(int(mt))
        # macOS available is trickier; best-effort via vm_stat
        vm = _run_cmd(["bash", "-lc", "vm_stat 2>/dev/null"])
        if vm:
            # Parse page size and free/inactive/speculative, etc.
            page_size = 4096
            mps = re.search(r"page size of (\d+) bytes", vm)
            if mps:
                page_size = int(mps.group(1))
            counts = {}
            for line in vm.splitlines():
                mm = re.match(r"^([^:]+):\s+(\d+)\.", line.strip())
                if mm:
                    counts[mm.group(1).strip()] = int(mm.group(2))
            # rough estimate: free + inactive + speculative
            avail_pages = (
                counts.get("Pages free", 0)
                + counts.get("Pages inactive", 0)
                + counts.get("Pages speculative", 0)
            )
            info["mem_available"] = _format_bytes(avail_pages * page_size)

    return info


# ---------------------------
# Scalene view parsing
# ---------------------------

def _run_scalene_view_cli(json_path: str, reduced: bool, columns: int) -> str:
    cmd = ["python", "-m", "scalene", "view", "--cli"]
    if reduced:
        cmd.append("--reduced")
    cmd.append(json_path)

    env = dict(os.environ)
    env["COLUMNS"] = str(columns)

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    return p.stdout


def _parse_cli_percent_totals(cli_text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Parse totals by summing per-line percent columns from Scalene `view --cli` output.

    We match table rows like:
       15 │    2% │   24% │   2%  │ ...

    Returns: (python_percent, native_percent, system_percent)
    """
    py_pct = 0.0
    nat_pct = 0.0
    sys_pct = 0.0
    saw_any = False

    row_re = re.compile(
        r"^\s*\d+\s*│\s*([0-9]+(?:\.[0-9]+)?)?\s*%?\s*│\s*([0-9]+(?:\.[0-9]+)?)?\s*%?\s*│\s*([0-9]+(?:\.[0-9]+)?)?\s*%?\s*│"
    )

    for line in cli_text.splitlines():
        m = row_re.match(line)
        if not m:
            continue
        saw_any = True
        a, b, c = m.group(1), m.group(2), m.group(3)
        py_pct += float(a) if a else 0.0
        nat_pct += float(b) if b else 0.0
        sys_pct += float(c) if c else 0.0

    if not saw_any:
        return None, None, None

    return py_pct, nat_pct, sys_pct


def _fmt(x: Optional[float], digits: int = 2, na: str = r"\textemdash") -> str:
    if x is None:
        return na
    return f"{x:.{digits}f}"


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="scalene_rank_*.json", help="Glob for Scalene JSON files")
    ap.add_argument("--out", default="scalene_summary.tex", help="Output LaTeX filename")
    ap.add_argument("--caption", default="Scalene timing summary by MPI rank", help="Table caption")
    ap.add_argument("--label", default="tab:scalene-summary", help="LaTeX label")
    ap.add_argument("--no-totals", action="store_true", help="Do not add totals row")
    ap.add_argument("--no-view", action="store_true", help="Do not run `scalene view --cli` (wall only)")
    ap.add_argument("--cache-cli", action="store_true", help="Cache CLI output to <json>.cli.txt and reuse")
    ap.add_argument("--reduced", action="store_true", help="Pass --reduced to scalene view --cli (recommended)")
    ap.add_argument("--columns", type=int, default=200, help="Set COLUMNS for scalene view output (default 200)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched glob: {args.glob}")

    # System info
    sysinfo = _collect_system_info()

    # Load JSON profiles
    runs = [(f, _load_json(f)) for f in files]

    # Use the first JSON file as the "run metadata" source
    first_json = runs[0][1]
    argv = _extract_argv(first_json)
    params: Dict[str, str] = _extract_params_from_argv(argv) if argv else {}

    rows: List[RankRow] = []
    for f, j in runs:
        wall = _coerce_float(j.get("elapsed_time_sec"))

        python_s = native_s = system_s = None

        if not args.no_view and wall is not None:
            cache_path = f"{f}.cli.txt"
            if args.cache_cli and os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as cf:
                    cli_text = cf.read()
            else:
                cli_text = _run_scalene_view_cli(f, reduced=args.reduced, columns=args.columns)
                if args.cache_cli:
                    with open(cache_path, "w", encoding="utf-8") as cf:
                        cf.write(cli_text)

            py_pct, nat_pct, sys_pct = _parse_cli_percent_totals(cli_text)
            if py_pct is not None and nat_pct is not None and sys_pct is not None:
                python_s = wall * (py_pct / 100.0)
                native_s = wall * (nat_pct / 100.0)
                system_s = wall * (sys_pct / 100.0)

        rows.append(
            RankRow(
                filename=os.path.basename(f),
                rank=_parse_rank_from_filename(f),
                wall_time=wall,
                python_time=python_s,
                native_time=native_s,
                system_time=system_s,
            )
        )

    rows.sort(key=lambda r: (999999 if r.rank is None else r.rank, r.filename))

    vals_wall = [r.wall_time for r in rows if r.wall_time is not None]
    vals_py = [r.python_time for r in rows if r.python_time is not None]
    vals_nat = [r.native_time for r in rows if r.native_time is not None]
    vals_sys = [r.system_time for r in rows if r.system_time is not None]

    job_wall = max(vals_wall) if vals_wall else None
    sum_py = sum(vals_py) if vals_py else None
    sum_nat = sum(vals_nat) if vals_nat else None
    sum_sys = sum(vals_sys) if vals_sys else None

    any_time_breakdown = bool(vals_py or vals_nat or vals_sys)

    # Build LaTeX
    lines: List[str] = []
    lines.append("% Auto-generated by make_scalene_latex_table.py")
    lines.append("")

    # System info block
    lines.append(r"\noindent\textbf{System information:}\\")
    lines.append(r"\begin{itemize}")
    if "os" in sysinfo:
        lines.append(rf"  \item \texttt{{os}}={{{_latex_escape(sysinfo['os'])}}}")
    if "cpu_model" in sysinfo:
        lines.append(rf"  \item \texttt{{cpu\_model}}={{{_latex_escape(sysinfo['cpu_model'])}}}")
    if "cpu_logical" in sysinfo:
        lines.append(rf"  \item \texttt{{cpu\_logical}}={{{_latex_escape(sysinfo['cpu_logical'])}}}")
    if "cpu_physical_cores" in sysinfo:
        lines.append(rf"  \item \texttt{{cpu\_physical\_cores}}={{{_latex_escape(sysinfo['cpu_physical_cores'])}}}")
    if "cpu_mhz" in sysinfo:
        lines.append(rf"  \item \texttt{{cpu\_mhz}}={{{_latex_escape(sysinfo['cpu_mhz'])}}}")
    if "cpu_sockets" in sysinfo:
        lines.append(rf"  \item \texttt{{cpu\_sockets}}={{{_latex_escape(sysinfo['cpu_sockets'])}}}")
    if "cores_per_socket" in sysinfo:
        lines.append(rf"  \item \texttt{{cores\_per\_socket}}={{{_latex_escape(sysinfo['cores_per_socket'])}}}")
    if "threads_per_core" in sysinfo:
        lines.append(rf"  \item \texttt{{threads\_per\_core}}={{{_latex_escape(sysinfo['threads_per_core'])}}}")
    if "mem_total" in sysinfo:
        lines.append(rf"  \item \texttt{{mem\_total}}={{{_latex_escape(sysinfo['mem_total'])}}}")
    if "mem_available" in sysinfo:
        lines.append(rf"  \item \texttt{{mem\_available}}={{{_latex_escape(sysinfo['mem_available'])}}}")
    lines.append(r"\end{itemize}")
    lines.append("")

    # Run parameters comment header
    lines.append("% Run parameters extracted from JSON (best-effort):")
    if argv:
        lines.append(f"% argv: {_latex_escape(' '.join(argv))}")
    else:
        lines.append("% argv: (not found in JSON)")
    if params:
        lines.append("% Parsed parameters:")
        for k in sorted(params.keys()):
            lines.append(f"%   {k}: {_latex_escape(params[k])}")
    lines.append("")

    # Run parameters block
    lines.append(r"\noindent\textbf{Run parameters (from Scalene JSON):}\\")
    if params:
        show_keys = [
            "target",
            "module-name",
            "num-scens",
            "solver-name",
            "max-iterations",
            "max-solver-threads",
            "default-rho",
            "rel-gap",
            "lagrangian",
            "xhatshuffle",
        ]
        parts = []
        for k in show_keys:
            if k in params:
                parts.append(rf"\texttt{{{_latex_escape(k)}}}={{{_latex_escape(params[k])}}}")
        lines.append(r"\begin{itemize}")
        for p in parts:
            lines.append(rf"  \item {p}")
        lines.append(r"\end{itemize}")
    else:
        lines.append(r"\emph{(Command line not found in JSON.)}\\")
    lines.append("")

    if args.no_view:
        lines.append(
            r"\noindent\emph{Note: Time breakdown requires parsing \texttt{python -m scalene view --cli --reduced <profile.json>}.}"
        )
        lines.append("")
    elif not any_time_breakdown:
        lines.append(
            r"\noindent\emph{Note: No per-line time percentages were found in the output of \texttt{scalene view --cli}.}"
        )
        lines.append("")

    # Table
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{r l r r r r}")
    lines.append(r"\hline")
    lines.append(r"Rank & File & Wall (s) & Python (s) & Native (s) & System (s) \\")
    lines.append(r"\hline")

    for r in rows:
        rank_str = "" if r.rank is None else str(r.rank)
        lines.append(
            rf"{rank_str} & {_latex_escape(r.filename)} & {_fmt(r.wall_time)} & {_fmt(r.python_time)} & {_fmt(r.native_time)} & {_fmt(r.system_time)} \\"
        )

    if not args.no_totals:
        lines.append(r"\hline")
        lines.append(
            rf"\textbf{{Job wall (max)}} &  & \textbf{{{_fmt(job_wall)}}} & \textbf{{{_fmt(sum_py)}}} & \textbf{{{_fmt(sum_nat)}}} & \textbf{{{_fmt(sum_sys)}}} \\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{_latex_escape(args.caption)}}}")
    lines.append(rf"\label{{{_latex_escape(args.label)}}}")
    lines.append(r"\end{table}")
    lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote LaTeX to: {args.out}")
    print(f"Read {len(files)} JSON files matched by: {args.glob}")


if __name__ == "__main__":
    main()
