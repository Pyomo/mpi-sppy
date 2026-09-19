#!/usr/bin/env python3
###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
summarize_reps.py

Aggregate the repeated runs produced by run_experiments.bash into LaTeX tables.

Scalene estimates the Python/native/system split by sampling, so one run of one
case is not evidence of much. This script reads every repetition of every case
and reports the spread across repetitions, which is what makes it possible to
say whether a difference between cases is real or just sampling noise.

Expected layout (as written by run_experiments.bash):

    <results>/<solver>/<case>/rep<n>/scalene_rank_<rank>.json

and, for runs made with PROFILE=0,

    <results>/<solver>-unprofiled/<case>/rep<n>/wall.txt

Two tables are produced:

  Summary   one row per case: wall time and the job-level Python/native/system
            split, averaged over repetitions, with the observed min--max range
            of the Python percentage. When unprofiled wall times are available,
            a column reports scalene's overhead, since the profiler's own cost
            falls mostly on Python and therefore inflates the Python share.

  Per-rank  one row per case and one column per rank: the Python percentage for
            that cylinder, averaged over repetitions with its range. This is the
            table that shows whether a particular cylinder is an outlier.

The job-level percentage for one repetition is computed from summed seconds
across ranks, not by averaging per-rank percentages, so that ranks are weighted
by how long they actually ran.

Usage:
  python3 summarize_reps.py --results results --out scalene_summary.tex
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from scalene_totals import consistency_error, totals_from_json


@dataclass
class RepResult:
    """One repetition of one case, aggregated over its ranks."""

    rep: str
    wall_sec: float               # max over ranks: the job's wall time
    python_pct: float             # job-level, percent of attributed time
    native_pct: float
    system_pct: float
    accounted_pct: float          # mean over ranks, for reporting
    per_rank_python_pct: Dict[int, float]


def _rank_of(path: str) -> Optional[int]:
    m = re.search(r"rank[_\-]?(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


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


def load_rep(rep_dir: str) -> Optional[RepResult]:
    files = sorted(glob.glob(os.path.join(rep_dir, "scalene_rank_*.json")))
    if not files:
        return None

    walls: List[float] = []
    accounted: List[float] = []
    sum_py = sum_nat = sum_sys = 0.0
    per_rank: Dict[int, float] = {}

    for f in files:
        bad = consistency_error(f)
        if bad:
            raise SystemExit(
                "Scalene JSON failed its internal consistency check, so its "
                "layout has probably changed:\n  " + bad
            )
        t = totals_from_json(f)
        if t.wall_sec is None:
            continue
        walls.append(t.wall_sec)
        accounted.append(t.accounted_pct)
        sum_py += t.python_sec or 0.0
        sum_nat += t.native_sec or 0.0
        sum_sys += t.system_sec or 0.0
        r = _rank_of(f)
        if r is not None and t.python_fraction is not None:
            per_rank[r] = t.python_fraction

    if not walls:
        return None

    denom = sum_py + sum_nat + sum_sys
    if denom <= 0.0:
        return None

    return RepResult(
        rep=os.path.basename(rep_dir),
        wall_sec=max(walls),
        python_pct=100.0 * sum_py / denom,
        native_pct=100.0 * sum_nat / denom,
        system_pct=100.0 * sum_sys / denom,
        accounted_pct=_mean(accounted),
        per_rank_python_pct=per_rank,
    )


def load_case(case_dir: str) -> List[RepResult]:
    reps = []
    for rep_dir in sorted(glob.glob(os.path.join(case_dir, "rep*"))):
        if not os.path.isdir(rep_dir):
            continue
        r = load_rep(rep_dir)
        if r is not None:
            reps.append(r)
    return reps


def load_unprofiled_walls(case_dir: str) -> List[float]:
    """Wall times from PROFILE=0 runs of one case, if any were made."""
    walls = []
    for wf in sorted(glob.glob(os.path.join(case_dir, "rep*", "wall.txt"))):
        try:
            with open(wf, "r", encoding="utf-8") as f:
                walls.append(float(f.read().strip()))
        except (OSError, ValueError):
            continue
    return walls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="Results root directory")
    ap.add_argument("--out", default="scalene_summary.tex", help="Output LaTeX filename")
    ap.add_argument("--solvers", nargs="*", default=None,
                    help="Solver subdirectories to report, in order "
                         "(default: all, alphabetical)")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Cases in the order to report (default: all, alphabetical)")
    ap.add_argument("--rank-labels", default="",
                    help="Comma-separated cylinder names for ranks 0,1,2,... "
                         "(e.g. 'PH hub,lagrangian,xhatshuffle')")
    args = ap.parse_args()

    if args.solvers:
        solvers = args.solvers
    else:
        solvers = sorted(
            d for d in os.listdir(args.results)
            if os.path.isdir(os.path.join(args.results, d))
            and not d.endswith("-unprofiled")
        )

    labels = [s.strip() for s in args.rank_labels.split(",") if s.strip()]

    def rank_label(r: int) -> str:
        return labels[r] if r < len(labels) else f"rank {r}"

    # loaded holds (solver, case, reps, unprofiled_walls)
    loaded = []
    for sv in solvers:
        sv_dir = os.path.join(args.results, sv)
        if args.cases:
            cases = args.cases
        else:
            cases = sorted(
                d for d in os.listdir(sv_dir)
                if os.path.isdir(os.path.join(sv_dir, d))
            )
        for c in cases:
            case_dir = os.path.join(sv_dir, c)
            if not os.path.isdir(case_dir):
                continue
            reps = load_case(case_dir)
            if not reps:
                print(f"warning: no usable repetitions for {sv}/{c}")
                continue
            bare = load_unprofiled_walls(
                os.path.join(args.results, f"{sv}-unprofiled", c)
            )
            loaded.append((sv, c, reps, bare))

    if not loaded:
        raise SystemExit(f"No usable results under {args.results}")

    multi_solver = len({sv for sv, _, _, _ in loaded}) > 1
    any_bare = any(bare for _, _, _, bare in loaded)

    out: List[str] = []
    out.append("% Auto-generated by summarize_reps.py -- do not edit by hand")
    out.append("")

    # ---- Summary table: one row per (solver, case) ----
    acct_all = [r.accounted_pct for _, _, reps, _ in loaded for r in reps]
    lead_cols = "l l" if multi_solver else "l"
    lead_head = "Solver & Case" if multi_solver else "Case"

    def lead_cells(sv: str, c: str) -> str:
        return (f"{_latex_escape(sv)} & {_latex_escape(c)}" if multi_solver
                else _latex_escape(c))

    out.append(r"\begin{table}[ht]")
    out.append(r"\centering")
    out.append(rf"\begin{{tabular}}{{{lead_cols} r r r r r{' r' if any_bare else ''}}}")
    out.append(r"\hline")
    out.append(lead_head + r" & Reps & Wall (s) & Python (\%) & Native (\%) & System (\%)"
               + (r" & Overhead" if any_bare else "") + r" \\")
    out.append(r"\hline")
    for sv, c, reps, bare in loaded:
        pys = [r.python_pct for r in reps]
        walls = [r.wall_sec for r in reps]
        row = (
            f"{lead_cells(sv, c)} & {len(reps)} & {_mean(walls):.1f} & "
            f"{_mean(pys):.1f} ({min(pys):.1f}--{max(pys):.1f}) & "
            f"{_mean([r.native_pct for r in reps]):.1f} & "
            f"{_mean([r.system_pct for r in reps]):.1f}"
        )
        if any_bare:
            row += (rf" & {_mean(walls) / _mean(bare):.2f}$\times$" if bare
                    else r" & \textemdash")
        out.append(row + r" \\")
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    caption = (
        r"\caption{Time split by case, averaged over repetitions, with the "
        r"min--max range over repetitions shown for the Python percentage. "
        r"Percentages are of the time Scalene attributed to a source line "
        rf"({min(acct_all):.1f}--{max(acct_all):.1f}\% of wall time here); "
        r"wall time is the maximum over ranks."
    )
    if any_bare:
        caption += (
            r" The overhead column is profiled wall time divided by unprofiled "
            r"wall time for the same case; because scalene's instrumentation cost "
            r"lands mostly on Python, it inflates the Python column."
        )
    out.append(caption + r"}")
    out.append(r"\label{tab:python-fraction-summary}")
    out.append(r"\end{table}")
    out.append("")

    # ---- Per-rank table: one row per (solver, case) ----
    all_ranks = sorted({r for _, _, reps, _ in loaded
                        for rep in reps for r in rep.per_rank_python_pct})
    out.append(r"\begin{table}[ht]")
    out.append(r"\centering")
    out.append(rf"\begin{{tabular}}{{{lead_cols}" + " r" * len(all_ranks) + r"}")
    out.append(r"\hline")
    out.append(lead_head + " & "
               + " & ".join(_latex_escape(rank_label(r)) for r in all_ranks) + r" \\")
    out.append(r"\hline")
    for sv, c, reps, _bare in loaded:
        cells = []
        for r in all_ranks:
            vals = [rep.per_rank_python_pct[r] for rep in reps if r in rep.per_rank_python_pct]
            cells.append(f"{_mean(vals):.1f} ({min(vals):.1f}--{max(vals):.1f})" if vals
                         else r"\textemdash")
        out.append(f"{lead_cells(sv, c)} & " + " & ".join(cells) + r" \\")
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    out.append(
        r"\caption{Python percentage by cylinder: mean over repetitions with the "
        r"min--max range in parentheses.}"
    )
    out.append(r"\label{tab:python-fraction-by-rank}")
    out.append(r"\end{table}")
    out.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"Wrote LaTeX to: {args.out}")
    for sv, c, reps, bare in loaded:
        pys = [r.python_pct for r in reps]
        walls = [r.wall_sec for r in reps]
        extra = f", overhead {_mean(walls) / _mean(bare):.2f}x" if bare else ""
        print(f"  {sv}/{c}: {len(reps)} reps, wall {_mean(walls):.1f}s, "
              f"Python {_mean(pys):.1f}% ({min(pys):.1f}-{max(pys):.1f}){extra}")


if __name__ == "__main__":
    main()
