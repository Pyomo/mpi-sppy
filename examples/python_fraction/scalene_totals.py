###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
scalene_totals.py

Extract Python / native / system time totals from a Scalene JSON profile.

Scalene's JSON gives, for every source line it attributed samples to, the
percentage of the run's wall time spent on that line in Python, in native
(compiled) code, and in the system.  Summing those per-line percentages over
every line of every file yields the whole-run totals, which is exactly what
this module does.

Reading the JSON is preferred over parsing `scalene view --cli` output:

  * the JSON carries full precision, whereas the CLI rounds each line to a
    whole percent,
  * `scalene view --reduced` omits low-usage lines, so summing its rows
    undercounts,
  * the CLI emits ANSI colour codes and rearranges its table between Scalene
    releases, which is what made the original version of this code fragile.

The result is self-checking: the sum of the per-line percentages is compared
against the sum of Scalene's own per-file ``percent_cpu_time`` values.  The two
agree exactly for most profiles and differ by at most a couple of tenths of a
percentage point for the rest, because a few samples are attributed to a file
without landing on one of the lines the file reports.  ``consistency_error``
therefore allows a small absolute slack; it exists to catch a change in
Scalene's JSON layout, which would show up as a large disagreement, not to
police that last tenth of a point.

Note that the per-file ``functions`` lists are *not* a usable substitute for the
per-line data: summing them overshoots ``percent_cpu_time`` by as much as 30
percentage points on these profiles, evidently because nested and wrapped
functions get counted more than once.

The three buckets together should account for close to 100% of wall time.
Whatever is missing from 100% is samples Scalene could not attribute at all; it
is reported as ``accounted_pct`` so a reader can judge how much is unexplained.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Totals:
    """Whole-run totals for one Scalene profile (one MPI rank)."""

    wall_sec: Optional[float]
    python_pct: float  # percent of wall time, Python (interpreted) code
    native_pct: float  # percent of wall time, compiled code (solver, numpy, MPI)
    system_pct: float  # percent of wall time, system/kernel
    argv: Optional[List[str]]

    @property
    def accounted_pct(self) -> float:
        """Percent of wall time Scalene attributed to some source line."""
        return self.python_pct + self.native_pct + self.system_pct

    @property
    def python_fraction(self) -> Optional[float]:
        """Python as a percent of attributed time; None if nothing attributed.

        This is the headline number: it divides out the unattributed residual
        so that the three buckets sum to 100%.
        """
        if self.accounted_pct <= 0.0:
            return None
        return 100.0 * self.python_pct / self.accounted_pct

    def seconds(self, pct: float) -> Optional[float]:
        if self.wall_sec is None:
            return None
        return self.wall_sec * pct / 100.0

    @property
    def python_sec(self) -> Optional[float]:
        return self.seconds(self.python_pct)

    @property
    def native_sec(self) -> Optional[float]:
        return self.seconds(self.native_pct)

    @property
    def system_sec(self) -> Optional[float]:
        return self.seconds(self.system_pct)


def totals_from_json_obj(j: Dict[str, Any]) -> Totals:
    py = nat = sys_ = 0.0
    for finfo in (j.get("files") or {}).values():
        for line in finfo.get("lines") or ():
            py += line.get("n_cpu_percent_python", 0.0) or 0.0
            nat += line.get("n_cpu_percent_c", 0.0) or 0.0
            sys_ += line.get("n_sys_percent", 0.0) or 0.0

    wall = j.get("elapsed_time_sec")
    try:
        wall = float(wall) if wall is not None else None
    except (TypeError, ValueError):
        wall = None

    argv = j.get("args")
    if isinstance(argv, str):
        argv = argv.split()
    elif not isinstance(argv, list):
        argv = None
    else:
        argv = [str(x) for x in argv]

    return Totals(wall_sec=wall, python_pct=py, native_pct=nat,
                  system_pct=sys_, argv=argv)


def totals_from_json(path: str) -> Totals:
    with open(path, "r", encoding="utf-8") as f:
        return totals_from_json_obj(json.load(f))


def consistency_error(path: str, tol_pct_points: float = 1.0) -> Optional[str]:
    """Cross-check the per-line sum against Scalene's own per-file totals.

    Returns None when they agree to within ``tol_pct_points`` percentage points,
    otherwise a message describing the mismatch.  The tolerance is deliberately
    loose: a real layout change misses by tens of points, while normal profiles
    agree exactly or to within a few tenths (see the module docstring).
    """
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    t = totals_from_json_obj(j)
    per_file = sum(
        (finfo.get("percent_cpu_time") or 0.0)
        for finfo in (j.get("files") or {}).values()
    )
    if abs(per_file - t.accounted_pct) > tol_pct_points:
        return (f"{path}: per-line sum {t.accounted_pct:.6f}% disagrees with sum of "
                f"per-file percent_cpu_time {per_file:.6f}% by more than "
                f"{tol_pct_points} percentage points")
    return None
