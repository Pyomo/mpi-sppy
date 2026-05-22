###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Summarize LOR_bug diagnostic output from PR #717.

PR #717 instruments mpisppy/spbase.py::SPBase.allreduce_or to print a 4-line
block on every call (from cyl_rk == 0 of the cylinder's mpicomm). This script
parses such a log and writes a short report on the four hypotheses being
tested:

  H1. self.mpicomm has wider membership than the cylinder it should.
  H2. Buffer memory underneath local_val was nonzero / non-boolean.
  H3. The Allreduce reducer path is malfunctioning.
  H4. Duplicate rank participation in self.mpicomm.

Usage:
    python lor_bug_report.py <log_file>
"""

import re
import sys
from collections import defaultdict


HEADER_RE = re.compile(
    r"^\[LOR_bug call=(?P<call>\d+) cls=(?P<cls>\S+) "
    r"world_rk=(?P<world_rk>\d+) host=(?P<host>\S+) pid=(?P<pid>\d+)\] "
    r"mpicomm size=(?P<size>\d+) name=(?P<name>.+)$"
)
WORLD_RANKS_RE = re.compile(
    r"^\s*world_ranks: min=(?P<wr_min>\d+) max=(?P<wr_max>\d+) "
    r"count=(?P<count>\d+) unique=(?P<unique>\d+)$"
)
REDUCTIONS_RE = re.compile(
    r"^\s*reductions: sum=(?P<sum>-?\d+) max=(?P<max>-?\d+) "
    r"lor=(?P<lor>-?\d+) rank_sum=(?P<rank_sum>-?\d+) "
    r"expected_rank_sum=(?P<expected_rank_sum>-?\d+)$"
)
GATHER_RE = re.compile(
    r"^\s*gather: gather_sum=(?P<gather_sum>-?\d+) "
    r"nonzero_reports=(?P<nonzero_reports>\d+)$"
)


def parse(path):
    """Return a list of dicts, one per [LOR_bug ...] block."""
    with open(path) as f:
        lines = f.readlines()

    entries = []
    i = 0
    n = len(lines)
    while i < n:
        m = HEADER_RE.match(lines[i].rstrip())
        if not m:
            i += 1
            continue
        entry = {
            "call": int(m["call"]),
            "cls": m["cls"],
            "world_rk": int(m["world_rk"]),
            "host": m["host"],
            "pid": int(m["pid"]),
            "size": int(m["size"]),
            "name": m["name"],
        }
        i += 1
        # The next three lines should be world_ranks / reductions / gather,
        # in that order. Tolerate missing lines defensively.
        for pat in (WORLD_RANKS_RE, REDUCTIONS_RE, GATHER_RE):
            if i >= n:
                break
            mm = pat.match(lines[i].rstrip())
            if not mm:
                break
            for k, v in mm.groupdict().items():
                entry[k] = int(v)
            i += 1
        entries.append(entry)
    return entries


def _examples(rows, n=5):
    out = []
    for e in rows[:n]:
        out.append(
            f"      cls={e['cls']} call={e['call']} "
            f"world_rk={e['world_rk']} host={e['host']}"
        )
    if len(rows) > n:
        out.append(f"      (... {len(rows) - n} more truncated ...)")
    return "\n".join(out)


def report(entries, path):
    print(f"LOR_bug report for: {path}")
    print(f"Parsed {len(entries)} [LOR_bug ...] blocks.")
    if not entries:
        print("\nNo diagnostic blocks found. Was the run on the LOR_bug branch?")
        return

    # ---------- per-comm summary ----------
    by_comm = defaultdict(list)
    for e in entries:
        by_comm[(e["cls"], e["name"])].append(e)

    print("\nPer-comm summary (one printer per comm; cyl_rk == 0 only):")
    for (cls, name), es in sorted(by_comm.items()):
        sizes = sorted({e["size"] for e in es})
        wrs = sorted({e["world_rk"] for e in es})
        print(f"  cls={cls} name={name}")
        print(f"    calls={len(es)} sizes={sizes} printer_world_rk={wrs}")

    # ---------- H1: wider membership ----------
    # Signal: size varies within a single (cls, name) bucket, OR printer
    # world_rk varies across calls for the same logical comm (meaning
    # different ranks took the "rank 0" role — only possible if comm
    # membership shifted).
    print("\nH1 — wider mpicomm membership than expected:")
    h1_hits = []
    for (cls, name), es in by_comm.items():
        sizes = {e["size"] for e in es}
        printers = {e["world_rk"] for e in es}
        if len(sizes) > 1 or len(printers) > 1:
            h1_hits.append((cls, name, sorted(sizes), sorted(printers)))
    if h1_hits:
        print("  WARNING: comm membership is not stable across calls:")
        for cls, name, sizes, printers in h1_hits:
            print(f"    cls={cls} name={name} sizes={sizes} "
                  f"printer_world_rks={printers}")
    else:
        print("  OK: every comm has a stable size and stable rank-0 printer.")

    # Also: if two different comms share the same printer world rank, that
    # rank straddles two cylinders -- possible cross-cylinder contamination.
    printer_to_comms = defaultdict(set)
    for (cls, name), es in by_comm.items():
        for e in es:
            printer_to_comms[e["world_rk"]].add((cls, name))
    shared = {wr: cs for wr, cs in printer_to_comms.items() if len(cs) > 1}
    if shared:
        print("  NOTE: world ranks acting as printer for multiple comms:")
        for wr, cs in sorted(shared.items()):
            print(f"    world_rk={wr} comms={sorted(cs)}")

    # ---------- H2: buffer aliasing / non-boolean input ----------
    # Signature per PR description: nonzero local_val where it should be 0.
    # The unambiguous tell is max > 1 (input was not a Python bool).
    print("\nH2 — buffer aliasing / non-boolean input:")
    nonbool = [e for e in entries if e.get("max", 0) > 1]
    nonzero = [e for e in entries if e.get("gather_sum", 0) > 0]
    print(f"  Calls with any nonzero local_val: {len(nonzero)} / {len(entries)}"
          f"  (these may be legitimate True returns)")
    if nonbool:
        print(f"  STRONG SIGNAL: {len(nonbool)} calls had max > 1 "
              f"(input was not boolean)")
        print(_examples(nonbool))
    else:
        print("  OK: every nonzero local_val was 1 (boolean).")

    # ---------- H3: reducer malfunction ----------
    # (a) Allreduce SUM disagrees with the Allgather-summed local_vals.
    # (b) rank_sum != expected sum-of-ranks for a comm of this size.
    print("\nH3 — Allreduce reducer malfunction:")
    sum_mismatch = [e for e in entries
                    if "sum" in e and "gather_sum" in e
                    and e["sum"] != e["gather_sum"]]
    rank_sum_fail = [e for e in entries
                     if "rank_sum" in e and "expected_rank_sum" in e
                     and e["rank_sum"] != e["expected_rank_sum"]]
    print(f"  sum != gather_sum (reducer disagreeing with gather): "
          f"{len(sum_mismatch)}")
    if sum_mismatch:
        print(_examples(sum_mismatch))
    print(f"  rank_sum sanity failures (SUM broken on this comm): "
          f"{len(rank_sum_fail)}")
    if rank_sum_fail:
        print(_examples(rank_sum_fail))

    # ---------- H4: duplicate rank participation ----------
    print("\nH4 — duplicate rank participation in mpicomm:")
    dups = [e for e in entries
            if "unique" in e and "count" in e and e["unique"] < e["count"]]
    print(f"  Calls with duplicate world ranks: {len(dups)}")
    if dups:
        print(_examples(dups))

    # ---------- Verdict ----------
    print("\nVerdict:")
    triggered = []
    if h1_hits:
        triggered.append("H1 (wider/unstable membership)")
    if nonbool:
        triggered.append("H2 (non-boolean input)")
    if sum_mismatch or rank_sum_fail:
        triggered.append("H3 (reducer)")
    if dups:
        triggered.append("H4 (duplicate ranks)")
    if triggered:
        print("  Hypotheses triggered: " + ", ".join(triggered))
    else:
        if nonzero:
            print("  No invariant violations. Some calls returned nonzero;"
                  " consistent with legitimate shutdown signals.")
        else:
            print("  Clean log: no anomalies on any of the four hypotheses.")


def main(argv):
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <log_file>", file=sys.stderr)
        return 2
    path = argv[1]
    entries = parse(path)
    report(entries, path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
