###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""What actually breaks a PH W-oscillation cycle?  (experiments on sizes)

The ``sizes`` model cycles hard under plain PH: with its shipped ``_rho_setter``
the W vector settles into a stable limit cycle that never converges. This driver
runs a battery of would-be remedies and scores each by two metrics per iteration
(recorded by ``w_osc_experiment_ext.WOscExperimentMonitor``):

* ``zc``  -- how many nonants the zero-crossings detector still flags (scale-free;
             can be fooled by anything that freezes W).
* ``gap`` -- the PH primal gap ``sum_s p_s |x_s - xbar|`` -- the ground truth:
             low only when the scenarios actually agree.

Three groups of runs:

1. **Interventions** at the model's native (small) rho: plain, W-damping,
   rho reduction (geometric), W-reset, and ``fix`` (the slam analogue -- fix one
   flagged nonant per cooldown to its per-scenario max).
2. **rho level**: disable the ``_rho_setter`` so ``--default-rho`` applies
   uniformly, and sweep it -- to show the cycle is a *small-rho* artifact.
3. **rho perturbation**: keep the native rho but add ±eps jitter to each value.

Headline finding (see the emitted summary): only ``fix`` (changing the problem
structure) and using a **larger rho** converge the primal gap. Every
state-perturbing move -- W-damping, W-reset, rho reduction, rho jitter -- leaves
the cycle intact, decouples the scenarios (gap explodes while W merely freezes),
or makes it worse.

Each arm runs in its own subprocess (``_run_one_arm.py``) for clean state.
Needs a MIP solver (gurobi_persistent by default). Not wired into CI.

Usage::

  python run_experiments.py
  python run_experiments.py --solver-name cplex --iters 60 --outdir results
"""

import argparse
import csv
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "_run_one_arm.py")

# (label, arm-overrides) at the model's native rho.
_INTERVENTIONS = [
    ("plain PH", {"intervention": "none"}),
    ("w_damping x0.5", {"intervention": "w_damping", "factor": 0.5}),
    ("rho reduction (geom x0.7)", {"intervention": "rho_reduce", "factor": 0.7}),
    ("W reset + rho x0.5", {"intervention": "w_reset", "factor": 0.5}),
    ("fix (slam analogue)", {"intervention": "fix"}),
]
_RHO_LEVELS = [0.001, 0.01, 0.1, 0.3, 1.0]  # uniform (rho_setter disabled)
_PERTURBATIONS = [
    ("native (eps 0)", {"eps": 0.0}),
    ("jitter +/-50% per-var", {"eps": 0.5, "pmode": "per_var"}),
    ("jitter +/-50% per-call", {"eps": 0.5, "pmode": "per_call"}),
]


def _run(arm, cfg):
    """Write the arm JSON, run the worker subprocess, return its metrics dict
    {iter -> (zc, gap, w)} plus a status string."""
    path = os.path.join(cfg["outdir"], f"{arm['arm']}.json")
    arm = dict(arm, num_scens=cfg["num_scens"], iters=cfg["iters"],
               solver=cfg["solver"],
               out_csv=os.path.join(cfg["outdir"], f"{arm['arm']}.csv"))
    with open(path, "w") as f:
        json.dump(arm, f, indent=2)
    status = "ok"
    try:
        p = subprocess.run([sys.executable, _WORKER, path], cwd=_HERE,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=cfg["timeout"])
        if p.returncode != 0:
            status = f"exit{p.returncode}"
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
    metrics = {}
    if os.path.exists(arm["out_csv"]):
        with open(arm["out_csv"]) as f:
            for r in csv.DictReader(f):
                metrics[int(r["iteration"])] = (
                    int(r["zc_flagged"]), float(r["primal_gap"]),
                    float(r["mean_abs_W"]))
    return metrics, status


def _tail(metrics, idx, n=10):
    its = sorted(metrics)[-n:]
    if not its:
        return 0.0
    return sum(metrics[i][idx] for i in its) / len(its)


def _report_gap(metrics, iters):
    """Representative primal gap. A run that stops before the budget has
    converged (or stalled) at its final value, so report that; a run that uses
    the whole budget is judged by its last-10 mean (a cycle has no single final
    value). Returns (gap, stopped_early, maxit)."""
    if not metrics:
        return 0.0, False, 0
    maxit = max(metrics)
    if maxit < iters:                       # PH stopped early
        return metrics[maxit][1], True, maxit
    return _tail(metrics, 1), False, maxit


def _row(label, metrics, status, iters):
    """One Markdown table row: label | zc | gap | reading."""
    if status != "ok":
        return f"| {label} | - | - | did not complete ({status}) |"
    zc = _tail(metrics, 0)
    gap, early, maxit = _report_gap(metrics, iters)
    if gap < 25:
        tag = "**converges (cycle broken)**"
    elif gap > 500:
        tag = "decoupled (gap exploded)"
    else:
        tag = "still cycling"
    if early:
        tag += f" (converged @ iter {maxit})" if gap < 25 \
            else f" (stopped @ iter {maxit})"
    return f"| {label} | {zc:.1f} | {gap:.0f} | {tag} |"


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solver-name", default="gurobi_persistent")
    p.add_argument("--num-scens", type=int, default=3)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--start-iter", type=int, default=10)
    p.add_argument("--timeout", type=int, default=120,
                   help="per-arm subprocess timeout (s)")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    cfg = {"num_scens": args.num_scens, "iters": args.iters,
           "solver": args.solver_name, "timeout": args.timeout, "outdir": outdir}

    print(f"sizes-{args.num_scens}, {args.iters} iters, interventions from iter "
          f"{args.start_iter}\noutput -> {outdir}\n")

    lines = [f"# What breaks the sizes-{args.num_scens} W-oscillation cycle?\n",
             f"Budget {args.iters} iters; interventions from iter "
             f"{args.start_iter}. Metrics averaged over the last 10 iterations "
             "(unless the run converged and stopped early).\n"]

    # ---- 1. interventions at native rho -------------------------------------
    per_iter = {}
    lines += ["## 1. Interventions (native rho)\n",
              "| arm | zc | primal gap | reading |", "|---|---|---|---|"]
    for label, over in _INTERVENTIONS:
        arm = dict(over, arm=label.split()[0] + "_" + over["intervention"],
                   start_iter=args.start_iter, rho_setter="native",
                   default_rho=1.0)
        print(f"  [{label}] ...")
        m, st = _run(arm, cfg)
        per_iter[label] = m
        lines.append(_row(label, m, st, cfg["iters"]))
    lines.append("")

    # ---- 2. rho level (uniform, setter disabled) ----------------------------
    lines += ["## 2. rho level (uniform; native `_rho_setter` disabled)\n",
              "| default_rho | zc | primal gap | reading |", "|---|---|---|---|"]
    for rho in _RHO_LEVELS:
        arm = {"arm": f"rholevel_{rho:g}", "intervention": "none",
               "start_iter": args.start_iter, "rho_setter": "off",
               "default_rho": rho}
        print(f"  [rho level {rho:g}] ...")
        m, st = _run(arm, cfg)
        lines.append(_row(f"{rho:g}", m, st, cfg["iters"]))
    lines.append("")

    # ---- 3. rho perturbation (native rho + jitter) --------------------------
    lines += ["## 3. rho perturbation (native rho + jitter)\n",
              "| perturbation | zc | primal gap | reading |", "|---|---|---|---|"]
    for label, over in _PERTURBATIONS:
        arm = dict(over, arm="pert_" + label.split()[0].replace("+/-", ""),
                   intervention="none", start_iter=args.start_iter,
                   rho_setter=("native" if over["eps"] == 0.0 else "perturb"),
                   default_rho=1.0)
        print(f"  [perturb {label}] ...")
        m, st = _run(arm, cfg)
        lines.append(_row(label, m, st, cfg["iters"]))
    lines += [
        "",
        "* **zc**: nonants the detector still flags (scale-free; fooled by "
        "frozen W).",
        "* **primal gap** `sum_s p_s |x_s - xbar|`: low only when scenarios "
        "actually agree.",
        "",
        "Only `fix` (changing the problem structure) and a larger rho converge "
        "the gap. Every state-perturbing move leaves the cycle intact, "
        "decouples the scenarios, or worsens it. The sizes cycle is a small-rho "
        "artifact -- its `_rho_setter` uses cost x 0.001.",
    ]

    # ---- per-iteration CSV for the intervention arms ------------------------
    labels = [lab for lab, _ in _INTERVENTIONS]
    all_iters = sorted(set().union(*[set(per_iter[lab]) for lab in labels]))
    per_csv = os.path.join(outdir, "interventions_by_iteration.csv")
    with open(per_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration"] + [f"{lab}::zc" for lab in labels]
                   + [f"{lab}::gap" for lab in labels])
        for it in all_iters:
            row = [it]
            row += [per_iter[lab].get(it, ("", ))[0] for lab in labels]
            row += [round(per_iter[lab][it][1], 3) if it in per_iter[lab] else ""
                    for lab in labels]
            w.writerow(row)

    summary = "\n".join(lines) + "\n"
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"per-iteration CSV -> {per_csv}")
    print(f"summary           -> {os.path.join(outdir, 'summary.md')}")


if __name__ == "__main__":
    main()
