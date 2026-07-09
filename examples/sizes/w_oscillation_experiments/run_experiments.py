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
runs a battery of would-be remedies and scores each on two axes: whether it
*stops the cycle* (per-iteration metrics from ``WOscExperimentMonitor``) and
whether what it produces is any *good* (end-of-run solution quality vs the EF
optimum).

Cycle metrics:

* ``zc``  -- how many nonants the zero-crossings detector still flags (scale-free;
             can be fooled by anything that freezes W).
* ``gap`` -- the PH primal gap ``sum_s p_s |x_s - xbar|`` -- the ground truth:
             low only when the scenarios actually agree.

Quality metrics (section 4), both vs the monolithic EF optimum ``z*``:

* x-gap -- expected cost of committing to the consensus xbar, above ``z*``.
* W-gap -- Lagrangian bound from the final W, below ``z*`` (a MIP has a duality
           gap, so read it relative across arms).

Groups of runs:

1. **Interventions** at the model's native (small) rho: plain, W-damping,
   rho reduction (geometric), W-reset, ``prox_boost`` (scale only the quadratic
   penalty, leaving the dual step at native rho -- one-shot, re-firing, held, and
   escalating variants), mpi-sppy's built-in ``smoothing`` (anchor each scenario
   to its own EMA trajectory), and ``fix`` (the slam analogue -- fix one flagged
   nonant per cooldown to its per-scenario max).
2. **rho level**: disable the ``_rho_setter`` so ``--default-rho`` applies
   uniformly, and sweep it -- to show the cycle is a *small-rho* artifact.
3. **rho perturbation**: keep the native rho but add ±eps jitter to each value.

Headline findings (see the emitted summary): ``fix``, an **escalating** prox
boost, and a **larger rho** each converge the primal gap; fixed-magnitude
state-perturbing moves (W-damping, W-reset, rho reduction/jitter, a one-shot or
re-firing prox boost) leave the cycle intact or only damp it. But convergence and
quality are different axes: a larger rho converges *fastest* yet lands the *worst*
decision and a loose bound, the native cycle orbits the optimum (best W-bound),
and escalating prox is the sweet spot for the decision (near-optimal x) at the
cost of a frozen, loose W. See ``README.md``.

Each arm runs in its own subprocess (``_run_one_arm.py``) for clean state.
Needs a MIP solver (gurobi_persistent by default). Not wired into CI.

Usage::

  python run_experiments.py
  python run_experiments.py --solver-name cplex --iters 120 --outdir results
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
    ("prox-boost (x10, 5 iters, one-shot)",
     {"intervention": "prox_boost", "boost_factor": 10.0, "boost_iters": 5}),
    ("prox-refire (x10, 5 iters, cooldown 5)",
     {"intervention": "prox_boost", "boost_factor": 10.0, "boost_iters": 5,
      "refire_cooldown": 5}),
    # boost_iters far exceeds the budget => once it fires (~start_iter) it never
    # reverts: a near-permanent prox-only boost, held to the end of the run.
    ("prox-hold (x10, held to end)",
     {"intervention": "prox_boost", "boost_factor": 10.0, "boost_iters": 100000}),
    # held AND escalating: ramp the multiplier x2 every 5 iters while the primal
    # gap persists (> escalate_gap_thresh) -- keep cranking the penalty until it
    # forces consensus. The only prox-only schedule that actually converges.
    ("prox-escalate (x10 base, x2/5 iters)",
     {"intervention": "prox_boost", "boost_factor": 10.0, "boost_iters": 100000,
      "escalate_mult": 2.0, "escalate_every": 5, "escalate_on": "gap"}),
    # mpi-sppy's built-in smoothed PH: anchor each scenario to its own EMA
    # trajectory (penalty p = ratio*rho, memory beta). A different mechanism
    # from prox (per-scenario, not consensus). Representative default; a ratio
    # sweep (r=1 .. 100) only makes it worse -- see README.
    ("smoothing (r=0.1, b=0.2)",
     {"intervention": "none", "smoothing": True,
      "smoothing_rho_ratio": 0.1, "smoothing_beta": 0.2}),
    ("fix (slam analogue)", {"intervention": "fix"}),
]
_RHO_LEVELS = [0.001, 0.01, 0.1, 0.3, 1.0]  # uniform (rho_setter disabled)
_PERTURBATIONS = [
    ("native (eps 0)", {"eps": 0.0}),
    ("jitter +/-50% per-var", {"eps": 0.5, "pmode": "per_var"}),
    ("jitter +/-50% per-call", {"eps": 0.5, "pmode": "per_call"}),
]


def _run(arm, cfg):
    """Write the arm JSON, run the worker subprocess, and return its metrics dict
    {iter -> (zc, gap, w)}, a status string, and its solution-quality bounds
    {inner, outer} (or None if not measured)."""
    path = os.path.join(cfg["outdir"], f"{arm['arm']}.json")
    arm = dict(arm, num_scens=cfg["num_scens"], iters=cfg["iters"],
               solver=cfg["solver"], measure_quality=True,
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
    bounds = None
    bpath = arm["out_csv"].rsplit(".csv", 1)[0] + ".bounds.json"
    if os.path.exists(bpath):
        with open(bpath) as f:
            bounds = json.load(f)
    return metrics, status, bounds


def _solve_ef(cfg):
    """Solve the monolithic extensive form once for the ground-truth optimum z*
    (the yardstick for both x-quality and W-quality). Returns z* or None."""
    sys.path.insert(0, os.path.dirname(_HERE))   # the examples/sizes dir (sizes)
    import sizes
    from mpisppy.opt.ef import ExtensiveForm
    names = [f"Scenario{i + 1}" for i in range(cfg["num_scens"])]
    ef = ExtensiveForm({"solver": cfg["solver"]}, names, sizes.scenario_creator,
                       scenario_creator_kwargs={"scenario_count": cfg["num_scens"]})
    ef.solve_extensive_form()
    return ef.get_objective_value()


def _quality_rows(labels, quality, z_star):
    """Markdown rows scoring each arm against z* (a minimization): x-gap is how
    far the committed consensus decision's expected cost sits above z*; W-gap is
    how far the Lagrangian bound from the final W sits below z*. Small = good."""
    rows = []
    for lab in labels:
        b = quality.get(lab)
        if not b or b.get("inner") is None or b.get("outer") is None:
            rows.append(f"| {lab} | - | - | - | - | not measured |")
            continue
        inner, outer = b["inner"], b["outer"]
        xgap = (inner - z_star) / abs(z_star) * 100.0
        wgap = (z_star - outer) / abs(z_star) * 100.0
        # thresholds separate the observed clusters: the small-rho arms sit near
        # x <0.5% / W <1%, the cycle-breakers push W past a few %, and a large
        # uniform rho pushes x past a couple %.
        if xgap < 1.0 and wgap < 1.5:
            tag = "good x, good W"
        elif xgap < 1.0:
            tag = "good x, **loose W**"
        elif wgap < 1.5:
            tag = "**poor x**, good W"
        else:
            tag = "poor x, loose W"
        rows.append(f"| {lab} | {inner:.0f} | {xgap:+.2f}% | {outer:.0f} | "
                    f"{wgap:+.1f}% | {tag} |")
    return rows


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
        # a huge gap with LOW zc means W froze while the scenarios flew apart
        # (decoupled); a huge gap with HIGH zc means the cycle got bigger, not
        # broken (amplified) -- e.g. smoothing anchors each scenario to its own
        # lagged EMA and fights consensus.
        tag = "decoupled (gap exploded)" if zc < 10 else "cycle amplified"
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
    p.add_argument("--iters", type=int, default=80)
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
    quality = {}          # label -> {inner, outer} solution-quality bounds
    lines += ["## 1. Interventions (native rho)\n",
              "| arm | zc | primal gap | reading |", "|---|---|---|---|"]
    for label, over in _INTERVENTIONS:
        arm = dict(over, arm=label.split()[0] + "_" + over["intervention"],
                   start_iter=args.start_iter, rho_setter="native",
                   default_rho=1.0)
        print(f"  [{label}] ...")
        m, st, b = _run(arm, cfg)
        per_iter[label] = m
        quality[label] = b
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
        m, st, b = _run(arm, cfg)
        quality[f"rho={rho:g}"] = b
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
        m, st, b = _run(arm, cfg)
        quality[label] = b
        lines.append(_row(label, m, st, cfg["iters"]))
    lines += [
        "",
        "* **zc**: nonants the detector still flags (scale-free; fooled by "
        "frozen W).",
        "* **primal gap** `sum_s p_s |x_s - xbar|`: low only when scenarios "
        "actually agree.",
        "",
        "Only `fix` (changing the problem structure), an escalating prox boost, "
        "and a larger rho converge the gap. A *fixed*-magnitude state-perturbing "
        "move (W-damping, W-reset, rho reduction, rho jitter, a one-shot or "
        "re-firing prox boost) leaves the cycle intact, decouples the scenarios, "
        "or only damps it to a residual. `smoothing` *amplifies* the cycle (it "
        "anchors each scenario to its own lagged EMA, fighting consensus), worse "
        "as the ratio grows. The sizes cycle is a small-rho artifact -- its "
        "`_rho_setter` uses cost x 0.001.",
    ]

    # ---- 4. solution quality (x and W) vs the EF optimum --------------------
    z_star = None
    try:
        z_star = _solve_ef(cfg)
    except Exception as e:                          # noqa: BLE001
        print(f"EF solve failed, skipping quality section: {e}")
    if z_star is not None:
        q_labels = ([lab for lab, _ in _INTERVENTIONS]
                    + [f"rho={rho:g}" for rho in _RHO_LEVELS])
        lines += [
            "",
            f"## 4. Solution quality vs EF optimum (z* = {z_star:.0f})\n",
            "Sections 1-3 say whether an arm stopped *moving*; this says whether "
            "what it produced is any *good*. **x-gap** = expected cost of "
            "committing to the consensus xbar, above z* (small = a good "
            "decision). **W-gap** = Lagrangian bound from the final W, below z* "
            "(small = duals good enough to certify optimality). sizes is a MIP, "
            "so a duality gap keeps W-gap > 0 even for good W -- read it "
            "*relative* across arms.\n",
            "| arm | inner Ū | x-gap | outer L | W-gap | reading |",
            "|---|---|---|---|---|---|",
        ]
        lines += _quality_rows(q_labels, quality, z_star)
        lines += [
            "",
            "**Convergence and solution quality are different axes.** The native "
            "small-rho cycle never meets a convergence *criterion*, yet it orbits "
            "the optimum: its average xbar is near-optimal (x-gap ~0.3%) and its W "
            "gives the *tightest* Lagrangian bound (W-gap <1%). Raising rho "
            "uniformly converges fast but to a *worse* consensus -- both gaps blow "
            "up (rho=1: x ~3%, W loose; rho=0.3: W-gap ~20%) -- so \"a larger rho "
            "converges\" costs real solution quality. An **escalating prox boost "
            "is the sweet spot for the decision**: it converges (primal gap -> 0) "
            "and keeps x near-optimal, because it retains the small native dual "
            "step and settles near the cycle's centre rather than a distorted "
            "large-rho point. But a strong prox forces primal consensus regardless "
            "of the duals (the W's cancel in aggregate), so it *freezes* W "
            "off-optimum -- a loose dual bound (W-gap ~3%). `fix` shares that "
            "profile. Net: for a good first-stage **decision**, escalating prox "
            "beats raising rho; for a tight dual **bound**, none of the "
            "cycle-breakers beats letting the small-rho cycle run.",
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
