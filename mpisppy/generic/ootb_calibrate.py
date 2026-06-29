###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Out-of-the-box (OOTB) effort-calibration tool.

The effort_scaling coefficients and the EF effort budget are abstract by
construction (arbitrary "effort units"); left as hand-guesses they are
unassessable. This tool turns them into data-tuned, interpretable values from
timed solves on the mpi-sppy examples (the producer side of the dated-policy
migration path -- design doc sec. 9). It:

  1. FITS the effort_scaling shape -- cont_coeff, int_weight, int_exponent,
     int_nonant_coeff -- so that modeled effort tracks measured solve time, by
     timing extensive-form solves over a spread of bundle sizes across models of
     differing continuous / integer / nonant content.
  2. CALIBRATES effort to ~seconds: the fitted coefficients are kept in seconds
     units, so modeled effort approximates predicted solve time and the absolute
     budgets read as roughly seconds (ef_effort_budget is set from the policy's
     ef_target_seconds) instead of opaque large numbers.
  3. WRITES a new dated policy file with the fitted numbers in place of the
     cold-start guesses.

    python -m mpisppy.generic.ootb_calibrate --solver-name gurobi \\
        --output mpisppy/generic/ootb_policies/ootb_policy_2026-07-01.json

Caveats: solve time is machine- and solver-dependent and (for MIPs) noisy, so a
calibrated scale is per reference machine/solver and approximate. The fit is the
LINEAR part (cont/int/nonant coefficients) for each candidate integer exponent;
the best exponent is chosen by R^2. The pure fit (fit_effort_model) is
solver-free and unit-tested; the measurement needs a solver and is not run in CI.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time

import mpisppy.utils.sputils as sputils
from mpisppy.generic import out_of_the_box as ootb
from mpisppy.generic import ootb_validate as val


# Integer-exponent grid searched during the fit (int_exponent in the effort
# model). 1.0 = integers scale like LP; > 1 captures branch-and-bound blow-up.
EXPONENT_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

# Bundle sizes (scenarios per EF solve) used to probe how solve time scales.
DEFAULT_SPB_GRID = [1, 2, 4, 8, 16]

# How many times to solve each point (the minimum is kept -- a denoiser).
DEFAULT_REPS = 2


# ---------------------------------------------------------------------------
# Fitting  [pure; unit-tested]
# ---------------------------------------------------------------------------


def _round_sig(x: float, n: int = 6) -> float:
    """Round to n significant figures. Decimal-place rounding would destroy the
    legitimately tiny coefficients that multiply huge (int*spb)^exponent terms
    (e.g. int_weight ~ 1e-9 when the exponent is 3 and there are ~150 integers)."""
    import math
    if x == 0 or not math.isfinite(x):
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def _design_columns(point: dict, exponent: float):
    """The three effort features for one measured point, matching ootb._effort:
    continuous content (~spb), integer content (^exponent), integer nonants
    (fixed per bundle)."""
    cont = point["vars_cont"] * point["spb"]
    nint = point["vars_int"] * point["spb"]
    return [cont, nint ** exponent, point["nonants_int"]]


def fit_effort_model(points: list, exponents=EXPONENT_GRID) -> dict:
    """Fit the effort_scaling coefficients + an effort->seconds scale to timed
    solves.

    points: list of dicts with PER-SCENARIO `vars_cont`, `vars_int`,
    `nonants_int`, the bundle size `spb`, and the measured `seconds`.

    For each candidate integer exponent we solve a non-negative least squares for
    (a, b, c) so that a*cont + b*int^p + c*nonant ~= seconds, and keep the
    exponent with the best R^2. The fitted coefficients are kept in SECONDS units
    (we do NOT divide the scale back out), so modeled effort approximates the
    predicted solve time directly -- the absolute budgets (ef_effort_budget) then
    read as roughly seconds (e.g. ~120) instead of opaque large numbers.
    seconds_per_effort_unit is therefore ~1; it stays in the schema to document
    the units and to let a focus rescale. Returns the effort_scaling fields plus
    r2 / n_points.
    """
    import numpy as np
    from scipy.optimize import nnls

    if len(points) < 3:
        raise ValueError(f"need at least 3 measured points to fit, got {len(points)}")

    y = np.array([p["seconds"] for p in points], dtype=float)
    best = None
    for p in exponents:
        A = np.array([_design_columns(pt, p) for pt in points], dtype=float)
        coef, _ = nnls(A, y)
        ss_res = float(((A @ coef - y) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if best is None or r2 > best["r2"]:
            best = {"coef": coef, "p": p, "r2": r2}

    a, b, c = (float(x) for x in best["coef"])
    return {
        "cont_coeff": _round_sig(a),
        "int_weight": _round_sig(b),
        "int_exponent": best["p"],
        "int_nonant_coeff": _round_sig(c),
        "seconds_per_effort_unit": 1.0,   # effort is calibrated to ~seconds
        "r2": round(best["r2"], 4),
        "n_points": len(points),
    }


# ---------------------------------------------------------------------------
# Measurement  [needs a solver; not run in CI]
# ---------------------------------------------------------------------------


def _pick_solver(policy: dict, requested: str | None) -> str:
    """A non-persistent solver for one-shot EF solves (persistent interfaces
    need set_instance and add setup overhead that would pollute the timing)."""
    if requested:
        return requested
    for name in policy["solver"]["preference_order"]:
        plain = name.replace("_persistent", "")
        if plain in ootb._detect_available_solvers([plain]):
            return plain
    raise RuntimeError("no known solver available; pass --solver-name")


def _time_ef_solve(names, scenario_creator, kwargs, solver_name, reps) -> float:
    """Build an EF over `names` scenarios and return the MIN wall-clock solve
    time over `reps` solves (model build excluded from the timing)."""
    import pyomo.environ as pyo
    best = None
    for _ in range(reps):
        ef = sputils.create_EF(names, scenario_creator, kwargs,
                               suppress_warnings=True)
        solver = pyo.SolverFactory(solver_name)
        t0 = time.perf_counter()
        solver.solve(ef)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best


def measure_example(spec: dict, solver_name: str, spb_grid, reps) -> list:
    """Probe one example's per-scenario size profile, then time EF solves over a
    range of bundle sizes. Returns a list of measured points."""
    module = val._load_example_module(spec)
    cfg = val._example_cfg(spec, module)
    kwargs = module.kw_creator(cfg)
    profile = ootb._size_profile(ootb._build_probe_scenario(module, cfg))
    total = ootb._detect_num_scens(module, cfg)
    points = []
    for spb in spb_grid:
        if spb > total:
            break
        names = module.scenario_names_creator(spb)
        try:
            seconds = _time_ef_solve(names, module.scenario_creator, kwargs,
                                     solver_name, reps)
        except Exception as e:                            # noqa: BLE001
            print(f"  [{spec['name']} spb={spb}] solve failed: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            continue
        pt = {"example": spec["name"], "spb": spb,
              "vars_cont": profile["vars_cont"], "vars_int": profile["vars_int"],
              "nonants_int": profile["nonants_int"], "seconds": round(seconds, 6)}
        points.append(pt)
        print(f"  {spec['name']:8} spb={spb:<3} "
              f"cont={profile['vars_cont']} int={profile['vars_int']} "
              f"nonants_int={profile['nonants_int']}  -> {seconds:.4f}s")
    return points


def _calibration_specs() -> list:
    """Example specs with scenario counts tuned for calibration: enough
    scenarios that several bundle sizes are reachable (the validator's run-tier
    deliberately uses tiny counts; calibration wants a wider spread, and more
    integer points come from a larger MIP EF)."""
    counts = {"farmer": {"num_scens": 16}, "sizes": {"num_scens": 10},
              "aircond": {"branching_factors": [4, 4]}}
    specs = []
    for spec in val.example_models():
        s = dict(spec)
        s["scens"] = counts.get(spec["name"], spec["scens"])
        specs.append(s)
    return specs


def collect_points(policy: dict, solver_name: str, spb_grid, reps,
                   specs=None) -> list:
    points = []
    for spec in (specs if specs is not None else _calibration_specs()):
        print(f"[calibrate] timing {spec['name']} ({spec['kind']}) ...")
        points.extend(measure_example(spec, solver_name, spb_grid, reps))
    return points


# ---------------------------------------------------------------------------
# Apply the fit to a policy
# ---------------------------------------------------------------------------


def calibrated_policy(base_policy: dict, fit: dict, points: list,
                      solver_name: str, today: str) -> dict:
    """Return a copy of base_policy with the fitted effort_scaling, the
    effort->seconds scale, and a seconds-derived ef_effort_budget."""
    import copy
    pol = copy.deepcopy(base_policy)

    es = pol["effort_scaling"]
    es["cont_coeff"] = fit["cont_coeff"]
    es["int_weight"] = fit["int_weight"]
    es["int_exponent"] = fit["int_exponent"]
    es["int_nonant_coeff"] = fit["int_nonant_coeff"]
    es["seconds_per_effort_unit"] = round(fit["seconds_per_effort_unit"], 8)
    es["_calibration"] = {
        "solver": solver_name, "r2": fit["r2"], "n_points": fit["n_points"],
        "date": today, "note": "Fitted by mpisppy.generic.ootb_calibrate on the "
        "example set; per reference machine/solver and approximate. "
        "seconds_per_effort_unit: modeled effort * this ~= seconds."}
    es.pop("_cold_start_guess", None)        # these numbers are now data-tuned

    # Derive the EF budget in effort units from the seconds target so the budget
    # is consistent with the fitted scale (effort <= seconds / sec_per_effort).
    ef = pol["ef_fallback"]
    spe = fit["seconds_per_effort_unit"]
    if spe > 0 and ef.get("ef_target_seconds"):
        ef["ef_effort_budget"] = int(round(ef["ef_target_seconds"] / spe))
        ef["_calibration_note"] = ("ef_effort_budget derived from "
                                   "ef_target_seconds / seconds_per_effort_unit "
                                   f"(calibrated {today}).")
        guesses = ef.get("_cold_start_guess", [])
        ef["_cold_start_guess"] = [g for g in guesses if g != "ef_effort_budget"]

    pol["policy_version"] = today
    pol["provenance"] = (f"CALIBRATED {today} by mpisppy.generic.ootb_calibrate "
                         f"(solver {solver_name}, R^2={fit['r2']}, "
                         f"{fit['n_points']} timed EF solves on the example set). "
                         "effort_scaling and ef_effort_budget are data-tuned; "
                         "remaining numbers are still authored. Per reference "
                         "machine/solver and approximate (MIP times are noisy).")
    return pol


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------


def run_calibration(base_policy_path, solver_name=None, spb_grid=DEFAULT_SPB_GRID,
                    reps=DEFAULT_REPS, today=None):
    """Measure, fit, and return (calibrated_policy_dict, fit, points)."""
    base = ootb.load_policy(base_policy_path or None)
    solver = _pick_solver(base, solver_name)
    print(f"[calibrate] solver: {solver}")
    points = collect_points(base, solver, spb_grid, reps)
    fit = fit_effort_model(points)
    if today is None:
        today = datetime.date.today().isoformat()
    pol = calibrated_policy(base, fit, points, solver, today)
    return pol, fit, points


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m mpisppy.generic.ootb_calibrate",
        description="Calibrate OOTB effort_scaling from timed example solves "
                    "(design doc sec. 9).")
    p.add_argument("--base", default="",
                   help="base policy to start from (default: shipped default)")
    p.add_argument("--solver-name", default=None,
                   help="solver for the timed solves (default: first available)")
    p.add_argument("--output", default=None,
                   help="write the calibrated policy here (default: print only)")
    p.add_argument("--reps", type=int, default=DEFAULT_REPS,
                   help=f"solves per point, min kept (default {DEFAULT_REPS})")
    p.add_argument("--spb", default=None,
                   help="comma-separated bundle sizes (default "
                        f"{','.join(map(str, DEFAULT_SPB_GRID))})")
    args = p.parse_args(argv)

    spb_grid = ([int(x) for x in args.spb.split(",")] if args.spb
                else DEFAULT_SPB_GRID)
    pol, fit, points = run_calibration(args.base, args.solver_name, spb_grid,
                                       args.reps)

    print("\n[calibrate] fit:")
    for k in ("cont_coeff", "int_weight", "int_exponent", "int_nonant_coeff",
              "seconds_per_effort_unit", "r2", "n_points"):
        print(f"    {k}: {fit[k]}")
    print(f"    ef_effort_budget -> {pol['ef_fallback']['ef_effort_budget']} "
          f"(from {pol['ef_fallback'].get('ef_target_seconds')}s target)")

    if args.output:
        with open(args.output, "w") as fp:
            json.dump(pol, fp, indent=2)
        print(f"\n[calibrate] wrote calibrated policy to {args.output}")
        print("[calibrate] validate it with: python -m mpisppy.generic."
              f"ootb_validate {args.output}")
    else:
        print("\n[calibrate] (no --output given; not writing a policy file)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
