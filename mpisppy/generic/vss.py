###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Value of the Stochastic Solution (VSS) report for generic_cylinders.

VSS = EEV - RP  (minimization; VSS = RP - EEV for maximization), where

  RP  = the stochastic-program optimum -- the here-and-now solution the run
        already computed: exact from an --EF run, or the decomposition
        incumbent (BestInnerBound), with a bracket from BestOuterBound.
  EV  = the mean-value (expected-value) problem: replace the random data by
        its average and solve. x_bar is its first-stage solution.
  EEV = x_bar fixed in the first stage and evaluated honestly across every
        scenario.

Two-stage only in this version. See doc/designs/vss_design.md.

Cost: computing EEV re-solves every scenario once with the first stage
fixed. Fixing the first stage decouples the scenarios, so each solve is
usually easier than the original; how much wall-clock this adds depends on
the model and can be significant for some problems (very many scenarios or
expensive recourse). The EV and EEV solves reuse the run's solver and
solver_options (including any mipgap) via solver_specification, so all three
numbers are solved consistently -- but VSS is a difference of two optimized
values, so a loose mipgap makes a small VSS unreliable.
"""

import math

import numpy as np
import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
from mpisppy import global_toc, MPI
from mpisppy.utils import xhat_eval
from mpisppy.utils import solver_spec
from mpisppy.generic.parsing import name_lists, proper_bundles


def vss_prep(module, cfg):
    """Validate that a VSS report can be produced, and fail fast -- before
    the main solve -- if not. Called early by generic_cylinders when --vss
    is set, so a long run is never wasted only to fail at the report step.

    V1 is two-stage only and does not (yet) support the objective-rewriting
    transform (CVaR) or the scenario/first-stage-restructuring paths (proper
    bundles, ADMM), because RP and EEV would then have to be defined against
    the transformed problem to be comparable.
    """
    if getattr(module, "average_scenario_creator", None) is None:
        raise RuntimeError(
            "--vss requires the scenario module to define "
            "average_scenario_creator (the same contract used by Jensen's "
            "bound; see doc/src/jensens.rst). It builds the mean-value "
            "scenario whose first-stage solution VSS evaluates."
        )
    if cfg.get("branching_factors") is not None:
        raise RuntimeError(
            "--vss is two-stage only in this version; it cannot be used with "
            "multistage runs (--branching-factors)."
        )
    if proper_bundles(cfg):
        raise RuntimeError("--vss cannot (yet) be combined with proper bundles.")
    if cfg.get("admm", ifmissing=False) or cfg.get("stoch_admm", ifmissing=False):
        raise RuntimeError("--vss cannot (yet) be combined with ADMM.")
    if cfg.get("cvar", ifmissing=False):
        raise RuntimeError("--vss cannot (yet) be combined with --cvar.")


def do_vss(module, cfg, scenario_creator, scenario_creator_kwargs,
           scenario_denouement, ef=None, wheel=None):
    """Compute and print a VSS report after the main algorithm.

    Exactly one of ``ef`` (from an --EF run) or ``wheel`` (from a
    decomposition run) supplies RP. MPI-collective: EEV is evaluated across
    all ranks, so every rank must call this together.

    Args:
        module: the model module (must define average_scenario_creator)
        cfg (Config): parsed options
        scenario_creator (function): the run's scenario creator
        scenario_creator_kwargs (dict): kwargs for the scenario creator
        scenario_denouement (function): unused for the quiet EEV pass
        ef (ExtensiveForm or None): source of an exact RP
        wheel (WheelSpinner or None): source of an incumbent RP + bracket
    """
    if (ef is None) == (wheel is None):
        raise ValueError("do_vss: pass exactly one of ef= or wheel=")
    comm = MPI.COMM_WORLD

    all_scenario_names, _ = name_lists(module, cfg)

    # The VSS solves (EV and EEV) reuse the SAME solver and solver_options as
    # the run that produced RP -- including any mipgap in that option string --
    # so the three numbers are solved consistently. After an EF run prefer the
    # EF spec (falling back to the default); after a decomposition run use the
    # default (subproblem) spec.
    prefix = ["EF", ""] if ef is not None else ""
    _, solver_name, solver_options = solver_spec.solver_specification(cfg, prefix)

    # EV problem: solve the mean-value scenario for its objective and its
    # first-stage solution x_bar.
    ev_obj, x_bar, is_min = _solve_average_scenario(
        module, solver_name, solver_options, scenario_creator_kwargs)

    # EEV: fix x_bar in the first stage, evaluate across all scenarios.
    eev, infeasible_names = _compute_eev(
        solver_name, solver_options, scenario_creator,
        scenario_creator_kwargs, all_scenario_names, x_bar)

    # RP: exact from EF, or incumbent (+ bracket) from the wheel.
    if ef is not None:
        rp_point = comm.bcast(ef.get_objective_value() if comm.Get_rank() == 0
                              else None, root=0)
        rp_source = "EF, exact"
        inner = outer = None
    else:
        inner, outer = _reduce_bounds(comm, wheel.BestInnerBound,
                                      wheel.BestOuterBound, is_min)
        rp_point = inner
        rp_source = "decomposition incumbent"

    vss = _vss_value(is_min, rp_point, eev)

    # Bracket: with RP only known to lie in [outer, inner], VSS lies in a
    # corresponding interval. Only meaningful when the run left a gap and
    # EEV is finite.
    have_bracket = (inner is not None and outer is not None
                    and math.isfinite(inner) and math.isfinite(outer)
                    and inner != outer)
    vss_bracket = None
    if have_bracket and math.isfinite(eev):
        if is_min:
            vss_bracket = (eev - inner, eev - outer)
        else:
            vss_bracket = (inner - eev, outer - eev)

    result = {"RP": rp_point, "EV": ev_obj, "EEV": eev, "VSS": vss,
              "is_min": is_min, "rp_source": rp_source,
              "inner": inner, "outer": outer, "vss_bracket": vss_bracket,
              "infeasible_scenarios": infeasible_names}
    _print_report(result)
    return result


def _solve_average_scenario(module, solver_name, solver_options,
                            scenario_creator_kwargs):
    """Build and solve the mean-value scenario. Return
    (EV_objective, x_bar, is_minimizing) where x_bar is the ROOT first-stage
    solution as a 1-D np.ndarray in nonant_vardata_list order.

    Two-stage only (asserted). Every rank solves this identical deterministic
    scenario independently -- no collective communication.
    """
    avg = module.average_scenario_creator(
        "AverageScenario", **(scenario_creator_kwargs or {}))
    if not hasattr(avg, "_mpisppy_node_list") or len(avg._mpisppy_node_list) != 1:
        raise RuntimeError(
            "--vss is two-stage only; average_scenario_creator must return a "
            "model with exactly one tree node (ROOT)."
        )
    solver = pyo.SolverFactory(solver_name)
    for k, v in (solver_options or {}).items():
        solver.options[k] = v
    if sputils.is_persistent(solver):
        solver.set_instance(avg)
        results = solver.solve(tee=False)
    else:
        results = solver.solve(avg, tee=False)
    if not pyo.check_optimal_termination(results):
        raise RuntimeError(
            "--vss: the mean-value (EV) problem did not solve to optimality "
            f"(termination_condition={results.solver.termination_condition}). "
            "VSS cannot be computed."
        )
    obj = sputils.find_active_objective(avg)
    ev_obj = pyo.value(obj)
    is_min = (obj.sense == pyo.minimize)
    root = avg._mpisppy_node_list[0]
    x_bar = np.array([pyo.value(v) for v in root.nonant_vardata_list], dtype="d")
    return ev_obj, x_bar, is_min


def _compute_eev(solver_name, solver_options, scenario_creator,
                 scenario_creator_kwargs, all_scenario_names, x_bar):
    """Fix x_bar in the first stage and evaluate expected cost across all
    scenarios. Return (EEV, infeasible_scenario_names). EEV is math.inf if
    the mean-value first stage is infeasible in any scenario.

    MPI-collective: every rank builds its share of scenarios, so all ranks
    must call this together.
    """
    options = {
        "iter0_solver_options": None,
        "iterk_solver_options": None,
        "display_timing": False,
        "solver_name": solver_name,
        "verbose": False,
        "solver_options": solver_options,
    }
    ev = xhat_eval.Xhat_Eval(
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement=None,
        scenario_creator_kwargs=scenario_creator_kwargs,
        all_nodenames=None,
        mpicomm=MPI.COMM_WORLD,
    )
    ev._lazy_create_solvers()
    ev._fix_nonants({"ROOT": x_bar})
    # compute_val_at_nonant=False => need_solution=False in solve_one, so a
    # clean per-scenario infeasibility does NOT raise. That matters for MPI:
    # a raise on only some ranks would deadlock the collectives below.
    ev.solve_loop(solver_options=solver_options, gripe=True, tee=False,
                  compute_val_at_nonant=False)

    local_infeasible = [
        k for k, s in ev.local_scenarios.items()
        if not getattr(s._mpisppy_data, "solution_available", False)
    ]
    n_local = np.array([float(len(local_infeasible))])
    n_global = np.zeros(1)
    ev.mpicomm.Allreduce(n_local, n_global, op=MPI.SUM)

    if n_global[0] > 0:
        gathered = ev.mpicomm.gather(local_infeasible, root=0)
        names = []
        if ev.mpicomm.Get_rank() == 0:
            for lst in gathered:
                names.extend(lst)
        return math.inf, names

    # All feasible: Eobjective is collective, so every rank calls it.
    return ev.Eobjective(), []


def _reduce_bounds(comm, best_inner, best_outer, is_min):
    """Make the wheel's best inner/outer bounds available on every rank.

    BestInnerBound / BestOuterBound are set only on the hub rank(s) and are
    None elsewhere. Reduce with the sense-appropriate op, treating a missing
    value as the identity (so the hub's real value wins).
    """
    if is_min:
        inner = comm.allreduce(best_inner if best_inner is not None
                               else math.inf, op=MPI.MIN)
        outer = comm.allreduce(best_outer if best_outer is not None
                               else -math.inf, op=MPI.MAX)
    else:
        inner = comm.allreduce(best_inner if best_inner is not None
                               else -math.inf, op=MPI.MAX)
        outer = comm.allreduce(best_outer if best_outer is not None
                               else math.inf, op=MPI.MIN)
    return inner, outer


def _vss_value(is_min, rp, eev):
    """VSS with the sign convention; math.inf if EEV is infinite."""
    if math.isinf(eev):
        return math.inf
    return (eev - rp) if is_min else (rp - eev)


def _print_report(result):
    """Print the VSS report once, on global rank 0, from the do_vss dict."""
    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    def _f(x):
        if x is None:
            return "n/a"
        if math.isinf(x):
            return "+inf" if x > 0 else "-inf"
        return f"{x:.6g}"

    def _row(label, value, suffix=""):
        return f"  {label:<39}: {value:>14}{suffix}"

    is_min = result["is_min"]
    eev = result["EEV"]
    inner, outer = result["inner"], result["outer"]
    sense = "min" if is_min else "max"
    formula = "EEV - RP" if is_min else "RP - EEV"

    lines = ["", "================= VSS report =================",
             _row("RP  (stochastic solution, here-and-now)",
                  _f(result["RP"]), f"   [{result['rp_source']}]")]

    have_bracket = (inner is not None and outer is not None
                    and math.isfinite(inner) and math.isfinite(outer)
                    and inner != outer)
    if have_bracket:
        lo, hi = (outer, inner) if is_min else (inner, outer)
        lines.append(_row("    optimality bracket [outer, inner]",
                          f"[{_f(lo)}, {_f(hi)}]"))

    lines.append(_row("EV  (mean-value problem objective)", _f(result["EV"])))

    if math.isinf(eev):
        names = result["infeasible_scenarios"]
        shown = names[:8]
        more = "" if len(names) <= 8 else f" (+{len(names) - 8} more)"
        lines.append(_row("EEV (EV first stage over scenarios)", "+inf",
                          f"   infeasible in: {', '.join(shown)}{more}"))
        lines.append(_row(f"VSS = {formula} (sense={sense})", "+inf",
                          "   (mean-value first stage not usable everywhere)"))
        lines.append("=============================================")
        print("\n".join(lines))
        global_toc("VSS = +inf (mean-value first stage infeasible "
                   "in some scenarios)")
        return

    lines.append(_row("EEV (EV first stage over scenarios)", _f(eev)))

    vss, rp = result["VSS"], result["RP"]
    pct = ""
    if rp != 0 and math.isfinite(rp):
        pct = f"   ({100.0 * vss / abs(rp):.2f}% of |RP|)"
    lines.append(_row(f"VSS = {formula} (sense={sense})", _f(vss), pct))

    if result["vss_bracket"] is not None:
        vlo, vhi = result["vss_bracket"]
        lines.append(_row("    VSS bracket from RP bracket",
                          f"[{_f(vlo)}, {_f(vhi)}]"))

    lines.append("=============================================")
    print("\n".join(lines))
    global_toc(f"VSS = {_f(vss)} ({formula})")
