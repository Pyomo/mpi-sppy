###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Out-of-the-box (OOTB) policy-file validator.

Given a dated policy file, check it is well-formed and that its recommended
configurations make sense -- and, on demand, that they actually RUN -- using the
mpi-sppy examples as test models. See doc/designs/out_of_the_box_design.md (sec.
8). It does NOT assert "expected results" (there is no cheap oracle); instead it
records what happened and FLAGS two run failure modes for a human to review.

    python -m mpisppy.generic.ootb_validate <policy.json>          # layers 1+2
    python -m mpisppy.generic.ootb_validate <policy.json> --examples  # +real models
    python -m mpisppy.generic.ootb_validate <policy.json> --run     # +layer 3 runs

Three layers, fast to slow:

  1. STATIC (schema): JSON parses; required keys/types; every referenced flag is
     a real CLI option; spoke rungs / rho setters / DECOMPOSITION_FLAGS all line
     up with the actual generic_cylinders vocabulary.
  2. DECISION (recommend() only, no solves): on hand-built synthetic Facts (the
     CI-gating subset) and, with --examples, on probe-instantiated real models.
     Asserts EF-when-it-should, decompose-when-it-should, forced-decomp-wins,
     bundling validity, and no conflicting rho setters.
  3. RUN (--run; slow, needs a solver, NEVER a CI gate): actually run the
     recommended configs on the small examples and FLAG (a) an EF that misses a
     1% gap in ten minutes and (b) cylinders that max out on iterations.

The CI gate (mpisppy/tests/test_ootb_validate.py) runs only layers 1 +
2-synthetic on the shipped policy file(s) -- solver-free, fast.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass

import mpisppy.utils.config as config
from mpisppy.generic import parsing
from mpisppy.generic import out_of_the_box as ootb


# Validator settings (NOT policy): the two layer-3 auto-flag thresholds.
EF_GAP_TARGET = 0.01          # flag an EF that misses this MIP gap ...
EF_TIME_LIMIT_SEC = 600       # ... within this wall-clock budget (ten minutes)


# ---------------------------------------------------------------------------
# Result carriers
# ---------------------------------------------------------------------------


@dataclass
class Check:
    layer: str          # "static" | "decision"
    name: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict:
        return {"layer": self.layer, "name": self.name, "ok": self.ok,
                "detail": self.detail}


@dataclass
class RunRecord:
    example: str
    env: dict
    mode: str                       # "EF" | "decompose"
    returncode: int | None = None
    walltime: float | None = None
    objective: float | None = None
    rel_gap: float | None = None
    iterations: int | None = None
    flagged: bool = False
    flag_reason: str = ""
    detail: str = ""

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("example", "env", "mode", "returncode", "walltime", "objective",
                 "rel_gap", "iterations", "flagged", "flag_reason", "detail")}


# ---------------------------------------------------------------------------
# The example models the validator exercises (layers 2-examples and 3).
# ---------------------------------------------------------------------------


def _repo_root() -> str:
    # mpisppy/generic/ootb_validate.py -> repo root is three directories up.
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_models() -> list[dict]:
    root = _repo_root()
    return [
        {"name": "farmer", "kind": "2-stage continuous",
         "dir": os.path.join(root, "examples", "farmer"), "module": "farmer",
         "scens": {"num_scens": 6}},
        {"name": "sizes", "kind": "2-stage MIP",
         "dir": os.path.join(root, "examples", "sizes"), "module": "sizes",
         "scens": {"num_scens": 3}},
        {"name": "aircond", "kind": "multistage",
         "dir": os.path.join(root, "mpisppy", "tests", "examples"),
         "module": "aircond",
         "scens": {"branching_factors": [3, 2]}},
    ]


def _scen_cli(spec: dict) -> list:
    """The --num-scens / --branching-factors CLI args for a spec. Branching
    factors are passed as ONE space-joined token (the ListOf(int) domain parses
    a single string, so '--branching-factors 3 2' would leave '2' unparsed)."""
    s = spec["scens"]
    if "num_scens" in s:
        return ["--num-scens", str(s["num_scens"])]
    return ["--branching-factors", " ".join(str(b) for b in s["branching_factors"])]


def _child_env() -> dict:
    """Environment for subprocess runs with this (singleton-MPI) process's MPI
    variables scrubbed. Importing mpi-sppy initializes mpi4py.MPI here, so the
    parent IS an MPI process; leaving OMPI_/PMI_/... in the child's environment
    makes a fresh mpiexec think it is being relaunched inside a rank and fail."""
    return {k: v for k, v in os.environ.items()
            if not k.startswith(("OMPI_", "PMI_", "PMIX_", "MPIR_", "HYDRA_"))}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def valid_flags() -> set:
    """The authoritative set of real CLI flags, from the same declaration the
    driver uses (parsing.add_driver_args with no model module)."""
    ref = config.Config()
    parsing.add_driver_args(ref)
    return {"--" + name.replace("_", "-") for name in ref}


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# ---------------------------------------------------------------------------
# Layer 1: static (schema) checks
# ---------------------------------------------------------------------------


def validate_static(policy: dict) -> list:
    checks = []
    flags = valid_flags()

    def add(name, ok, detail=""):
        checks.append(Check("static", name, ok, detail))

    # required top-level keys
    required = ["schema_version", "policy_version", "ef_fallback", "solver",
                "hub", "spoke_ladder", "rank_allocation", "effort_scaling",
                "bundle_sizing", "option_categories", "additional_options",
                "suggestions"]
    missing = [k for k in required if k not in policy]
    add("required top-level keys present", not missing,
        f"missing: {missing}" if missing else "all present")
    if missing:
        return checks  # the rest assume the structure exists

    # ef_fallback numbers
    ef = policy["ef_fallback"]
    add("ef_fallback.min_ranks_for_decomposition is a positive int",
        isinstance(ef.get("min_ranks_for_decomposition"), int)
        and ef["min_ranks_for_decomposition"] >= 1,
        str(ef.get("min_ranks_for_decomposition")))
    for k in ("ef_if_num_scens_at_most", "ef_effort_budget", "ef_target_seconds"):
        add(f"ef_fallback.{k} is a positive number",
            _is_number(ef.get(k)) and ef[k] > 0, str(ef.get(k)))

    # solver
    sp = policy["solver"]
    pref = sp.get("preference_order")
    add("solver.preference_order is a non-empty list of strings",
        isinstance(pref, list) and len(pref) > 0
        and all(isinstance(s, str) for s in pref))
    for key in ("commercial", "qp_capable", "lp_mip_only_force_linearize_prox"):
        val = sp.get(key, [])
        add(f"solver.{key} is a subset of preference_order",
            isinstance(val, list) and set(val) <= set(pref or []),
            f"stray: {sorted(set(val) - set(pref or []))}")

    # spoke ladder
    ladder = policy["spoke_ladder"]
    rungs = ladder.get("rungs", [])
    rung_flags = [r.get("flag") for r in rungs]
    add("spoke_ladder.rungs flags are real CLI options",
        all(f in flags for f in rung_flags),
        f"unknown: {[f for f in rung_flags if f not in flags]}")
    add("spoke_ladder.rungs flags are decomposition (spoke) flags",
        all(f in ootb.DECOMPOSITION_FLAGS for f in rung_flags),
        f"not spokes: {[f for f in rung_flags if f not in ootb.DECOMPOSITION_FLAGS]}")
    add("spoke_ladder.rungs have unique priorities",
        len({r.get("priority") for r in rungs}) == len(rungs))
    add("spoke_ladder.rungs bounds are outer/inner",
        all(r.get("bound") in ("outer", "inner") for r in rungs))
    core = ladder.get("core_roster_min", {})
    add("spoke_ladder.core_roster_min has outer & inner counts",
        isinstance(core.get("outer"), int) and isinstance(core.get("inner"), int))
    add("spoke_ladder.max_cylinders is an int >= core roster size",
        isinstance(ladder.get("max_cylinders"), int)
        and ladder["max_cylinders"] >= 1 + core.get("outer", 0) + core.get("inner", 0))
    # the ladder must be able to satisfy the core roster
    avail = {"outer": 0, "inner": 0}
    for r in rungs:
        if r.get("bound") in avail:
            avail[r["bound"]] += 1
    need = {k: core.get(k, 0) for k in ("outer", "inner")}
    add("spoke_ladder can satisfy core_roster_min",
        avail["outer"] >= need["outer"] and avail["inner"] >= need["inner"],
        f"available {avail} vs needed {need}")

    # rank allocation
    ra = policy["rank_allocation"]
    add("rank_allocation.min_ranks_per_cylinder is a positive int",
        isinstance(ra.get("min_ranks_per_cylinder"), int)
        and ra["min_ranks_per_cylinder"] >= 1)
    add("rank_allocation.default_rank_ratio is a positive number",
        _is_number(ra.get("default_rank_ratio")) and ra["default_rank_ratio"] > 0)
    rr = ra.get("rank_ratios", {})
    add("rank_allocation.rank_ratios keys are real spoke flags",
        all(f in flags for f in rr), f"unknown: {[f for f in rr if f not in flags]}")
    add("rank_allocation.rank_ratios values are positive numbers",
        all(_is_number(v) and v > 0 for v in rr.values()))

    # effort scaling
    es = policy["effort_scaling"]
    for k in ("cont_coeff", "int_weight", "int_exponent", "int_nonant_coeff"):
        add(f"effort_scaling.{k} is a non-negative number",
            _is_number(es.get(k)) and es[k] >= 0, str(es.get(k)))
    add("effort_scaling.int_exponent >= 1 (integers at least linear)",
        _is_number(es.get("int_exponent")) and es["int_exponent"] >= 1)

    # bundle sizing
    bs = policy["bundle_sizing"]
    for k in ("min_scens_to_consider_bundling", "min_bundles_per_intra_rank",
              "base_max_hardness_vs_single_scenario",
              "plus_target_seconds_per_bundle", "plus_probe_solve_time_cap_seconds"):
        add(f"bundle_sizing.{k} is a positive number",
            _is_number(bs.get(k)) and bs[k] > 0, str(bs.get(k)))
    add("bundle_sizing.base_max_hardness_vs_single_scenario >= 1",
        _is_number(bs.get("base_max_hardness_vs_single_scenario"))
        and bs["base_max_hardness_vs_single_scenario"] >= 1)

    # option categories
    oc = policy.get("option_categories", {})
    for name, cat in oc.items():
        if name.startswith("_"):
            continue
        add(f"option_categories.{name}.flag is a real CLI option",
            cat.get("flag") in flags, str(cat.get("flag")))
        sup = cat.get("superseded_by", [])
        add(f"option_categories.{name}.superseded_by are real CLI options",
            isinstance(sup, list) and all(f in flags for f in sup),
            f"unknown: {[f for f in sup if f not in flags]}")
    # the rho_setter concern, if present, must list all rho setters (only one may
    # be active, so OOTB stacking a second would be a hard error).
    if "rho_setter" in oc:
        rho_setters = {"--grad-rho", "--sensi-rho", "--coeff-rho", "--sep-rho"}
        sup = set(oc["rho_setter"].get("superseded_by", []))
        add("option_categories.rho_setter.superseded_by lists every rho setter",
            rho_setters <= sup, f"missing: {sorted(rho_setters - sup)}")

    # additional options catch-all
    ao = policy.get("additional_options", {}).get("options", [])
    for opt in ao:
        add(f"additional_options flag {opt.get('flag')} is a real CLI option",
            opt.get("flag") in flags, str(opt.get("flag")))
        sup = opt.get("superseded_by", [opt.get("flag")])
        add(f"additional_options {opt.get('flag')} superseded_by are real options",
            all(f in flags for f in sup))

    # suggestions: disabled names real generators
    gen_names = {g.__name__ for g in ootb.SUGGESTION_GENERATORS}
    disabled = policy.get("suggestions", {}).get("disabled", [])
    add("suggestions.disabled names real generators",
        isinstance(disabled, list) and all(d in gen_names for d in disabled),
        f"unknown: {[d for d in disabled if d not in gen_names]}")

    # _cold_start_guess entries name real keys in their block
    for block_name, block in policy.items():
        if not isinstance(block, dict) or "_cold_start_guess" not in block:
            continue
        guesses = block["_cold_start_guess"]
        bad = [g for g in guesses if g not in block]
        add(f"{block_name}._cold_start_guess entries name real keys", not bad,
            f"stray: {bad}")

    # DECOMPOSITION_FLAGS line up with the real vocabulary
    bad = sorted(f for f in ootb.DECOMPOSITION_FLAGS if f not in flags)
    add("out_of_the_box.DECOMPOSITION_FLAGS are all real CLI options", not bad,
        f"unknown: {bad}")

    return checks


# ---------------------------------------------------------------------------
# Layer 2: decision checks on hand-built synthetic Facts (the CI-gating subset)
# ---------------------------------------------------------------------------


def _facts(**kw):
    """Build a Facts with sensible defaults for the synthetic decision checks."""
    base = dict(module_name="synthetic", num_ranks=3,
                available_solvers={"gurobi_persistent"}, num_scens=10,
                effort="base", vars_int=0, vars_cont=10, nonants_total=3,
                nonants_int=0)
    base.update(kw)
    return ootb.Facts(**base)


def _bundle_arg(decision):
    for a in decision.args:
        if a.flag == "--scenarios-per-bundle":
            return int(a.value)
    return None


def _rho_setter_args(decision):
    setters = {"--grad-rho", "--sensi-rho", "--coeff-rho", "--sep-rho"}
    return [a.flag for a in decision.args if a.flag in setters]


def validate_decisions_synthetic(policy: dict) -> list:
    checks = []
    minr = policy["ef_fallback"]["min_ranks_for_decomposition"]

    def add(name, ok, detail=""):
        checks.append(Check("decision", name, ok, detail))

    # EF when below the rank floor
    d = ootb.recommend(_facts(num_ranks=minr - 1), policy)
    add("below rank floor -> EF", d.run_ef and d.ef_reason == "min_ranks",
        f"run_ef={d.run_ef} reason={d.ef_reason}")

    # EF when the whole problem is tiny (base effort gate)
    d = ootb.recommend(_facts(num_ranks=minr, num_scens=2,
                              vars_cont=1, vars_int=0), policy)
    add("tiny problem above rank floor -> EF", d.run_ef,
        f"run_ef={d.run_ef} reason={d.ef_reason}")

    # NOT EF when the whole problem is large / integer-heavy
    d = ootb.recommend(_facts(num_ranks=6, num_scens=1000,
                              vars_cont=500, vars_int=200, nonants_int=50), policy)
    add("large integer-heavy problem -> decompose (not EF)", not d.run_ef,
        f"run_ef={d.run_ef} reason={d.ef_reason}")

    # forced decomposition wins even on a tiny problem
    d = ootb.recommend(_facts(num_ranks=minr, num_scens=2, vars_cont=1,
                              user_flags={"--lagrangian", "--xhatshuffle"}), policy)
    add("user-forced decomposition (>= rank floor) -> never EF", not d.run_ef,
        f"run_ef={d.run_ef} reason={d.ef_reason}")

    # bundling validity: when OOTB bundles, spb divides num_scens & #bundles>=ranks
    d = ootb.recommend(_facts(num_ranks=minr, num_scens=120, vars_cont=5,
                              user_flags={"--lagrangian"}), policy)
    spb = _bundle_arg(d)
    if spb is None:
        add("bundling validity (no bundle chosen here)", True, "no --scenarios-per-bundle")
    else:
        ok = (120 % spb == 0) and (120 // spb >= d.intra_ranks)
        add("bundling: spb divides num_scens and #bundles >= intra_ranks", ok,
            f"spb={spb}, #bundles={120 // spb}, intra_ranks={d.intra_ranks}")

    # minus tier never bundles (no size profile)
    d = ootb.recommend(_facts(effort="minus", vars_int=None, vars_cont=None,
                              nonants_total=None, nonants_int=None,
                              num_ranks=minr, num_scens=120,
                              user_flags={"--lagrangian"}), policy)
    add("minus tier never bundles", _bundle_arg(d) is None,
        f"spb={_bundle_arg(d)}")

    # no conflicting rho setters: the user's rho setter must suppress OOTB's
    d = ootb.recommend(_facts(num_ranks=6, num_scens=10, vars_int=5,
                              vars_cont=10, nonants_int=2,
                              user_flags={"--lagrangian", "--sensi-rho"}), policy)
    add("user rho setter suppresses OOTB's (<=0 OOTB rho setters)",
        len(_rho_setter_args(d)) == 0, f"OOTB rho setters: {_rho_setter_args(d)}")

    # OOTB adds at most one rho setter on its own
    d = ootb.recommend(_facts(num_ranks=6, num_scens=10, vars_int=5, vars_cont=10,
                              user_flags={"--lagrangian"}), policy)
    add("OOTB adds at most one rho setter", len(_rho_setter_args(d)) <= 1,
        f"OOTB rho setters: {_rho_setter_args(d)}")

    return checks


# ---------------------------------------------------------------------------
# Layer 2: decision checks on real, probe-instantiated example models (out of CI)
# ---------------------------------------------------------------------------


def _load_example_module(spec: dict):
    if spec["dir"] not in sys.path:
        sys.path.insert(0, spec["dir"])
    return importlib.import_module(spec["module"])


def _example_cfg(spec: dict, module):
    """Build a fully-declared cfg for an example without parsing a command line,
    then set its scenario count / branching factors from spec["scen_args"]."""
    cfg = config.Config()
    parsing.add_driver_args(cfg, module)
    s = spec["scens"]
    if "num_scens" in s:
        cfg.num_scens = s["num_scens"]
    else:
        cfg.branching_factors = list(s["branching_factors"])
    return cfg


def validate_decisions_examples(policy: dict) -> list:
    checks = []
    for spec in example_models():
        name = spec["name"]
        try:
            module = _load_example_module(spec)
            cfg = _example_cfg(spec, module)
        except Exception as e:                       # noqa: BLE001
            checks.append(Check("decision", f"example {name}: set up", False,
                                f"{type(e).__name__}: {e}"))
            continue
        for nranks in (1, 3, 8):
            try:
                cfg.inspect_only = str(nranks)       # plan as if nranks (no MPI)
                facts = ootb.gather_facts(module, cfg, "base", policy)
                d = ootb.recommend(facts, policy)
            except Exception as e:                   # noqa: BLE001
                checks.append(Check("decision",
                                    f"example {name} @ {nranks} ranks: recommend",
                                    False, f"{type(e).__name__}: {e}"))
                continue
            # the probe must have populated a size profile
            prof_ok = facts.vars_cont is not None and facts.vars_int is not None
            checks.append(Check("decision",
                                f"example {name} @ {nranks} ranks: probe size profile",
                                prof_ok,
                                f"int={facts.vars_int} cont={facts.vars_cont} "
                                f"nonants={facts.nonants_total}"))
            # internal consistency of the recommendation
            spb = _bundle_arg(d)
            bundle_ok = spb is None or (facts.num_scens % spb == 0
                                        and facts.num_scens // spb >= d.intra_ranks)
            checks.append(Check("decision",
                                f"example {name} @ {nranks} ranks: bundling valid",
                                bundle_ok, f"spb={spb} intra={d.intra_ranks} "
                                f"num_scens={facts.num_scens}"))
            checks.append(Check("decision",
                                f"example {name} @ {nranks} ranks: <=1 rho setter",
                                len(_rho_setter_args(d)) <= 1,
                                str(_rho_setter_args(d))))
    return checks


# ---------------------------------------------------------------------------
# Layer 3: actually run the recommended configs (--run; slow; never a CI gate)
# ---------------------------------------------------------------------------


_OBJ_RE = re.compile(r"EF objective:\s*([-\d.eE+]+)")
# A termination stats row, tolerating global_toc's leading "[   0.86] " prefix:
#   [   0.86]    99    X    -124309.6133    -123177.9852    0.910%    1131.6281
_STAT_RE = re.compile(r"^(?:\[[^\]]*\]\s*)?(\d+)\s+.*?([\d.]+)%\s+[-\d.eE+]+\s*$")


def _parse_decompose(out: str):
    """Return (converged, iterations, rel_gap) parsed from a decomposition run's
    stdout. 'Terminating based on inter-cylinder' (or 'Cylinder convergence')
    marks gap convergence; otherwise the run exhausted its iterations."""
    converged = ("Terminating based on inter-cylinder" in out
                 or "Cylinder convergence" in out)
    iters = rel_gap = None
    seen_stats = False
    for line in out.splitlines():
        if "Statistics at termination" in line:
            seen_stats = True
            continue
        if seen_stats:
            m = _STAT_RE.match(line)
            if m:
                iters = int(m.group(1))
                rel_gap = float(m.group(2)) / 100.0
    return converged, iters, rel_gap


def _run_one(spec, mode, nranks, extra_args, timeout):
    """Run one example via generic_cylinders in a subprocess and time it."""
    if mode == "EF":
        cmd = [sys.executable, "-m", "mpisppy.generic_cylinders"]
    else:
        cmd = ["mpiexec", "-np", str(nranks), sys.executable, "-m", "mpi4py",
               "-m", "mpisppy.generic_cylinders"]
    cmd += ["--module-name", spec["module"]] + _scen_cli(spec) + extra_args
    rec = RunRecord(example=spec["name"],
                    env={"ranks": nranks, "mode": mode}, mode=mode)
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=spec["dir"], capture_output=True,
                              text=True, timeout=timeout, env=_child_env())
        rec.returncode = proc.returncode
        out = proc.stdout + "\n" + proc.stderr
        rec.detail = (proc.stderr.strip().splitlines() or [""])[-1]
    except subprocess.TimeoutExpired:
        rec.returncode = None
        rec.walltime = time.time() - start
        rec.flagged = True
        rec.flag_reason = f"timed out after {timeout}s"
        return rec
    rec.walltime = time.time() - start
    return rec, out


def validate_runs(policy_path: str, *, ef_time_limit=EF_TIME_LIMIT_SEC,
                  ef_gap=EF_GAP_TARGET) -> list:
    """Run two configurations per example and flag the two failure modes."""
    records = []
    pol_arg = ["--out-of-the-box", policy_path] if policy_path else ["--out-of-the-box"]
    for spec in example_models():
        # (a) EF: few ranks -> OOTB picks the EF. Flag if it misses the gap/time.
        res = _run_one(spec, "EF", 1, ["--out-of-the-box-minus"] + (
            ["--EF-mipgap", str(ef_gap)] if spec["kind"] == "2-stage MIP" else []),
            timeout=ef_time_limit)
        if isinstance(res, RunRecord):           # timed out
            res.flag_reason = f"EF missed a {ef_gap:.0%} gap within {ef_time_limit}s"
            records.append(res)
        else:
            rec, out = res
            m = _OBJ_RE.search(out)
            rec.objective = float(m.group(1)) if m else None
            if rec.returncode != 0 or rec.objective is None:
                rec.flagged = True
                rec.flag_reason = "EF run did not complete cleanly"
            records.append(rec)

        # (b) decomposition (forced via --lagrangian, to exercise that path even
        # on a small problem). Flag if it maxes out iterations instead of
        # converging on the inter-cylinder gap.
        res = _run_one(spec, "decompose", 3, pol_arg + ["--lagrangian"],
                       timeout=ef_time_limit)
        if isinstance(res, RunRecord):           # timed out
            records.append(res)
        else:
            rec, out = res
            converged, iters, rel_gap = _parse_decompose(out)
            rec.iterations, rec.rel_gap = iters, rel_gap
            if rec.returncode == 0 and not converged:
                rec.flagged = True
                rec.flag_reason = "cylinders maxed out on iterations (no gap convergence)"
            elif rec.returncode != 0:
                rec.flagged = True
                rec.flag_reason = "decomposition run did not complete cleanly"
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Orchestration + report
# ---------------------------------------------------------------------------


def run_validation(policy_path: str, *, examples=False, run=False) -> dict:
    """Run the requested layers and return a machine-readable report dict."""
    report = {"policy_file": policy_path, "policy_version": None,
              "checks": [], "runs": [], "ok": True}
    try:
        policy = ootb.load_policy(policy_path or None)
    except Exception as e:                            # noqa: BLE001
        report["checks"].append(
            Check("static", "policy file parses", False,
                  f"{type(e).__name__}: {e}").as_dict())
        report["ok"] = False
        return report
    report["policy_version"] = policy.get("policy_version")

    checks = validate_static(policy) + validate_decisions_synthetic(policy)
    if examples:
        checks += validate_decisions_examples(policy)
    report["checks"] = [c.as_dict() for c in checks]
    report["ok"] = all(c.ok for c in checks)

    if run:
        records = validate_runs(policy_path)
        report["runs"] = [r.as_dict() for r in records]

    return report


def format_report(report: dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append(f"OOTB policy validation: {report['policy_file'] or '(default)'}")
    lines.append(f"policy_version: {report['policy_version']}")
    lines.append("=" * 72)

    by_layer = {}
    for c in report["checks"]:
        by_layer.setdefault(c["layer"], []).append(c)
    for layer in ("static", "decision"):
        layer_checks = by_layer.get(layer, [])
        if not layer_checks:
            continue
        npass = sum(1 for c in layer_checks if c["ok"])
        lines.append(f"\n[{layer}] {npass}/{len(layer_checks)} passed")
        for c in layer_checks:
            mark = "PASS" if c["ok"] else "FAIL"
            suffix = f"  ({c['detail']})" if c["detail"] else ""
            if not c["ok"] or c["detail"]:
                lines.append(f"  {mark}  {c['name']}{suffix}")

    failures = [c for c in report["checks"] if not c["ok"]]
    if failures:
        lines.append(f"\n*** {len(failures)} CHECK FAILURE(S) ***")
        for c in failures:
            lines.append(f"  - [{c['layer']}] {c['name']}: {c['detail']}")

    if report["runs"]:
        lines.append("\n" + "-" * 72)
        lines.append("Layer 3 runs (recorded; not auto-judged except the flags):")
        for r in report["runs"]:
            wt = f"{r['walltime']:.1f}s" if r["walltime"] is not None else "?"
            extra = []
            if r["objective"] is not None:
                extra.append(f"obj={r['objective']:.4g}")
            if r["rel_gap"] is not None:
                extra.append(f"gap={r['rel_gap']:.2%}")
            if r["iterations"] is not None:
                extra.append(f"iters={r['iterations']}")
            lines.append(f"  {r['example']:8} {r['mode']:10} rc={r['returncode']} "
                         f"{wt:>8}  {' '.join(extra)}")
        flagged = [r for r in report["runs"] if r["flagged"]]
        lines.append("\n*** FLAGGED RUNS (review these) ***" if flagged
                     else "\nNo runs flagged.")
        for r in flagged:
            lines.append(f"  !! {r['example']} [{r['mode']}]: {r['flag_reason']}")

    lines.append("\n" + ("OVERALL: PASS" if report["ok"] else "OVERALL: FAIL"))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m mpisppy.generic.ootb_validate",
        description="Validate an OOTB policy file (see design doc sec. 8).")
    p.add_argument("policy", nargs="?", default="",
                   help="policy file path (default: the shipped default policy)")
    p.add_argument("--examples", action="store_true",
                   help="also run decision checks on probe-instantiated real "
                        "example models (needs a solver to instantiate; not CI)")
    p.add_argument("--run", action="store_true",
                   help="also actually RUN the recommended configs (layer 3; "
                        "slow, needs a solver and mpiexec; never a CI gate)")
    p.add_argument("--json", metavar="PATH", default=None,
                   help="write the machine-readable report to PATH")
    args = p.parse_args(argv)

    report = run_validation(args.policy, examples=args.examples, run=args.run)
    print(format_report(report))
    if args.json:
        with open(args.json, "w") as fp:
            json.dump(report, fp, indent=2)
        print(f"\n[wrote JSON report to {args.json}]")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
