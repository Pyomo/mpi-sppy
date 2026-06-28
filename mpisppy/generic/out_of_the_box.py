###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Out-of-the-box (OOTB) auto-configuration -- SKETCH, not yet wired.

This module is the thin Python *interpreter* described in
``doc/designs/out_of_the_box_design.md`` (sec. 5/5.1). It turns:

  * a set of FACTS about the environment (MPI ranks, installed solvers,
    optional cores/memory) and the model (scenario count, stage structure), and
  * a dated declarative POLICY file (``ootb_policies/ootb_policy_*.json``)

into a recommended mpi-sppy configuration, a human-readable reason for every
choice, a post-run "Suggestions" list, and the equivalent explicit command
line.

Design commitments this sketch embodies:
  * USER OPTIONS ALWAYS WIN -- ``choose()`` defers to anything in
    ``facts.user_flags`` (requirement 0).
  * The decision logic is plain, ordered, readable Python -- no rules engine,
    no neural net. The *numbers* live in the policy file, not here.
  * Every decision records WHY, so transparency (requirement 4) falls out.

What is COMPLETE here (study this): ``recommend()`` and its helpers, the named
predicates, and command-line assembly -- all pure functions of (facts, policy),
so you can hand-build a ``Facts`` and call ``recommend()`` to see the choices.

What is STUBBED here (the wiring, deliberately deferred): the environment/model
probing in ``gather_facts()`` and applying the ``Decision`` back onto a
``Config`` object. These touch MPI/Pyomo/Config and are marked ``TODO``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass
class Facts:
    """Everything OOTB is allowed to look at (besides the policy)."""

    module_name: str
    num_ranks: int
    available_solvers: set[str]      # names for which SolverFactory(.).available()
    num_scens: int
    # probe size profile -- populated at the base/plus tiers, None at minus
    vars_int: int | None = None       # integer/binary vars per scenario
    vars_cont: int | None = None      # continuous vars per scenario
    nonants_total: int | None = None  # first-stage (nonanticipative) vars
    nonants_int: int | None = None    # integer nonants
    multistage: bool = False
    num_cores: int | None = None     # best effort; may be None
    memory_gb: float | None = None   # best effort; may be None
    under_slurm: bool = False
    user_flags: set[str] = field(default_factory=set)  # CLI flags the user set


@dataclass
class ChosenArg:
    flag: str                # e.g. "--lagrangian" or "--solver-name"
    value: str | None        # None => boolean flag
    reason: str


@dataclass
class Decision:
    run_ef: bool = False
    ef_reason: str | None = None          # "min_ranks" | "few_scens" | "user"
    chosen_solver: str | None = None
    num_cylinders: int = 1                # hub + spokes actually configured
    intra_ranks: int = 1                  # widest cylinder's rank count
    rank_split: dict = field(default_factory=dict)   # cylinder -> ranks (flex)
    args: list[ChosenArg] = field(default_factory=list)   # what OOTB added
    notes: list[str] = field(default_factory=list)        # full reasoning trace
    suggestions: list[str] = field(default_factory=list)  # filled AFTER the run

    def command_line(self, num_ranks: int) -> str:
        """The explicit command the OOTB choices are equivalent to (req. 4)."""
        parts = [
            f"mpiexec -np {num_ranks} python -m mpi4py -m mpisppy.generic_cylinders",
        ]
        for a in self.args:
            parts.append(a.flag if a.value is None else f"{a.flag} {a.value}")
        return " ".join(parts)


# Flags that mean "the user wants a decomposition": any wired spoke or a
# non-default hub. If the user set one of these AND has >= the rank floor, OOTB
# must NOT substitute the EF, even for a small problem (requirement 0). This is
# the generic_cylinders vocabulary -- a fact, not a focus preference -- so it
# lives in code, and the validator checks it against the actual CLI flags.
DECOMPOSITION_FLAGS = frozenset({
    # wired spokes
    "--lagrangian", "--fwph", "--ph-dual", "--ph-xfeas-spoke", "--relaxed-ph",
    "--subgradient", "--reduced-costs", "--xhatshuffle", "--xhatxbar",
    "--xhatlshaped",
    # non-default hubs
    "--APH", "--subgradient-hub", "--fwph-hub", "--ph-primal-hub",
    "--lshaped-hub", "--cg-hub", "--dualcg-hub",
})


# ---------------------------------------------------------------------------
# The interpreter: pure (facts, policy) -> Decision  [COMPLETE]
# ---------------------------------------------------------------------------


def recommend(facts: Facts, policy: dict) -> Decision:
    """Apply the policy to the facts. Ordered, each step records its reason."""
    d = Decision()

    def choose(flag: str, value: str | None, reason: str) -> bool:
        # requirement 0: never override an explicit user choice.
        if flag in facts.user_flags:
            d.notes.append(f"{flag}: kept user's value (OOTB defers)")
            return False
        d.args.append(ChosenArg(flag, value, reason))
        shown = flag if value is None else f"{flag} {value}"
        d.notes.append(f"{shown}: {reason}")
        return True

    # --- step 1: solver -----------------------------------------------------
    sp = policy["solver"]
    if "--solver-name" in facts.user_flags:
        d.chosen_solver = None  # user's solver stands; we don't know its name here
        d.notes.append("--solver-name: kept user's value (OOTB defers)")
    else:
        for name in sp["preference_order"]:
            if name in facts.available_solvers:
                d.chosen_solver = name
                choose("--solver-name", name,
                       f"first available in preference order ({name})")
                break
        if d.chosen_solver is None:
            d.notes.append("WARNING: no known solver detected; user must supply one")

    # LP/MIP-only solvers cannot take the quadratic PH prox term.
    if d.chosen_solver in sp["lp_mip_only_force_linearize_prox"]:
        choose("--linearize-proximal-terms", None,
               f"{d.chosen_solver} is LP/MIP-only; prox must be linearized")

    # --- step 2: EF gate ----------------------------------------------------
    # Reuses the bundle effort() model on the WHOLE problem (all scenarios as one
    # model) vs an ABSOLUTE EF budget. base: effort(num_scens) <= ef_effort_budget;
    # plus would use ef_target_seconds via a measured t1 (stubbed); minus (no
    # size profile): the count rule.
    ef = policy["ef_fallback"]
    have_profile = facts.vars_cont is not None or facts.vars_int is not None
    if "--EF" in facts.user_flags:
        d.run_ef, d.ef_reason = True, "user"
        d.notes.append("--EF: user requested")
    elif facts.num_ranks < ef["min_ranks_for_decomposition"]:
        d.run_ef, d.ef_reason = True, "min_ranks"
        choose("--EF", None,
               f"only {facts.num_ranks} ranks; decomposition needs "
               f">= {ef['min_ranks_for_decomposition']}")
    elif facts.user_flags & DECOMPOSITION_FLAGS:
        # User explicitly asked for a decomposition and has enough ranks, so we
        # never substitute the EF -- even for a small problem (requirement 0).
        forced = ", ".join(sorted(facts.user_flags & DECOMPOSITION_FLAGS))
        d.notes.append(f"EF gate skipped: user requested decomposition ({forced}) "
                       f"with >= {ef['min_ranks_for_decomposition']} ranks")
    elif have_profile:
        # base/plus: EF when the whole monolith is within the absolute EF budget.
        whole = _effort(facts.num_scens, facts, policy["effort_scaling"])
        if whole <= ef["ef_effort_budget"]:
            d.run_ef, d.ef_reason = True, "small_effort"
            choose("--EF", None,
                   f"whole-problem effort {whole:.0f} <= EF budget "
                   f"{ef['ef_effort_budget']}")
    elif facts.num_scens <= ef["ef_if_num_scens_at_most"]:
        # minus: no profile -> count rule
        d.run_ef, d.ef_reason = True, "few_scens"
        choose("--EF", None,
               f"only {facts.num_scens} scenarios; too few to decompose")

    if d.run_ef:
        # EF bypasses the hub/spoke system entirely; skip spokes + bundling.
        d.num_cylinders = 1
        return d

    # --- step 3: spoke roster -- small core, widened by ranks --------------
    # Take the minimal core (>=1 outer + >=1 inner). Add further ladder rungs
    # only while each cylinder keeps >= min_ranks_per_cylinder ranks (coarse,
    # uniform gate) and we stay <= max_cylinders -- prefer giving cylinders
    # width over piling on weaker spokes (6 ranks -> 3 cylinders, not 6).
    ladder = policy["spoke_ladder"]
    max_cyl = ladder["max_cylinders"]
    min_rpc = policy["rank_allocation"]["min_ranks_per_cylinder"]
    all_rungs = sorted(ladder["rungs"], key=lambda r: r["priority"])
    need = dict(ladder["core_roster_min"])       # e.g. {"outer": 1, "inner": 1}
    chosen = []
    for r in all_rungs:                           # minimal core
        if need.get(r["bound"], 0) > 0:
            chosen.append(r)
            need[r["bound"]] -= 1
    for r in all_rungs:                           # widen-aware additions
        if r in chosen:
            continue
        cyl_if_added = 2 + len(chosen)            # hub + chosen + this rung
        if cyl_if_added > max_cyl or facts.num_ranks // cyl_if_added < min_rpc:
            break
        chosen.append(r)
    for r in chosen:
        choose(r["flag"], None, f"spoke ({r['bound']}, priority {r['priority']})")
    d.num_cylinders = 1 + len(chosen)

    # --- step 4: rank allocation -- UNBALANCED by per-cylinder ratio --------
    # Split ranks across cylinders by ratio (flex-ranks), not uniformly; xhat
    # cylinders are cheaper so get a smaller share. CRUDE cold-start: the real
    # split depends on subproblem solve cost (a plus-tier refinement).
    ra = policy["rank_allocation"]
    ratios = {"(hub)": ra["default_rank_ratio"]}
    for r in chosen:
        ratios[r["flag"]] = ra["rank_ratios"].get(r["flag"], ra["default_rank_ratio"])
    d.rank_split = _allocate_ranks(facts.num_ranks, ratios)
    d.intra_ranks = max(d.rank_split.values())    # widest cylinder governs bundling
    d.notes.append("rank split (flex-ranks, crude cold-start): "
                   + ", ".join(f"{k}={v}" for k, v in d.rank_split.items()))
    # TODO(wiring): emit the flex-ranks per-cylinder rank flags from rank_split.

    # --- step 5: proper bundling -- how BIG? (design "Bundle sizing") -------
    # minus CANNOT bundle: with no size profile there is no safe way to size a
    # bundle, so the minus tier always runs unbundled.
    bs = policy["bundle_sizing"]
    if not have_profile:        # have_profile computed in the EF-gate step above
        d.notes.append("bundling: minus tier (no size profile); running unbundled")
    elif facts.num_scens >= bs["min_scens_to_consider_bundling"]:
        b_min = max(d.intra_ranks,
                    bs["min_bundles_per_intra_rank"] * d.intra_ranks)
        spb = _pick_spb_by_effort(
            facts.num_scens, b_min, facts, policy["effort_scaling"],
            bs["base_max_hardness_vs_single_scenario"])
        if spb is not None:
            nb = facts.num_scens // spb
            choose("--scenarios-per-bundle", str(spb),
                   f"{facts.num_scens} scenarios -> {nb} bundles of {spb} "
                   f"(effort <= {bs['base_max_hardness_vs_single_scenario']}x a "
                   f"single scenario, >= {b_min} bundles)")
        else:
            d.notes.append(
                f"bundling: no divisor of {facts.num_scens} qualifies "
                f"(>= {b_min} bundles within budget); running unbundled"
            )

    # --- step 6: extra options by concern; any superseded_by flag defers ----
    # Per-concern override: OOTB backs off a whole concern (e.g. its rho setter)
    # if the user set ANY equivalent flag, not just the identical one -- mp-sppy
    # allows only one rho setter, so stacking would be a hard error.
    for name, cat in policy.get("option_categories", {}).items():
        if name.startswith("_"):
            continue
        if any(f in facts.user_flags for f in cat.get("superseded_by", [cat["flag"]])):
            d.notes.append(f"{name}: superseded by a user option; OOTB defers")
        else:
            choose(cat["flag"], cat.get("value"), f"policy option ({name})")
    # catch-all: per-flag override (superseded_by defaults to the flag itself)
    for opt in policy.get("additional_options", {}).get("options", []):
        if any(f in facts.user_flags for f in opt.get("superseded_by", [opt["flag"]])):
            d.notes.append(f"{opt['flag']}: superseded by a user option; OOTB defers")
        else:
            choose(opt["flag"], opt.get("value"), "policy additional option")

    return d


def _effort(spb: int, facts: Facts, scaling: dict) -> float:
    """Modeled solve effort of a bundle of `spb` scenarios, from the probe size
    profile (facts.vars_cont/vars_int/nonants_int) and the policy effort_scaling
    shape. Continuous ~linear, integers superlinear (int_exponent > 1), integer
    nonants a fixed per-bundle coupling cost. See design "Bundle sizing"."""
    cont = (facts.vars_cont or 0) * spb
    nint = (facts.vars_int or 0) * spb
    return (scaling["cont_coeff"] * cont
            + scaling["int_weight"] * (nint ** scaling["int_exponent"])
            + scaling["int_nonant_coeff"] * (facts.nonants_int or 0))


def _pick_spb_by_effort(num_scens: int, min_bundles: int, facts: Facts,
                        scaling: dict, max_hardness: float) -> int | None:
    """base/plus sizer: the LARGEST scenarios_per_bundle that divides num_scens,
    leaves >= min_bundles bundles, and keeps a bundle's modeled effort within
    `max_hardness` x a single scenario (the relative, unit-free budget). The plus
    tier feeds the same function a time-calibrated effective hardness. Returns
    None when nothing past the degenerate spb==1 qualifies."""
    e1 = _effort(1, facts, scaling)
    best = None
    for spb in range(2, num_scens + 1):           # spb==1 is "no bundling"
        if num_scens % spb or num_scens // spb < min_bundles:
            continue
        if e1 > 0 and _effort(spb, facts, scaling) / e1 > max_hardness:
            continue
        if best is None or spb > best:            # largest = max amortization
            best = spb
    return best


def _allocate_ranks(total: int, ratios: dict) -> dict:
    """Split `total` ranks across cylinders by ratio (flex-ranks), flooring at 1
    rank each and distributing the remainder by largest fractional remainder.
    CRUDE cold-start; the real split depends on subproblem solve cost (design
    'Rank allocation')."""
    names = list(ratios)
    if total <= len(names):
        return {nm: 1 for nm in names}            # too few to weight; 1 each
    extra = total - len(names)                     # ranks above the 1-each floor
    s = sum(ratios.values()) or 1.0
    raw = {nm: extra * ratios[nm] / s for nm in names}
    alloc = {nm: 1 + int(raw[nm]) for nm in names}
    leftover = total - sum(alloc.values())
    for nm in sorted(names, key=lambda k: raw[k] - int(raw[k]), reverse=True)[:leftover]:
        alloc[nm] += 1
    return alloc


# ---------------------------------------------------------------------------
# Suggestions -- MOSTLY COMPUTED  [COMPLETE]
#   Each generator inspects facts/decision/run-outcome and COMPUTES a message
#   (with live values) or returns None. The prose lives here in code; the policy
#   only toggles (suggestions.disabled) or tunes them. This is the diagnostics
#   layer -- deliberately distinct from the data-driven DECISIONS (design 5).
#   Generators run in list order (priority); add outcome-based ones freely --
#   they receive the post-run `outcome`.
# ---------------------------------------------------------------------------


def _sg_ran_ef_few_ranks(d, facts, policy, outcome):
    if d.run_ef and d.ef_reason == "min_ranks":
        need = policy["ef_fallback"]["min_ranks_for_decomposition"]
        return (f"Ran the monolithic EF because only {facts.num_ranks} MPI "
                f"rank(s) were available; with >= {need} ranks OOTB would "
                f"decompose (hub + bound spokes).")
    return None


def _sg_no_persistent_solver(d, facts, policy, outcome):
    s = d.chosen_solver
    if s is not None and not s.endswith("_persistent"):
        return (f"Chosen solver '{s}' has no persistent interface available; "
                f"'{s}_persistent' would warm-start subproblems and is usually "
                f"much faster for PH.")
    return None


def _sg_linearized_prox(d, facts, policy, outcome):
    if d.chosen_solver in policy["solver"]["lp_mip_only_force_linearize_prox"]:
        return (f"'{d.chosen_solver}' is LP/MIP-only, so the PH prox is being "
                f"linearized; a QP-capable solver (gurobi/cplex/xpress, or ipopt "
                f"for continuous models) avoids the approximation.")
    return None


def _sg_more_ranks(d, facts, policy, outcome):
    cap = policy["spoke_ladder"]["max_cylinders"]
    if not d.run_ef and d.num_cylinders < cap:
        return (f"Only {d.num_cylinders} cylinders configured (cap {cap}); more "
                f"MPI ranks would add bound-tightening spokes or intra-cylinder "
                f"parallelism.")
    return None


def _sg_from_outcome(d, facts, policy, outcome):
    # Example of an outcome-based (post-run) computed suggestion; inert until the
    # run captures an outcome (None in the current sketch).
    if outcome and outcome.get("converged") is False:
        return (f"PH stopped at {outcome.get('iterations')} iterations with a "
                f"{outcome.get('rel_gap', 0):.0%} gap; consider raising "
                f"--max-iterations or adding a tighter bound spoke.")
    return None


# generators in priority order (lower first)
SUGGESTION_GENERATORS = [
    _sg_ran_ef_few_ranks,
    _sg_no_persistent_solver,
    _sg_linearized_prox,
    _sg_more_ranks,
    _sg_from_outcome,
]


def make_suggestions(d: Decision, facts: Facts, policy: dict,
                     outcome: dict | None = None) -> list[str]:
    """Build the post-run "Suggestions" list (req. 4) by running the computed
    generators in priority order, skipping any named in suggestions.disabled.
    Called AFTER the run so generators may use `outcome` (convergence, gap,
    iters, time)."""
    disabled = set(policy.get("suggestions", {}).get("disabled", []))
    out = []
    for gen in SUGGESTION_GENERATORS:
        if gen.__name__ in disabled:
            continue
        msg = gen(d, facts, policy, outcome)
        if msg:
            out.append(msg)
    return out


# ---------------------------------------------------------------------------
# Policy loading  [COMPLETE]
# ---------------------------------------------------------------------------


def _policies_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "ootb_policies")


def load_policy(policy_file: str | None = None) -> dict:
    """Load a policy. Default: newest dated file in ootb_policies/. The dated
    filename sorts lexically by ISO date, so max() is the newest."""
    if policy_file is None:
        d = _policies_dir()
        candidates = sorted(f for f in os.listdir(d)
                            if f.startswith("ootb_policy_") and f.endswith(".json"))
        if not candidates:
            raise FileNotFoundError(f"no OOTB policy files in {d}")
        policy_file = os.path.join(d, candidates[-1])
    with open(policy_file) as fp:
        return json.load(fp)


# ---------------------------------------------------------------------------
# Environment/model probing + apply-to-Config  [STUBBED -- the wiring]
# ---------------------------------------------------------------------------


def _detect_num_ranks() -> int:
    from mpisppy import MPI
    return MPI.COMM_WORLD.Get_size()


def _inspect_ranks(cfg) -> int:
    """Ranks to PLAN for. With --inspect-only N, use N so HPC users can get a
    recommended command line for a target job size WITHOUT launching that many
    ranks; otherwise the actual detected size. (N rides on --inspect-only, the
    only flag that carries it; everything else -- solvers, model size -- still
    comes from the real, possibly small, session.)"""
    io = cfg.get("inspect_only", None)
    if io not in (None, "", "detected"):
        return int(io)
    return _detect_num_ranks()


def _detect_available_solvers(candidates: list[str]) -> set[str]:
    import pyomo.environ as pyo
    found = set()
    for name in candidates:
        try:
            if pyo.SolverFactory(name).available(exception_flag=False):
                found.add(name)
        except Exception:
            pass
    return found


def _detect_num_scens(module, cfg) -> int:
    # TODO(wiring): mirror mpisppy/generic/parsing.py::name_lists --
    #   cfg.num_scens, else prod(cfg.branching_factors),
    #   else len(module.scenario_names_creator(None)).
    raise NotImplementedError("num_scens probing not wired yet")


def verify_instantiation(module, cfg):
    """Build a single scenario to confirm the model instantiates -- a cheap
    model smoke-test. SHARED CODE: used by --inspect-only when given WITHOUT
    --out-of-the-box, and by the base/plus probe in gather_facts (which also
    reads the size profile off the built model). STUB: wiring deferred."""
    # TODO(wiring): name = module.scenario_names_creator(1, cfg=cfg)[0]
    #   model = module.scenario_creator(name, **kwargs)  # raises if it can't build
    #   return the size profile (vars_int, vars_cont, nonants_total, nonants_int)
    raise NotImplementedError("scenario instantiation/verification not wired yet")


def gather_facts(module, cfg) -> Facts:
    """Assemble Facts from the environment, the model module, and cfg.
    STUB: the pieces below marked TODO are the integration work."""
    policy = load_policy()
    solvers = _detect_available_solvers(policy["solver"]["preference_order"])
    facts = Facts(
        module_name=cfg.get("module_name", "<module>"),
        num_ranks=_inspect_ranks(cfg),
        available_solvers=solvers,
        num_scens=_detect_num_scens(module, cfg),
        num_cores=os.cpu_count(),
        under_slurm=("SLURM_JOB_ID" in os.environ),
        # TODO(wiring): memory_gb (OS/SLURM-dependent), multistage flag,
        # user_flags (which options the user explicitly set on cfg), and -- at
        # the base/plus tiers -- the probe size profile (vars_int, vars_cont,
        # nonants_total, nonants_int) by instantiating probe_scenarios scenarios.
        user_flags=set(),
    )
    return facts


def out_of_the_box(module, cfg):
    """Top-level entry (STUB). Probe, recommend, report the config up front,
    run, then print Suggestions afterward."""
    facts = gather_facts(module, cfg)
    policy = load_policy()
    decision = recommend(facts, policy)

    # report the configuration up front so the user sees what is running
    # (rank-0 printing is the caller's concern).
    print(f"[out-of-the-box] using policy {policy['policy_version']}")
    for note in decision.notes:
        print(f"  - {note}")
    print(f"[out-of-the-box] equivalent command line:\n  "
          f"{decision.command_line(facts.num_ranks)}")

    if cfg.get("inspect_only", None) is not None:
        # --inspect-only: inspection is already done -- instantiation, and any
        # plus calibration solves, happened in gather_facts (resolution B) --
        # so just skip the production run. facts.num_ranks may be an assumed
        # count (--inspect-only N) for HPC planning.
        print(f"[out-of-the-box] --inspect-only: planning for {facts.num_ranks} "
              f"ranks; skipping the production run")
        outcome = None
    else:
        # TODO(wiring): apply decision.args onto cfg, then run via the normal
        # generic_cylinders driver path (EF vs cylinders); capture an `outcome`
        # (convergence, gap, iters, time) to enrich the suggestions.
        outcome = None

    # Suggestions are written AFTER the run (req. 4), labelled "Suggestions".
    decision.suggestions = make_suggestions(decision, facts, policy, outcome)
    if decision.suggestions:
        print("[out-of-the-box] Suggestions:")
        for s in decision.suggestions:
            print(f"  * {s}")

    return decision
