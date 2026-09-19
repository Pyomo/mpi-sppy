###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Out-of-the-box (OOTB) auto-configuration.

This module is the thin Python *interpreter* described in
``doc/designs/out_of_the_box_design.md`` (sec. 5/5.1). It turns:

  * a set of FACTS about the environment (MPI ranks, installed solvers,
    optional cores/memory) and the model (scenario count, stage structure, and
    -- at the base/plus tiers -- a probe size profile), and
  * a dated declarative POLICY file (``ootb_policies/ootb_policy_*.json``)

into a recommended mpi-sppy configuration, a human-readable reason for every
choice, a post-run "Suggestions" list, and the equivalent explicit command
line. The recommendation is applied back onto the ``Config`` object so the
normal ``generic_cylinders`` driver path executes it.

Design commitments:
  * USER OPTIONS ALWAYS WIN -- ``recommend()`` defers to anything in
    ``facts.user_flags`` (requirement 0).
  * The decision logic is plain, ordered, readable Python -- no rules engine,
    no neural net. The *numbers* live in the policy file, not here.
  * Every decision records WHY, so transparency (requirement 4) falls out.

Pure vs. wired:
  * Pure functions of (facts, policy): ``recommend()`` and its helpers, the
    suggestion generators, ``load_policy``. Hand-build a ``Facts`` and call
    ``recommend()`` to see the choices (this is what the validator's
    synthetic-facts layer does).
  * Wiring (touches MPI/Pyomo/Config): ``gather_facts``, ``verify_instantiation``,
    ``apply_decision``, and the ``configure``/``report_suggestions`` entry
    points the driver calls.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Effort tiers: the three OOTB flags map to one internal level. The flag name
# is the cfg key (dashes -> underscores); the value is an optional policy path.
# ---------------------------------------------------------------------------

EFFORT_FLAGS = {
    "out_of_the_box_minus": "minus",
    "out_of_the_box": "base",
    "out_of_the_box_plus": "plus",
}


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
    effort: str = "base"             # "minus" | "base" | "plus"
    # probe size profile -- populated at the base/plus tiers, None at minus
    vars_int: int | None = None       # integer/binary vars per scenario
    vars_cont: int | None = None      # continuous vars per scenario
    nonants_total: int | None = None  # first-stage (nonanticipative) vars
    nonants_int: int | None = None    # integer nonants
    # user-model nonlinearity from the probe: "linear" | "quadratic" | "nonlinear"
    # (more nonlinear than quadratic, i.e. degree > 2 or non-polynomial); None at
    # the minus tier (nothing is instantiated, so the class cannot be determined).
    model_degree: str | None = None
    multistage: bool = False
    branching_factors: list | None = None  # equivalent command line + bundle sizing
    user_solver_name: str | None = None    # name the user gave (if any)
    num_cores: int | None = None     # best effort; may be None
    memory_gb: float | None = None   # best effort; may be None
    under_slurm: bool = False
    user_flags: set[str] = field(default_factory=set)  # CLI flags the user set
    # (flag, value-or-None) for each flag the user set, minus the OOTB flags
    # themselves and the anchors command_line() already emits. Echoed in the
    # equivalent command line so it reproduces the run.
    user_args: list = field(default_factory=list)


@dataclass
class ChosenArg:
    flag: str                # e.g. "--lagrangian" or "--solver-name"
    value: str | None        # None => boolean flag
    reason: str


@dataclass
class Decision:
    run_ef: bool = False
    ef_reason: str | None = None          # "min_ranks" | "few_scens" | "user" ...
    chosen_solver: str | None = None
    problem_class: str | None = None      # LP|MIP|QP|MIQP|NLP|MINLP (base/plus)
    num_cylinders: int = 1                # hub + spokes actually configured
    intra_ranks: int = 1                  # widest cylinder's rank count
    rank_split: dict = field(default_factory=dict)   # cylinder -> ranks (flex)
    args: list[ChosenArg] = field(default_factory=list)   # what OOTB added
    notes: list[str] = field(default_factory=list)        # full reasoning trace
    suggestions: list[str] = field(default_factory=list)  # filled AFTER the run

    def command_line(self, facts: "Facts") -> str:
        """The explicit command the OOTB choices are equivalent to (req. 4).

        Anchored with the module and scenario specification, then the flags the
        user set, then the ones OOTB added, so the line RUNS and reproduces the
        run without --out-of-the-box. The user's flags have to be echoed: they
        were part of the configuration, and leaving them out produced a line
        that silently dropped the solver and any requested spoke, which is what
        the docs tell people to paste.
        """
        parts = [
            f"mpiexec -np {facts.num_ranks} python -m mpi4py -m "
            f"mpisppy.generic_cylinders",
            f"--module-name {facts.module_name}",
        ]
        if facts.multistage and facts.branching_factors:
            parts.append("--branching-factors "
                         + " ".join(str(b) for b in facts.branching_factors))
        else:
            parts.append(f"--num-scens {facts.num_scens}")
        for flag, value in facts.user_args:
            parts.append(flag if value is None else f"{flag} {value}")
        for a in self.args:
            parts.append(a.flag if a.value is None else f"{a.flag} {a.value}")
        return " ".join(parts)


# Flags that mean "the user wants a decomposition": any wired spoke or a
# non-default hub. If the user set one of these AND has >= the rank floor, OOTB
# must NOT substitute the EF, even for a small problem (requirement 0). This is
# the generic_cylinders vocabulary -- a fact, not a focus preference -- so it
# lives in code, and the validator checks it against the actual CLI flags.
SPOKE_FLAGS = frozenset({
    "--lagrangian", "--fwph", "--ph-dual", "--ph-xfeas-spoke", "--relaxed-ph",
    "--subgradient", "--reduced-costs", "--xhatshuffle", "--xhatxbar",
    "--xhatlshaped",
})
# A hub flag selects WHICH hub runs; it does not add a cylinder.
HUB_FLAGS = frozenset({
    "--APH", "--subgradient-hub", "--fwph-hub", "--ph-primal-hub",
    "--lshaped-hub", "--cg-hub", "--dualcg-hub",
})
DECOMPOSITION_FLAGS = SPOKE_FLAGS | HUB_FLAGS


# Options that read the incumbent (Field.BEST_XHAT). With no inner-bound spoke
# in the roster nothing publishes one and the run dies in do_decomp with
# KeyError: <Field.BEST_XHAT>, so OOTB must not add these to such a roster.
# Verified per flag: --sensi-rho and --sep-rho do NOT need one.
NEEDS_INNER_BOUND = frozenset({
    "--grad-rho", "--dynamic-rho-primal-crit", "--dynamic-rho-dual-crit",
})


def _requested_cylinders(user_flags) -> int:
    """How many cylinders the user's own flags ask for: the hub plus each
    requested spoke. Used to decide whether an explicit decomposition request
    actually fits in the ranks available."""
    return 1 + len(user_flags & SPOKE_FLAGS)


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

    # --- step 1: pick the solver NAME ---------------------------------------
    # The flag is emitted only once the EF gate decides: --EF-solver-name for the
    # EF, --solver-name to decompose (mpi-sppy uses different keys). Here we just
    # settle on the name -- routed by the model's PROBLEM CLASS (LP/MIP/QP/MIQP/
    # NLP/MINLP) so a more-than-quadratic model gets an NLP solver (ipopt) and an
    # integer model never gets a continuous-only one.
    sp = policy["solver"]
    d.problem_class = _problem_class(facts)
    if d.problem_class is not None:
        d.notes.append(f"model class: {d.problem_class} ({_class_english(facts)})")
    if facts.user_flags & {"--solver-name", "--EF-solver-name"} and facts.user_solver_name:
        d.chosen_solver = facts.user_solver_name
        d.notes.append(f"solver: kept user's value ({facts.user_solver_name}); "
                       f"OOTB defers")
    else:
        by_class = sp.get("preference_order_by_class", {})
        if d.problem_class is not None and d.problem_class in by_class:
            order = by_class[d.problem_class]
            why = f"first available preferred for a {d.problem_class} model"
        else:
            # minus tier (class unknown) or a class with no dedicated list: use
            # the master preference order.
            order = sp["preference_order"]
            why = "first available in preference order"
        d.chosen_solver = _first_available(order, facts.available_solvers)
        if d.chosen_solver is None:
            d.notes.append(
                f"WARNING: no installed solver for this model "
                f"({d.problem_class or 'class unknown'}); tried {', '.join(order)}. "
                f"User must supply one (--solver-name).")
        else:
            d.notes.append(f"solver: {d.chosen_solver} ({why})")

    # --- step 2: EF gate ----------------------------------------------------
    # Reuses the bundle effort() model on the WHOLE problem (all scenarios as one
    # model) vs an ABSOLUTE EF budget. base: effort(num_scens) <= ef_effort_budget;
    # plus would use ef_target_seconds via a measured t1 (stubbed); minus (no
    # size profile): the count rule.
    ef = policy["ef_fallback"]
    have_profile = facts.vars_cont is not None or facts.vars_int is not None
    requested = facts.user_flags & DECOMPOSITION_FLAGS
    # Ranks the user's OWN request needs. The policy's rank floor is about when
    # OOTB would CHOOSE to decompose; it must not veto a decomposition the user
    # asked for and that mpi-sppy can actually run (hub + one spoke on 2 ranks
    # is a real configuration), so the request is tested against this instead.
    need_for_request = _requested_cylinders(facts.user_flags)
    if "--EF" in facts.user_flags:
        d.run_ef, d.ef_reason = True, "user"
        d.notes.append("--EF: user requested")
    elif requested and facts.num_ranks >= need_for_request:
        # User explicitly asked for a decomposition and it fits, so we never
        # substitute the EF -- even for a small problem (requirement 0).
        forced = ", ".join(sorted(requested))
        d.notes.append(f"EF gate skipped: user requested decomposition ({forced}) "
                       f"and {facts.num_ranks} ranks cover the "
                       f"{need_for_request} cylinder(s) it needs")
    elif requested:
        # Asked for, but genuinely does not fit. Say so rather than silently
        # running something else.
        d.run_ef, d.ef_reason = True, "min_ranks"
        forced = ", ".join(sorted(requested))
        choose("--EF", None,
               f"user requested decomposition ({forced}) needs "
               f"{need_for_request} ranks but only {facts.num_ranks} are "
               f"available; running the EF instead")
    elif facts.num_ranks < ef["min_ranks_for_decomposition"]:
        d.run_ef, d.ef_reason = True, "min_ranks"
        choose("--EF", None,
               f"only {facts.num_ranks} ranks; decomposition needs "
               f">= {ef['min_ranks_for_decomposition']}")
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
        # The EF path reads cfg.EF_solver_name (its own key).
        if d.chosen_solver is not None:
            choose("--EF-solver-name", d.chosen_solver,
                   f"EF solver ({d.chosen_solver})")
        d.num_cylinders = 1
        return d

    # decomposition solver + (if LP/MIP-only) prox linearization. The hub/spoke
    # path reads cfg.solver_name.
    if d.chosen_solver is not None:
        choose("--solver-name", d.chosen_solver,
               f"decomposition solver ({d.chosen_solver})")
    if d.chosen_solver in sp["lp_mip_only_force_linearize_prox"]:
        choose("--linearize-proximal-terms", None,
               f"{d.chosen_solver} is LP/MIP-only; the PH prox must be linearized")

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

    # Every cylinder needs at least one rank -- apportion_ranks raises
    # otherwise -- so the roster cannot be wider than num_ranks - 1 spokes.
    # The core roster above is taken unconditionally, so below the policy's
    # rank floor (reachable now that an explicit request is honored there) it
    # could ask for more cylinders than there are ranks. A spoke the user
    # asked for is kept ahead of one OOTB chose for itself.
    user_spokes = facts.user_flags & SPOKE_FLAGS
    max_spokes = max(0, facts.num_ranks - 1)
    if len(chosen) > max_spokes:
        chosen.sort(key=lambda r: (r["flag"] not in user_spokes, r["priority"]))
        dropped = [r["flag"] for r in chosen[max_spokes:]]
        chosen = sorted(chosen[:max_spokes], key=lambda r: r["priority"])
        d.notes.append(
            f"spoke roster trimmed to {len(chosen)} for {facts.num_ranks} "
            f"ranks (one rank minimum per cylinder); dropped "
            + ", ".join(dropped))

    for r in chosen:
        choose(r["flag"], None, f"spoke ({r['bound']}, priority {r['priority']})")
    d.num_cylinders = 1 + len(chosen)

    # --- step 4: rank allocation -- UNBALANCED by per-cylinder ratio --------
    # Split ranks across cylinders by ratio (flex-ranks), not uniformly; xhat
    # cylinders are cheaper so get a smaller share. We emit the per-spoke
    # --<spoke>-rank-ratio flags (only when a spoke's ratio differs from the
    # default) and let WheelSpinner.apportion_ranks do the real split. CRUDE
    # cold-start: the real split depends on subproblem solve cost (a plus-tier
    # refinement).
    ra = policy["rank_allocation"]
    default_ratio = ra["default_rank_ratio"]
    # ratios in cylinder order: hub first (always the default), then spokes.
    ratios = [default_ratio]
    # A ratio the user set is the ratio the RUN will use, so model that one --
    # otherwise intra_ranks (and the bundle floor it drives) is computed from a
    # split that never happens.
    user_ratio = dict(facts.user_args)
    for r in chosen:
        ratio_flag = f"{r['flag']}-rank-ratio"
        spoke_ratio = ra["rank_ratios"].get(r["flag"], default_ratio)
        if ratio_flag in facts.user_flags:
            try:
                spoke_ratio = float(user_ratio[ratio_flag])
            except (KeyError, TypeError, ValueError):
                pass                       # unparseable: keep the policy value
        ratios.append(spoke_ratio)
        if spoke_ratio != default_ratio:
            choose(ratio_flag, _fmt_ratio(spoke_ratio),
                   f"flex-ranks: cheaper cylinder gets a {spoke_ratio} share "
                   f"(crude cold-start)")
    d.intra_ranks, d.rank_split, divisible = _rank_layout(
        facts.num_ranks, ratios, chosen)
    if not divisible:
        # The equal-rank path in WheelSpinner refuses a rank count that is not
        # a multiple of the cylinder count, so say it here instead of letting
        # the run abort in _make_comms.
        d.notes.append(
            f"WARNING: {facts.num_ranks} ranks is not a multiple of "
            f"{d.num_cylinders} cylinders; mpi-sppy requires a multiple when "
            f"every cylinder has the same rank ratio. Use "
            f"{d.num_cylinders * (facts.num_ranks // d.num_cylinders)} ranks "
            f"(or a multiple of {d.num_cylinders}).")
    d.notes.append("rank split (flex-ranks, crude cold-start): "
                   + ", ".join(f"{k}={v}" for k, v in d.rank_split.items()))

    # --- step 5: proper bundling -- how BIG? (design "Bundle sizing") -------
    # minus CANNOT bundle: with no size profile there is no safe way to size a
    # bundle, so the minus tier always runs unbundled.
    bs = policy["bundle_sizing"]
    if not have_profile:        # have_profile computed in the EF-gate step above
        d.notes.append("bundling: minus tier (no size profile); running unbundled")
    elif facts.num_scens >= bs["min_scens_to_consider_bundling"]:
        b_min = max(d.intra_ranks,
                    bs["min_bundles_per_intra_rank"] * d.intra_ranks)
        mult = _spb_multiple_of(facts)
        spb = _pick_spb_by_effort(
            facts.num_scens, b_min, facts, policy["effort_scaling"],
            bs["base_max_hardness_vs_single_scenario"], multiple_of=mult)
        whole_nodes = (f", multiple of {mult} (whole second-stage nodes)"
                       if mult > 1 else "")
        if spb is not None:
            nb = facts.num_scens // spb
            choose("--scenarios-per-bundle", str(spb),
                   f"{facts.num_scens} scenarios -> {nb} bundles of {spb} "
                   f"(effort <= {bs['base_max_hardness_vs_single_scenario']}x a "
                   f"single scenario, >= {b_min} bundles{whole_nodes})")
        else:
            d.notes.append(
                f"bundling: no divisor of {facts.num_scens} qualifies "
                f"(>= {b_min} bundles within budget{whole_nodes}); "
                f"running unbundled"
            )

    # --- step 6: extra options by concern; any superseded_by flag defers ----
    # Per-concern override: OOTB backs off a whole concern (e.g. its rho setter)
    # if the user set ANY equivalent flag, not just the identical one -- mpi-sppy
    # allows only one rho setter, so stacking would be a hard error.
    have_inner = any(r["bound"] == "inner" for r in chosen)

    def _skip_for_no_inner(flag):
        """True if `flag` reads the incumbent but no spoke will publish one."""
        if have_inner or flag not in NEEDS_INNER_BOUND:
            return False
        d.notes.append(
            f"{flag}: skipped -- it reads the incumbent, and this roster has "
            f"no inner-bound spoke to publish one")
        return True

    for name, cat in policy.get("option_categories", {}).items():
        if name.startswith("_"):
            continue
        if any(f in facts.user_flags for f in cat.get("superseded_by", [cat["flag"]])):
            d.notes.append(f"{name}: superseded by a user option; OOTB defers")
        elif not _skip_for_no_inner(cat["flag"]):
            choose(cat["flag"], cat.get("value"), f"policy option ({name})")
    # catch-all: per-flag override (superseded_by defaults to the flag itself)
    for opt in policy.get("additional_options", {}).get("options", []):
        if any(f in facts.user_flags for f in opt.get("superseded_by", [opt["flag"]])):
            d.notes.append(f"{opt['flag']}: superseded by a user option; OOTB defers")
        elif not _skip_for_no_inner(opt["flag"]):
            choose(opt["flag"], opt.get("value"), "policy additional option")

    return d


def _fmt_ratio(x: float) -> str:
    """Render a rank ratio without a trailing ``.0`` (so 0.2 stays 0.2)."""
    return str(int(x)) if float(x).is_integer() else str(x)


def _first_available(order, available) -> str | None:
    """First solver in `order` that is installed (present in `available`)."""
    for name in order:
        if name in available:
            return name
    return None


# (model_degree, has_integer_vars) -> problem class label. Used to route solver
# selection to the matching preference_order_by_class list.
_PROBLEM_CLASS = {
    ("linear", False): "LP",
    ("quadratic", False): "QP",
    ("nonlinear", False): "NLP",
    ("linear", True): "MIP",
    ("quadratic", True): "MIQP",
    ("nonlinear", True): "MINLP",
}


def _problem_class(facts: Facts) -> str | None:
    """The model's problem class (LP/MIP/QP/MIQP/NLP/MINLP), or None when the
    model was not instantiated (minus tier) so integrality/degree are unknown.
    Integrality is `vars_int > 0`; the degree is `model_degree` -- "linear",
    "quadratic", or "nonlinear" (more nonlinear than quadratic)."""
    if facts.model_degree is None or facts.vars_int is None:
        return None
    return _PROBLEM_CLASS[(facts.model_degree, facts.vars_int > 0)]


def _class_english(facts: Facts) -> str:
    """Plain-language basis for the problem class, e.g. "integer + quadratic"."""
    integer = "integer" if (facts.vars_int or 0) > 0 else "continuous"
    return f"{integer} + {facts.model_degree}"


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


def _spb_multiple_of(facts: Facts) -> int:
    """The granularity a multistage bundle size must respect.

    ``ProperBundler.set_bunBFs`` requires ``scenarios_per_bundle`` to be a
    multiple of ``prod(branching_factors[1:])`` -- a bundle has to consume whole
    second-stage nodes -- and raises otherwise. Two-stage runs have no such
    constraint, so this returns 1 there."""
    bf = facts.branching_factors
    if not facts.multistage or not bf or len(bf) < 2:
        return 1
    beyond2size = 1
    for b in bf[1:]:
        b = int(b)
        if b <= 0:
            # Degenerate branching factors (a run with those cannot build a
            # scenario tree anyway); 1 keeps the sizer's `spb % multiple_of`
            # from dividing by zero and leaves the size unconstrained.
            return 1
        beyond2size *= b
    return beyond2size


def _pick_spb_by_effort(num_scens: int, min_bundles: int, facts: Facts,
                        scaling: dict, max_hardness: float,
                        multiple_of: int = 1) -> int | None:
    """base/plus sizer: the LARGEST scenarios_per_bundle that divides num_scens,
    leaves >= min_bundles bundles, and keeps a bundle's modeled effort within
    `max_hardness` x a single scenario (the relative, unit-free budget). The plus
    tier feeds the same function a time-calibrated effective hardness. Returns
    None when nothing past the degenerate spb==1 qualifies.

    `multiple_of` is the multistage granularity from `_spb_multiple_of`: a size
    that is not a multiple of it would abort the run in `set_bunBFs`, so such
    sizes are not candidates. That test is also what makes the leading bundle
    branching factor (`spb // multiple_of`) come out whole; dividing num_scens
    additionally makes the bundle count whole."""
    e1 = _effort(1, facts, scaling)
    best = None
    for spb in range(2, num_scens + 1):           # spb==1 is "no bundling"
        if num_scens % spb or num_scens // spb < min_bundles:
            continue
        if spb % multiple_of:
            continue
        if e1 > 0 and _effort(spb, facts, scaling) / e1 > max_hardness:
            continue
        if best is None or spb > best:            # largest = max amortization
            best = spb
    return best


def _rank_layout(total: int, ratios: list, chosen: list) -> tuple:
    """Return (intra_ranks, rank_split) for the chosen cylinders.

    `ratios` is hub-first, matching `[hub] + chosen`. We mirror what
    WheelSpinner actually branches on (`spin_the_wheel.py`): it apportions by
    ratio when ANY ratio differs from 1.0, and otherwise takes the equal split.
    Testing "all ratios equal" instead would model an equal split for a policy
    whose ratios are uniform but not 1.0, while the run apportioned.

    The equal-split path is the one `_make_comms` guards with "Need a multiple
    of N processes", so that case is reported here rather than surfacing as a
    startup abort. intra_ranks is the widest cylinder's rank count (it governs
    the bundle floor)."""
    names = ["(hub)"] + [r["flag"] for r in chosen]
    if any(r != 1.0 for r in ratios):
        from mpisppy.utils.rank_apportionment import apportion_ranks
        counts = apportion_ranks(ratios, total)
        split = dict(zip(names, counts))
        divisible = True
    else:
        per = max(1, total // len(ratios))
        split = {nm: per for nm in names}
        divisible = total % len(ratios) == 0
    return max(split.values()), split, divisible


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


def _sg_no_class_solver(d, facts, policy, outcome):
    # A problem class was detected but nothing installed handles it -- e.g. an
    # MINLP with no baron/scip/bonmin/couenne, or an NLP with no ipopt. (Integer
    # + more-than-quadratic models route to MINLP, which rarely has an installed
    # solver.) recommend() already left chosen_solver None here.
    if d.chosen_solver is None and d.problem_class:
        listed = policy["solver"].get("preference_order_by_class", {}) \
            .get(d.problem_class, [])
        names = ", ".join(listed) if listed else "none listed"
        return (f"Detected a {d.problem_class} model, but none of the solvers "
                f"OOTB prefers for it ({names}) are installed; install one or "
                f"pass --solver-name explicitly.")
    return None


def _sg_no_persistent_solver(d, facts, policy, outcome):
    """Suggest the persistent interface only when one actually exists.

    Testing just `not s.endswith("_persistent")` told cbc/glpk/ipopt/highs
    users to switch to e.g. "cbc_persistent", which is not a Pyomo solver --
    a dead end. A persistent interface exists when the policy's master
    preference order names one, which is what the run could actually use.
    """
    s = d.chosen_solver
    if d.run_ef or s is None or s.endswith("_persistent"):
        return None
    persistent = f"{s}_persistent"
    if persistent not in policy["solver"]["preference_order"]:
        return None
    if persistent in facts.available_solvers:
        return (f"Chosen solver '{s}' is installed in its persistent form too; "
                f"naming '{persistent}' instead would warm-start subproblems "
                f"and is usually much faster for PH.")
    return (f"Chosen solver '{s}' has no persistent interface available; "
            f"installing '{persistent}' would warm-start subproblems and is "
            f"usually much faster for PH.")


def _sg_linearized_prox(d, facts, policy, outcome):
    if not d.run_ef and \
            d.chosen_solver in policy["solver"]["lp_mip_only_force_linearize_prox"]:
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


def _sg_minus_no_bundling(d, facts, policy, outcome):
    if facts.effort == "minus" and not d.run_ef:
        return ("Ran the --out-of-the-box-minus tier, which instantiates nothing "
                "and therefore cannot size proper bundles; --out-of-the-box (base) "
                "probes one scenario and can bundle when there are many scenarios.")
    return None


def _sg_from_outcome(d, facts, policy, outcome):
    # Outcome-based (post-run) computed suggestion; inert until the run captures
    # an outcome (None in the minus/base tiers for now).
    if outcome and outcome.get("converged") is False:
        return (f"PH stopped at {outcome.get('iterations')} iterations with a "
                f"{outcome.get('rel_gap', 0):.0%} gap; consider raising "
                f"--max-iterations or adding a tighter bound spoke.")
    return None


# generators in priority order (lower first)
SUGGESTION_GENERATORS = [
    _sg_ran_ef_few_ranks,
    _sg_no_class_solver,
    _sg_no_persistent_solver,
    _sg_linearized_prox,
    _sg_more_ranks,
    _sg_minus_no_bundling,
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
    """Load a policy. Default (policy_file None or empty): the newest dated file
    with no focus token in ootb_policies/ (``ootb_policy_<date>.json``). The
    dated filename sorts lexically by ISO date, so the last is the newest."""
    if not policy_file:
        d = _policies_dir()
        # focus is conveyed by extra filename tokens (ootb_policy_quick_<date>);
        # the bare default is ootb_policy_<date>.json -- two underscores exactly.
        candidates = sorted(
            f for f in os.listdir(d)
            if f.startswith("ootb_policy_") and f.endswith(".json")
            and f.count("_") == 2
        )
        if not candidates:
            raise FileNotFoundError(f"no default OOTB policy files in {d}")
        policy_file = os.path.join(d, candidates[-1])
    with open(policy_file) as fp:
        return json.load(fp)


# ---------------------------------------------------------------------------
# Environment/model probing + apply-to-Config  [the wiring]
# ---------------------------------------------------------------------------


def effort_and_policy(cfg) -> tuple:
    """Return (effort, policy_path) from the cfg, or (None, None) if OOTB is off.

    The three tier flags are mutually exclusive; supplying more than one is an
    error. ``policy_path`` is the optional value attached to the chosen flag
    (empty string -> use the shipped default policy)."""
    selected = [(cfg_key, tier) for cfg_key, tier in EFFORT_FLAGS.items()
                if cfg.get(cfg_key) is not None]
    if not selected:
        return None, None
    if len(selected) > 1:
        names = ", ".join("--" + k.replace("_", "-") for k, _ in selected)
        raise RuntimeError(
            f"At most one out-of-the-box tier may be given; got {names}.")
    cfg_key, tier = selected[0]
    policy_path = cfg.get(cfg_key) or None      # "" (bare flag) -> default
    return tier, policy_path


def requested(cfg) -> bool:
    """True if any out-of-the-box tier flag was supplied."""
    return any(cfg.get(k) is not None for k in EFFORT_FLAGS)


def _rank0() -> bool:
    from mpisppy import MPI
    return MPI.COMM_WORLD.Get_rank() == 0


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
        try:
            n = int(io)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"--inspect-only takes an MPI rank count (e.g. --inspect-only "
                f"512) or no value at all; got {io!r}.") from None
        if n < 1:
            raise RuntimeError(
                f"--inspect-only rank count must be at least 1; got {n}.")
        return n
    return _detect_num_ranks()


def _detect_available_solvers(candidates) -> set:
    """Which of the policy's candidate solvers are installed.

    Asking for a solver that is not installed is normal here, but Pyomo logs a
    WARNING and a full traceback for each missing ASL solver (bonmin, couenne,
    ...) before raising. The `except` below swallows the exception, not the
    logging, so the first command a new user runs printed two tracebacks ahead
    of its own output and looked like a crash. Silence those loggers for the
    duration of the probe only.
    """
    import logging
    import pyomo.environ as pyo

    quieted = [logging.getLogger(n)
               for n in ("pyomo.opt", "pyomo.solvers", "pyomo.common")]
    previous = [lg.level for lg in quieted]
    found = set()
    try:
        for lg in quieted:
            lg.setLevel(logging.CRITICAL)
        for name in candidates:
            try:
                if pyo.SolverFactory(name).available(exception_flag=False):
                    found.add(name)
            except Exception:
                pass
    finally:
        for lg, lvl in zip(quieted, previous):
            lg.setLevel(lvl)
    return found


def _detect_num_scens(module, cfg) -> int:
    """Mirror mpisppy/generic/parsing.py::name_lists: cfg.num_scens, else the
    product of the branching factors, else the module's full scenario list."""
    if cfg.get("num_scens") is not None:
        return int(cfg.num_scens)
    bf = cfg.get("branching_factors")
    if bf is not None:
        return int(math.prod(bf))
    return len(module.scenario_names_creator(None))


def _cfg_key_to_flag(key: str) -> str:
    """cfg key -> the CLI flag that sets it ("solver_name" -> "--solver-name")."""
    return "--" + key.replace("_", "-")


# Set by the user but NOT echoed in the equivalent command line: the OOTB flags
# themselves (the line is what to run instead of them) and the anchors
# command_line() already emits.
_NOT_ECHOED = frozenset(EFFORT_FLAGS) | {
    "inspect_only", "module_name", "num_scens", "branching_factors",
}


def _user_args(cfg) -> list:
    """(flag, value) pairs for what the user set, formatted for a command line.

    A bool that is True becomes a bare flag; a list (e.g. branching factors)
    is space-joined. A False bool is dropped -- it is the default, and there
    is no --no-<flag> spelling to emit.
    """
    out = []
    for val in cfg.user_values():
        try:
            key = val.name()
        except Exception:
            continue
        if key in _NOT_ECHOED:
            continue
        v = cfg.get(key)
        if isinstance(v, bool):
            if v:
                out.append((_cfg_key_to_flag(key), None))
        elif isinstance(v, (list, tuple)):
            out.append((_cfg_key_to_flag(key),
                        " ".join(str(x) for x in v)))
        elif v is not None:
            out.append((_cfg_key_to_flag(key), str(v)))
    return sorted(out)


def _user_flags(cfg=None) -> set:
    """The set of long CLI flags the user actually set.

    This is how requirement 0 is honored: recommend() defers to any flag in
    here. It comes from Pyomo's own record of which ConfigValues were assigned
    (``cfg.user_values()``) rather than from scanning ``sys.argv``, because
    argparse accepts unambiguous abbreviations: ``--solver-nam cbc`` sets
    solver_name, but an argv scan yields the token "--solver-nam", which
    matches no flag OOTB knows, so OOTB would decide the user had not chosen a
    solver and overwrite it. user_values() reports ``solver_name`` either way,
    and also covers ``--flag=value`` and values that did not come from the
    command line at all.

    `cfg` is optional only so the argv fallback keeps working for callers that
    have no Config (hand-built Facts in tests); prefer passing it.
    """
    if cfg is None:
        return {tok.split("=", 1)[0]
                for tok in sys.argv[1:] if tok.startswith("--")}
    flags = set()
    for val in cfg.user_values():
        try:
            flags.add(_cfg_key_to_flag(val.name()))
        except Exception:
            continue
    return flags


def _model_degree(model) -> str:
    """Classify the user model's nonlinearity from the probe: "linear" (every
    active objective/constraint has polynomial degree <= 1), "quadratic" (max
    degree == 2), or "nonlinear" (degree > 2, or non-polynomial such as
    log/exp/x*y/y -- reported by Pyomo as polynomial_degree() == None).

    The probe is the RAW scenario_creator model, so the PH proximal term (a
    quadratic that mpi-sppy attaches later) is absent and does not inflate the
    degree -- this reflects the user's own model."""
    import pyomo.environ as pyo
    max_deg = 0
    for obj in model.component_data_objects(pyo.Objective, active=True,
                                            descend_into=True):
        deg = obj.expr.polynomial_degree() if obj.expr is not None else 0
        if deg is None:
            return "nonlinear"
        max_deg = max(max_deg, deg)
    for con in model.component_data_objects(pyo.Constraint, active=True,
                                            descend_into=True):
        body = con.body
        deg = body.polynomial_degree() if body is not None else 0
        if deg is None:
            return "nonlinear"
        max_deg = max(max_deg, deg)
    if max_deg > 2:
        return "nonlinear"
    return "quadratic" if max_deg == 2 else "linear"


def _size_profile(model) -> dict:
    """Read a built scenario's size profile: integer vs continuous variable
    counts, nonant (first-stage) counts, and the model degree (linear/quadratic/
    nonlinear). Used by the base/plus probe and by --inspect-only's
    instantiation check."""
    import pyomo.environ as pyo
    vars_int = vars_cont = 0
    for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if v.is_continuous():
            vars_cont += 1
        else:
            vars_int += 1
    nonants_total = nonants_int = 0
    for node in getattr(model, "_mpisppy_node_list", []):
        for v in node.nonant_vardata_list:
            nonants_total += 1
            if not v.is_continuous():
                nonants_int += 1
    return {"vars_int": vars_int, "vars_cont": vars_cont,
            "nonants_total": nonants_total, "nonants_int": nonants_int,
            "model_degree": _model_degree(model)}


def _build_probe_scenario(module, cfg):
    """Instantiate a single RAW scenario (no bundle/ADMM/cvar wrapper) from the
    model module so OOTB can measure per-scenario size. Raises if the model
    cannot build."""
    names = module.scenario_names_creator(1)
    kwargs = module.kw_creator(cfg)
    return module.scenario_creator(names[0], **kwargs)


def verify_instantiation(module, cfg) -> dict:
    """Build a single scenario to confirm the model instantiates -- a cheap
    model smoke-test -- and return its size profile. SHARED CODE: used by the
    base/plus probe in gather_facts and by --inspect-only when given WITHOUT
    --out-of-the-box (the standalone model smoke-test)."""
    model = _build_probe_scenario(module, cfg)
    return _size_profile(model)


def gather_facts(module, cfg, effort: str, policy: dict) -> Facts:
    """Assemble Facts from the environment, the model module, and cfg. The
    `effort` tier sets how deep we look: minus reads structure only; base/plus
    also instantiate `probe_scenarios` scenario(s) for the size profile."""
    solvers = _detect_available_solvers(policy["solver"]["preference_order"])
    bf = cfg.get("branching_factors")
    facts = Facts(
        module_name=cfg.get("module_name", "<module>") or "<module>",
        num_ranks=_inspect_ranks(cfg),
        available_solvers=solvers,
        num_scens=_detect_num_scens(module, cfg),
        effort=effort,
        multistage=bf is not None,
        branching_factors=list(bf) if bf is not None else None,
        user_solver_name=cfg.get("solver_name") or cfg.get("EF_solver_name"),
        num_cores=os.cpu_count(),
        under_slurm=("SLURM_JOB_ID" in os.environ),
        user_flags=_user_flags(cfg),
        user_args=_user_args(cfg),
    )
    if effort in ("base", "plus"):
        # one probe scenario (discarded) feeds the size-aware decisions; the
        # real, layout-correct instantiation happens later in the driver.
        profile = verify_instantiation(module, cfg)
        facts.vars_int = profile["vars_int"]
        facts.vars_cont = profile["vars_cont"]
        facts.nonants_total = profile["nonants_total"]
        facts.nonants_int = profile["nonants_int"]
        facts.model_degree = profile["model_degree"]
    # plus tier (PR2): instantiate all + brief timed solve for solve-time facts.
    return facts


def apply_decision(decision: Decision, cfg) -> None:
    """Apply the recommended args onto the Config so the normal driver path runs
    the chosen configuration. User-set flags are already absent from
    decision.args (recommend() never adds them), so this never overrides the
    user (requirement 0). Pyomo coerces string values through each option's
    domain."""
    if decision.run_ef:
        cfg["EF"] = True
    for a in decision.args:
        key = a.flag[2:].replace("-", "_")     # "--solver-name" -> "solver_name"
        cfg[key] = True if a.value is None else a.value

    # The EF path reads cfg.EF_solver_name, the decomposition path cfg.solver_name
    # -- two different keys. A naive user (and OOTB itself) naturally sets
    # --solver-name; if OOTB then runs the EF, carry that name over so the EF
    # path is never left without a solver. (recommend() already emits
    # --EF-solver-name; this guarantees it independent of how cfg was populated.)
    if decision.run_ef and cfg.get("EF_solver_name") is None \
            and cfg.get("solver_name") is not None:
        cfg["EF_solver_name"] = cfg["solver_name"]

    # parse_args() ran cfg.checker() BEFORE OOTB existed, so every combination
    # added above is unvalidated. Config.checker rejects pairs OOTB can build
    # -- e.g. a user's --coeff-rho with OOTB's --dynamic-rho-primal-crit, which
    # checker counts as "dynamic rho without an automated rho setter" -- and
    # those were slipping through to a silently inert flag or a later failure.
    # Re-check the configuration OOTB actually produced.
    cfg.checker()


@dataclass
class OOTBState:
    decision: Decision
    facts: Facts
    policy: dict
    effort: str


def configure(module, cfg) -> OOTBState:
    """Top-level OOTB entry called by the driver right after arg parsing. Probe
    the environment/model, recommend a configuration, print it and the
    equivalent command line (rank 0), and APPLY it onto cfg so the normal driver
    path runs it. Returns the state so the driver can print Suggestions after
    the run (see report_suggestions)."""
    effort, policy_path = effort_and_policy(cfg)
    policy = load_policy(policy_path)
    facts = gather_facts(module, cfg, effort, policy)
    decision = recommend(facts, policy)

    if _rank0():
        print(f"[out-of-the-box] tier '{effort}', policy "
              f"{policy.get('policy_version', '?')}")
        for note in decision.notes:
            print(f"  - {note}")
        print("[out-of-the-box] equivalent command line:\n  "
              + decision.command_line(facts))

    apply_decision(decision, cfg)
    return OOTBState(decision=decision, facts=facts, policy=policy, effort=effort)


def report_suggestions(state: OOTBState, outcome: dict | None = None) -> None:
    """Print the prioritized "Suggestions" list (req. 4), AFTER the run so it can
    reflect how the run went (via `outcome`). Rank 0 only."""
    if not _rank0():
        return
    state.decision.suggestions = make_suggestions(
        state.decision, state.facts, state.policy, outcome)
    if state.decision.suggestions:
        print("[out-of-the-box] Suggestions:")
        for s in state.decision.suggestions:
            print(f"  * {s}")


def inspect_only_standalone(module, cfg) -> None:
    """--inspect-only WITHOUT any --out-of-the-box tier: verify one scenario can
    be instantiated (a cheap model smoke-test), report its size, and stop. Rank
    0 prints; all ranks build (cheap, deterministic)."""
    profile = verify_instantiation(module, cfg)
    if _rank0():
        print("[inspect-only] model instantiates; one-scenario size profile:")
        for k, v in profile.items():
            print(f"    {k}: {v}")
        print("[inspect-only] no out-of-the-box tier given; "
              "skipping the production run.")
