###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Preference-driven in-hub slamming.

Slamming forces (fixes) a nonanticipative variable according to pre-specified
user preferences while a decomposition hub is running, to drive toward a
feasible / converged incumbent when ordinary convergence is slow.  Unlike the
other in-hub fixers (``fixer.py``, ``reduced_costs_fixer.py``,
``relaxed_ph_fixer.py``), which fix a variable once its scenarios *agree* and so
can infer a direction automatically, slamming is meant for variables that are
*not* settling -- where there is no agreement to read a direction from -- which
is why it needs *user-supplied* direction preferences.  Those preferences come
from a directives file (see :func:`parse_directives_file`).

This "not settling" framing is the intended use, not an enforced precondition:
slamming does not test convergence per variable.  Any variable matched by a
``can_slam`` rule is eligible whether or not its scenarios already agree --
slamming an already-settled variable simply pins it where it was heading.

The trigger is an iteration count (slam after a start iteration, then once
every so many iterations); one variable is slammed per event and the slam is
sticky.  The ``Slammer`` is added to a hub only when
``--slamming-directives-file`` is supplied; with no slamming options a run
behaves exactly as it does today.  See ``doc/designs/slamming_design.md`` for
design details and rationale.

The ``Slammer`` is distinct from the ``SlamMin`` / ``SlamMax`` *spokes* in
``mpisppy/cylinders/slam_heuristic.py``: those are non-destructive incumbent
finders that fix *all* nonants to a per-variable extremum and never perturb the
hub; the ``Slammer`` alters the hub search per user preference.
"""

import csv
import math
import re

import numpy as np
import pyomo.environ as pyo

import mpisppy.MPI as MPI
from mpisppy.extensions.extension import Extension
from mpisppy.utils.sputils import is_persistent


# Ordered set of legal direction tokens (see the design, §3).
VALID_DIRECTIONS = ("lb", "ub", "nearest", "anywhere", "min", "max")


def _compile_glob(pattern):
    """Compile a shell-style name pattern into a regex.

    ``*`` matches any run of characters and ``?`` matches exactly one; every
    other character -- crucially including ``[`` and ``]`` -- is literal.  This
    is deliberately *not* :mod:`fnmatch`: nonant names are indexed Pyomo names
    such as ``DoBuild[Seattle]``, and ``fnmatch`` would read the brackets as a
    character class, so ``DoBuild[*]`` would fail to match.  Here the brackets
    are literal, so ``DoBuild[*]`` matches every ``DoBuild[...]`` as intended.
    """
    out = []
    for ch in pattern:
        if ch == "*":
            out.append(".*")
        elif ch == "?":
            out.append(".")
        else:
            out.append(re.escape(ch))
    return re.compile("".join(out), re.DOTALL)


class SlamDirective:
    """One parsed row of a directives file.

    Attributes:
        pattern (str): a shell-style name pattern (``*``/``?`` wildcards, with
            ``[``/``]`` literal) matched against ``xvar.name`` (see
            :func:`_compile_glob`).
        can_slam (bool): whether matched variables may be slammed.
        directions (tuple): ordered direction tokens; the first *applicable*
            one is used (see :data:`VALID_DIRECTIONS`).
        priority (float): larger priority is slammed first; ties broken by name.
    """
    __slots__ = ("pattern", "can_slam", "directions", "priority", "_regex")

    def __init__(self, pattern, can_slam, directions, priority):
        self.pattern = pattern
        self.can_slam = can_slam
        self.directions = tuple(directions)
        self.priority = priority
        self._regex = _compile_glob(pattern)

    def matches(self, name):
        """True if ``name`` matches this directive's pattern in full."""
        return self._regex.fullmatch(name) is not None

    def __repr__(self):
        return (f"SlamDirective(pattern={self.pattern!r}, "
                f"can_slam={self.can_slam}, directions={self.directions}, "
                f"priority={self.priority})")


def _parse_can_slam(raw):
    token = raw.strip().lower()
    if token in ("", "1", "true", "yes"):
        return True
    if token in ("0", "false", "no"):
        return False
    raise ValueError(f"slamming directives: cannot parse can_slam value {raw!r} "
                     "(expected 0/1)")


def _parse_directions(raw):
    tokens = [t.strip().lower() for t in raw.split("|") if t.strip() != ""]
    for t in tokens:
        if t not in VALID_DIRECTIONS:
            raise ValueError(
                f"slamming directives: unknown direction token {t!r} "
                f"(legal tokens: {', '.join(VALID_DIRECTIONS)})")
    return tuple(tokens)


def parse_directives_file(path):
    """Parse a slamming directives CSV into a list of :class:`SlamDirective`.

    The file is keyed by nonant name with wildcards.  Columns:

    ``name`` (required)
        shell-style pattern (``*``/``?`` wildcards; ``[``/``]`` literal) matched
        against ``xvar.name`` (e.g. ``DoBuild[*]``); see :func:`_compile_glob`.
        A multi-index name contains a comma (e.g. ``Cut[*,*]``) and so must be
        quoted to survive the CSV split (standard CSV quoting).
    ``can_slam`` (optional; default ``1``)
        ``1``/``0``; ``0`` carves out an exception so matched variables are
        never slammed.
    ``directions`` (required unless ``can_slam`` is ``0``)
        ``|``-separated ordered list drawn from :data:`VALID_DIRECTIONS`.
    ``priority`` (optional; default ``0``)
        float; the eligible nonant with the largest priority is slammed first.

    Rows are returned in file order; matching is **last-match-wins** so write
    broad defaults first and exceptions after.  Blank lines and whole-line
    comments (starting with ``#``) are ignored anywhere, including before the
    header.

    Raises:
        ValueError: on a missing ``name`` column, an unknown direction token,
            an unparseable ``can_slam``/``priority``, or a ``can_slam=1`` row
            with no directions.
    """
    with open(path, "r", newline="") as f:
        raw_lines = f.readlines()

    # Drop blank lines and whole-line '#' comments, but remember each kept
    # line's original 1-based number for error messages.
    kept = [(i, line) for i, line in enumerate(raw_lines, start=1)
            if line.strip() != "" and not line.strip().startswith("#")]
    if not kept:
        raise ValueError(
            f"slamming directives file {path!r} has no header row")

    reader = csv.DictReader([line for _, line in kept])
    if reader.fieldnames is None or "name" not in reader.fieldnames:
        raise ValueError(
            f"slamming directives file {path!r} must have a 'name' column")

    directives = []
    # DictReader consumes kept[0] as the header, so kept[1:] aligns with rows.
    for (lineno, _), row in zip(kept[1:], reader):
        pattern = (row.get("name") or "").strip()
        if pattern == "":
            continue  # tolerate a stray empty name field
        try:
            can_slam = _parse_can_slam(row.get("can_slam") or "")
            directions = _parse_directions(row.get("directions") or "")
            priority_raw = (row.get("priority") or "").strip()
            priority = float(priority_raw) if priority_raw != "" else 0.0
        except ValueError as e:
            raise ValueError(f"{path}:{lineno}: {e}") from None
        if can_slam and len(directions) == 0:
            raise ValueError(
                f"{path}:{lineno}: row for {pattern!r} has can_slam=1 but "
                "no directions")
        directives.append(
            SlamDirective(pattern, can_slam, directions, priority))
    return directives


def resolve_directive(directives, name):
    """Return the last :class:`SlamDirective` matching ``name``, else ``None``.

    Last-match-wins (gitignore-style): a later row overrides an earlier one.
    """
    match = None
    for d in directives:
        if d.matches(name):
            match = d
    return match


def _finite(x):
    return x is not None and math.isfinite(x)


class Slammer(Extension):
    """Preference-driven in-hub slammer.

    Constructed from ``opt.options["slammer_options"]``, a dict with keys:

    ``directives`` or ``directives_file``
        a pre-parsed list of :class:`SlamDirective` (used directly) or a path
        to a CSV parsed with :func:`parse_directives_file`.
    ``slam_start_iter`` (int, default 1)
        first hub iteration at which slamming may occur.
    ``iters_between_slams`` (int, default 1)
        cadence once started; slam at most once every this many iterations.
    ``rounding_bias`` (float, default 0.0)
        added before rounding integer variables (matches the slam spokes).
    ``verbose`` (bool, default False)
        report each slam on rank 0.

    ``options`` may be passed directly to the constructor by a *consumer* that
    drives the action layer itself (e.g. a stall/cycle detector calling
    :meth:`slam_nonant`); when given it is used in place of
    ``spobj.options["slammer_options"]``, so such a consumer can run its own
    Slammer without disturbing a separately-configured one.
    """

    def __init__(self, spobj, options=None):
        super().__init__(spobj)
        opts = options if options is not None \
            else spobj.options.get("slammer_options", {})
        # Kept for error messages; None when directives are passed in directly.
        self._directives_file = opts.get("directives_file")
        if "directives" in opts:
            self.directives = opts["directives"]
        else:
            self.directives = parse_directives_file(opts["directives_file"])
        self.slam_start_iter = opts.get("slam_start_iter", 1)
        self.iters_between_slams = opts.get("iters_between_slams", 1)
        if self.iters_between_slams < 1:
            raise ValueError("iters_between_slams must be a positive integer")
        self.rounding_bias = opts.get("rounding_bias", 0.0)
        self.verbose = opts.get("verbose", False)

        # (ndn, i) -> value, for every slam done so far.  Slams are sticky:
        # this also records what would have to be released to ever un-slam.
        self._slammed = {}
        # Built in pre_iter0(): the eligibility map and bookkeeping.
        self._directive_of = {}   # (ndn, i) -> SlamDirective (can_slam only)
        self._name_of = {}        # (ndn, i) -> xvar.name
        self._modeler_fixed = set()

    # ------------------------------------------------------------------ #
    # Trigger layer (WHEN)
    # ------------------------------------------------------------------ #
    def iteration_for_slam(self, phiter):
        """Whether to slam at hub iteration ``phiter``: at or after
        ``slam_start_iter``, then once every ``iters_between_slams`` iterations.
        A pure function of the iteration counter, so it is identical on every
        rank with no communication."""
        if phiter < self.slam_start_iter:
            return False
        return (phiter - self.slam_start_iter) % self.iters_between_slams == 0

    # ------------------------------------------------------------------ #
    # Preferences layer (WHO / WHERE)
    # ------------------------------------------------------------------ #
    def pre_iter0(self):
        """Build the eligibility map by matching each nonant's name against the
        directives.  The same nonant has the same ``xvar.name`` in every
        scenario, so one representative scenario suffices and the map is
        scenario-independent and multistage-clean.  Raises ``ValueError`` (on
        every rank) if any directive pattern matches no nonant anywhere."""
        rep = self.opt.local_scenarios[self.opt.local_scenario_names[0]]
        local_names = []
        for ndn_i, xvar in rep._mpisppy_data.nonant_indices.items():
            name = xvar.name
            self._name_of[ndn_i] = name
            local_names.append(name)
            if xvar.fixed:
                self._modeler_fixed.add(ndn_i)
                continue  # modeler-fixed nonants are never slammed
            d = resolve_directive(self.directives, name)
            if d is not None and d.can_slam:
                self._directive_of[ndn_i] = d

        # A directive pattern that matches no nonanticipative variable anywhere
        # is almost always a typo, so it is a hard error.  Each rank sees only
        # the nonants of the nodes it owns, so reduce the per-directive match
        # flags across the hub (ROOT spans every hub rank); a pattern matched on
        # some other rank is then not falsely flagged (this matters for
        # multistage).  The reduced result is identical on every rank, so all
        # ranks raise together.
        local_match = np.array(
            [1 if any(d.matches(nm) for nm in local_names) else 0
             for d in self.directives], 'i')
        global_match = np.zeros(len(self.directives), 'i')
        self.opt.comms["ROOT"].Allreduce(
            [local_match, MPI.INT], [global_match, MPI.INT], op=MPI.MAX)
        unmatched = [d.pattern for d, hit in zip(self.directives, global_match)
                     if not hit]
        if unmatched:
            where = f" in {self._directives_file}" if self._directives_file \
                    else ""
            raise ValueError(
                f"slamming directives{where}: pattern(s) matched no "
                "nonanticipative variable: "
                f"{', '.join(repr(p) for p in unmatched)}")

    # ------------------------------------------------------------------ #
    # Action layer (WHAT)
    # ------------------------------------------------------------------ #
    def miditer(self):
        phiter = self.opt._PHIter
        if not self.iteration_for_slam(phiter):
            return
        self._slam_one()

    def _slam_one(self, candidates=None):
        """Select the highest-priority eligible nonant and slam it; return 1 if
        one was slammed, else 0.

        When ``candidates`` is given (a set/collection of ``(ndn, i)`` keys),
        only those nonants are considered -- the entry point a stall/cycle
        detector uses to slam the worst of *its* flagged nonants while still
        honoring the directives file's priority ranking.  Ineligible or
        no-applicable-direction candidates are skipped, so this effectively
        walks the priority order until it finds one it can slam.

        Selection uses only globally-consistent inputs (file-supplied priority
        and name; the fixed mask, which is coherent because slamming/fixing is
        applied to all scenarios on all ranks; and, for the detector-driven
        path, a rank-identical ``candidates`` set), so every rank picks the same
        nonant with no communication.  This holds when the nonant catalog is
        rank-coherent (two-stage, or single-rank multistage); node-split
        multistage would need a cross-rank reduction to agree on the selection,
        which is not done here.
        """
        rep = self.opt.local_scenarios[self.opt.local_scenario_names[0]]
        surrogates = rep._mpisppy_data.all_surrogate_nonants

        best_ndn_i = None
        best_key = None    # (priority, name) for max-priority, name-tiebreak
        for ndn_i, d in self._directive_of.items():
            if candidates is not None and ndn_i not in candidates:
                continue
            if not self._slam_eligible(rep, ndn_i, surrogates):
                continue
            if self._first_applicable_direction(rep, ndn_i, d.directions) is None:
                continue  # no applicable direction -> not eligible this event
            name = self._name_of[ndn_i]
            # Largest priority wins; ties broken by name (ascending) so the
            # choice is deterministic and identical across ranks.
            key = (d.priority, name)
            if best_key is None or key[0] > best_key[0] \
               or (key[0] == best_key[0] and key[1] < best_key[1]):
                best_ndn_i = ndn_i
                best_key = key

        if best_ndn_i is None:
            return 0  # nothing eligible this event
        return 1 if self.slam_nonant(best_ndn_i) else 0

    def _slam_eligible(self, rep, ndn_i, surrogates):
        """Whether ``ndn_i`` may be slammed: it is in the directive map (checked
        by the caller / :meth:`slam_nonant`), not already slammed, not
        modeler-/fixer-fixed, and not a surrogate nonant (§7.1 of the design).
        Every input is rank-coherent, so the verdict is identical on every rank.
        """
        if ndn_i in self._slammed:
            return False
        xvar = rep._mpisppy_data.nonant_indices[ndn_i]
        if xvar.fixed:
            return False
        if xvar in surrogates:
            return False
        return True

    def slam_nonant(self, ndn_i):
        """Action layer: slam the specific nonant ``ndn_i`` per its directive.

        This is the reusable action entry the slamming design (§7) anticipated
        for an external stall/cycle detector: rather than the built-in
        iteration trigger choosing *which* nonant, a caller that has already
        decided ``ndn_i`` is stuck invokes this directly.  The nonant is slammed
        in the first applicable direction of its directive (so a detector need
        only supply the directives file and the cycling nonant).

        Returns ``True`` if the nonant was slammed, ``False`` if it is not in
        the directive map, is not eligible (already fixed/slammed/surrogate), or
        no direction applies this event.

        **Collective safety:** a ``min``/``max`` direction triggers a per-node
        ``Allreduce`` (:meth:`_node_extremum`); a caller iterating several
        nonants must therefore call this for the *same* nonants in the *same
        order on every rank*.  That holds when the caller's target set comes
        from a rank-identical reduction (as the W-oscillation interrupter's
        does) and the nonant catalog is rank-coherent (two-stage / single-rank
        multistage), matching :meth:`_slam_one`'s own assumption.
        """
        d = self._directive_of.get(ndn_i)
        if d is None:
            return False
        rep = self.opt.local_scenarios[self.opt.local_scenario_names[0]]
        surrogates = rep._mpisppy_data.all_surrogate_nonants
        if not self._slam_eligible(rep, ndn_i, surrogates):
            return False
        choice = self._first_applicable_direction(rep, ndn_i, d.directions)
        if choice is None:
            return False

        direction, value = choice
        if direction in ("min", "max"):
            value = self._node_extremum(ndn_i, direction)
            xvar = rep._mpisppy_data.nonant_indices[ndn_i]
            if xvar.is_binary() or xvar.is_integer():
                value = round(value + self.rounding_bias)

        self._fix_everywhere(ndn_i, value)
        self._slammed[ndn_i] = value
        self._report(self._name_of[ndn_i], value, direction, d.priority)
        return True

    def _first_applicable_direction(self, scenario, ndn_i, directions):
        """Return ``(direction, value)`` for the first applicable direction, or
        ``None`` if none applies.  For ``min``/``max`` the value is deferred
        (returned as ``None``) and computed by a reduction only if this nonant
        is the one actually slammed (see :meth:`_node_extremum`)."""
        xvar = scenario._mpisppy_data.nonant_indices[ndn_i]
        is_int = xvar.is_binary() or xvar.is_integer()
        for direction in directions:
            if direction == "lb":
                if _finite(xvar.lb):
                    return ("lb", xvar.lb)
            elif direction == "ub":
                if _finite(xvar.ub):
                    return ("ub", xvar.ub)
            elif direction == "nearest":
                lb_ok, ub_ok = _finite(xvar.lb), _finite(xvar.ub)
                if lb_ok or ub_ok:
                    xb = pyo.value(scenario._mpisppy_model.xbars[ndn_i])
                    if lb_ok and ub_ok:
                        val = xvar.lb if abs(xb - xvar.lb) <= abs(xvar.ub - xb) \
                              else xvar.ub
                    else:
                        val = xvar.lb if lb_ok else xvar.ub
                    return ("nearest", val)
            elif direction == "anywhere":
                xb = pyo.value(scenario._mpisppy_model.xbars[ndn_i])
                if _finite(xb):
                    val = round(xb + self.rounding_bias) if is_int else xb
                    return ("anywhere", val)
            elif direction in ("min", "max"):
                # A solved variable's value is always finite, so min/max is
                # always applicable; the actual extremum is reduced later.
                return (direction, None)
        return None

    def _node_extremum(self, ndn_i, direction):
        """min/max of the selected variable's value across all scenarios at its
        node, reduced over the per-node communicator the hub already uses for
        x-bar.  Paid only when a min/max slam actually fires."""
        ndn = ndn_i[0]
        op = MPI.MIN if direction == "min" else MPI.MAX
        local = None
        for s in self.opt.local_scenarios.values():
            v = pyo.value(s._mpisppy_data.nonant_indices[ndn_i])
            if local is None:
                local = v
            else:
                local = min(local, v) if direction == "min" else max(local, v)
        local_arr = np.array([local], dtype="d")
        global_arr = np.empty(1, dtype="d")
        self.opt.comms[ndn].Allreduce(
            [local_arr, MPI.DOUBLE], [global_arr, MPI.DOUBLE], op=op)
        return float(global_arr[0])

    def _fix_everywhere(self, ndn_i, value):
        """Fix the nonant to ``value`` in every local scenario and push the
        change to any persistent solver."""
        for s in self.opt.local_scenarios.values():
            xvar = s._mpisppy_data.nonant_indices[ndn_i]
            xvar.fix(value)
            if is_persistent(s._solver_plugin):
                s._solver_plugin.update_var(xvar)

    def _report(self, name, value, direction, priority):
        if self.verbose and self.opt.cylinder_rank == 0:
            print(f"(rank0) Slammer: slammed {name} to {value} via "
                  f"'{direction}' (priority {priority}); "
                  f"total slammed = {len(self._slammed)}")

    def post_everything(self):
        if self.verbose and self.opt.cylinder_rank == 0:
            print(f"(rank0) Slammer: final count of nonants slammed = "
                  f"{len(self._slammed)}")
