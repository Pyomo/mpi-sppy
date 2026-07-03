###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Detect (and optionally interrupt) oscillation / cycling in the PH dual
weight (W) vector.

This is a *hub* extension that observes the W vector while a synchronous PH hub
runs and reports detected oscillation to a CSV.  With only
``--detect-W-oscillations`` it is **pure observation**: it attaches no rho or
fixing changes, so the optimization trajectory is identical to a run without
the flag.  With ``--interrupt-W-oscillations`` it additionally *acts* on the
detected oscillation to break it -- by **damping W** (rescaling the dual-weight
step) on every cycling nonant and/or **slamming** one of them (fixing it via the
existing :class:`Slammer <mpisppy.extensions.slammer.Slammer>` action layer,
with at least ``iters_between_slams`` iterations between successive slams).
Interruption implies detection; see :func:`parse_interrupt_config`.

Two detection methods are available, selected and parameterized from a JSON
control file (see :func:`parse_detect_config`):

``zero_crossings``
    A port of PySP's ``sorgw`` plugin: for each (scenario, nonant) W trajectory,
    count sign changes of W and of the consecutive differences, plus a damping
    ratio of the back half vs. the front half of |ΔW|.  Flag the trajectory when
    any threshold is exceeded.  Per-trajectory and embarrassingly parallel.

``w_hash_recurrence``
    The Watson--Woodruff "Progressive Hedging Innovations" (Computational
    Management Science, 2011) §2.4 cycle detector: hash the *per-scenario* W
    vector for each nonant and flag a recurrence of that vector.  In the
    distributed setting we form a **distribution-independent signature** by
    summing identity-mixed per-scenario 64-bit hashes (an additive/multiset
    hash; see Bellare & Micciancio 1997, Clarke et al. ASIACRYPT 2003) and
    reducing with ``MPI.SUM``; recurrence of the signature is the cycle.

Per-(scenario, nonant) verdicts are aggregated across scenarios with per-node
``SUM``/``MAX`` reductions, and cylinder rank 0 writes the CSV.  See
``doc/designs/w_oscillation_design.md`` for the full design, and the wtracker
utilities (``mpisppy/utils/w_utils/``) for general W-trajectory logging and
moving statistics.
"""

import csv
import json

import numpy as np
import pyomo.environ as pyo

import mpisppy.MPI as MPI
from mpisppy import global_toc
from mpisppy.extensions.extension import Extension
from mpisppy.extensions.slammer import Slammer


# Method names recognized in the JSON "methods" block.
VALID_METHODS = ("zero_crossings", "w_hash_recurrence")
VALID_REPORT_MODES = ("on_detect", "final", "every_check")
# Interruption actions recognized in the interrupt JSON "action" field.
VALID_ACTIONS = ("w_damping", "slam", "both")

_MASK64 = (1 << 64) - 1

# numpy dtype -> MPI datatype for the arrays reduced by _reduce_per_node. Keyed
# explicitly (rather than an int/else fallback) so an unexpected dtype fails
# loudly instead of being silently treated as a double.
_NUMPY_TO_MPI = {
    np.dtype(np.int32): MPI.INT,
    np.dtype(np.float64): MPI.DOUBLE,
}

# Default parameters per method; a method's omitted JSON keys fall back here.
_ZERO_CROSSINGS_DEFAULTS = {
    "tol": 1e-6,
    "window": None,                 # None -> use the whole retained history
    "thresh_w_crossings": 2,
    "thresh_diff_crossings": 3,
    "thresh_diffs_ratio": 0.2,
}
_W_HASH_DEFAULTS = {
    "window": 20,                   # look back this many checks for a repeat
    "quantum": 1e-6,                # W quantized to this before hashing
    "min_period": 2,                # >=2 so constant W (convergence) is ignored
}

# Interruption (PR2) defaults; an interrupt file's omitted keys fall back here.
_TRIGGER_DEFAULTS = {
    "min_scenarios_flagged": 1,     # act when >= this many scenarios flag a nonant
    "start_iter": 5,                # first iteration at which to act; then every
                                    # iteration a nonant is still flagged
}
_W_DAMPING_DEFAULTS = {
    "factor": 0.5,                  # retained fraction of each dual (W) step on a
                                    # flagged nonant; 0 <= factor < 1
}
_SLAM_DEFAULTS = {
    "iters_between_slams": 3,       # cooldown: iterations that must pass after a
                                    # successful slam before the next one
}


# --------------------------------------------------------------------------- #
# Method A -- zero crossings (pure, per-trajectory; unit-testable, no MPI)
# --------------------------------------------------------------------------- #
def count_sign_changes(values, tol):
    """Number of sign changes in a sequence, ignoring entries with ``|v| < tol``.

    A "sign change" is counted each time the running sign flips from strictly
    positive to strictly negative or vice versa (near-zero entries do not reset
    the running sign).  This matches PySP ``sorgw``'s zero-crossing counter.
    """
    crossings = 0
    above = False
    below = False
    for v in values:
        if abs(v) < tol:
            continue
        if v > 0:
            if below:
                crossings += 1
                below = False
            above = True
        else:
            if above:
                crossings += 1
                above = False
            below = True
    return crossings


def zero_crossings_detect(traj, tol=1e-6, window=None,
                          thresh_w_crossings=2, thresh_diff_crossings=3,
                          thresh_diffs_ratio=0.2):
    """Run the zero-crossing detector on one W trajectory.

    Args:
        traj (sequence of float): the W values over iterations for one
            (scenario, nonant), oldest first.
        tol, window, thresh_* : see :data:`_ZERO_CROSSINGS_DEFAULTS`.

    Returns:
        dict with ``flagged`` (bool) and the stats ``w_crossings`` (int),
        ``diff_crossings`` (int), ``diffs_ratio`` (float).  ``flagged`` is true
        iff any of the three thresholds is met.
    """
    w = list(traj)
    if window is not None and window > 0:
        w = w[-window:]

    w_crossings = count_sign_changes(w, tol)

    diffs = [w[i + 1] - w[i] for i in range(len(w) - 1)]
    diff_crossings = count_sign_changes(diffs, tol)

    # diffs_ratio is a damping check: split |ΔW| into a front (older) and back
    # (newer) half and compare their mean step sizes. If the W swings are
    # shrinking over time (converging), the back-half mean is small relative to
    # the front-half mean and the ratio is well below 1; if the swings are not
    # damping out (sustained oscillation), the ratio stays near or above 1. A
    # high ratio therefore signals oscillation that is failing to settle.
    absdiffs = [abs(d) for d in diffs]
    half = len(absdiffs) // 2
    front = absdiffs[:half]
    back = absdiffs[half:]
    frontavg = sum(front) / len(front) if front else 0.0
    diffs_ratio = (sum(back) / len(back) / frontavg) if (back and frontavg > tol) \
        else 0.0

    flagged = (w_crossings >= thresh_w_crossings
               or diff_crossings >= thresh_diff_crossings
               or diffs_ratio >= thresh_diffs_ratio)
    return {
        "flagged": flagged,
        "w_crossings": w_crossings,
        "diff_crossings": diff_crossings,
        "diffs_ratio": diffs_ratio,
    }


# --------------------------------------------------------------------------- #
# Method B -- W-vector hashing / recurrence
# --------------------------------------------------------------------------- #
def _mix64(x):
    """splitmix64 finalizer: deterministic 64-bit avalanche mix (no salt)."""
    x &= _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    x ^= x >> 31
    return x & _MASK64


def signature_term(scenario_index, w_value, quantum):
    """64-bit hash of ``(scenario_index, round(w_value / quantum))``.

    Mixing the scenario index in keeps per-scenario identity, so the SUM of
    these terms over a node's scenarios is a signature of the whole per-scenario
    W vector that is independent of how scenarios are spread across MPI ranks
    (the sum is commutative; see the module docstring).
    """
    q = int(round(w_value / quantum)) & _MASK64
    return _mix64(_mix64(int(scenario_index) & _MASK64) ^ q)


class RecurrenceTracker:
    """Per-nonant ring buffer of W-vector signatures; flags genuine cycles.

    ``push(sig)`` records the latest signature and returns ``(flagged, period)``.
    A cycle of period ``p`` (``min_period <= p <= window``) is flagged when the
    current signature equals the one ``p`` steps back **and** at least one
    signature in between differs from it -- so a *constant* signature
    (convergence, period 1) is never mistaken for a cycle.
    """

    def __init__(self, window=20, min_period=2):
        self.window = window
        self.min_period = max(2, int(min_period))
        self._buf = []   # most-recent-last, capped at window + 1

    def push(self, sig):
        sig = int(sig) & _MASK64
        self._buf.append(sig)
        if len(self._buf) > self.window + 1:
            self._buf.pop(0)
        n = len(self._buf)
        for p in range(self.min_period, n):
            if self._buf[n - 1 - p] != sig:
                continue
            # equal at distance p; require a genuine change in between
            if any(self._buf[n - 1 - t] != sig for t in range(1, p)):
                return True, p
        return False, 0


# --------------------------------------------------------------------------- #
# JSON control-file parsing / validation
# --------------------------------------------------------------------------- #
def parse_detect_config(path):
    """Load and validate a ``--detect-W-oscillations`` JSON control file.

    Returns the config dict with method parameters filled in from the per-method
    defaults.  Raises ``ValueError`` on a missing ``output_csv``, an empty or
    unknown ``methods`` block, or a bad ``report_mode``.
    """
    with open(path, "r") as f:
        cfg = json.load(f)
    return validate_detect_config(cfg, where=path)


def validate_detect_config(cfg, where="<detect config>"):
    if not isinstance(cfg, dict):
        raise ValueError(f"{where}: top level must be a JSON object")
    if not cfg.get("output_csv"):
        raise ValueError(f"{where}: 'output_csv' is required")

    report_mode = cfg.get("report_mode", "on_detect")
    if report_mode not in VALID_REPORT_MODES:
        raise ValueError(
            f"{where}: report_mode {report_mode!r} not in {VALID_REPORT_MODES}")

    methods = cfg.get("methods")
    if not methods or not isinstance(methods, dict):
        raise ValueError(
            f"{where}: 'methods' must be a non-empty object selecting at least "
            f"one of {VALID_METHODS}")
    resolved = {}
    for name, params in methods.items():
        if name not in VALID_METHODS:
            raise ValueError(
                f"{where}: unknown method {name!r} (legal: {VALID_METHODS})")
        params = dict(params or {})
        defaults = _ZERO_CROSSINGS_DEFAULTS if name == "zero_crossings" \
            else _W_HASH_DEFAULTS
        merged = dict(defaults)
        merged.update(params)
        resolved[name] = merged

    out = {
        "output_csv": cfg["output_csv"],
        "per_scenario_csv": cfg.get("per_scenario_csv"),
        "warmup_iters": int(cfg.get("warmup_iters", 5)),
        "check_every": max(1, int(cfg.get("check_every", 1))),
        "report_mode": report_mode,
        "min_scenarios_to_report": cfg.get("min_scenarios_to_report", 1),
        "min_frac_to_report": cfg.get("min_frac_to_report"),
        "max_history": int(cfg.get("max_history", 50)),
        "methods": resolved,
    }
    return out


def default_detect_config():
    """A sensible default detection config for a run that supplies only
    ``--interrupt-W-oscillations`` (interruption implies detection): both
    methods at their defaults, reporting to ``w_oscillations.csv``."""
    return validate_detect_config({
        "output_csv": "w_oscillations.csv",
        "methods": {"zero_crossings": {}, "w_hash_recurrence": {}},
    })


def parse_interrupt_config(path):
    """Load and validate an ``--interrupt-W-oscillations`` JSON control file
    (see :func:`validate_interrupt_config`)."""
    with open(path, "r") as f:
        cfg = json.load(f)
    return validate_interrupt_config(cfg, where=path)


def validate_interrupt_config(cfg, where="<interrupt config>"):
    """Validate an interruption control dict and fill in defaults.

    Schema (see ``doc/designs/w_oscillation_design.md`` §2.2)::

        {
          "action": "w_damping" | "slam" | "both",
          "trigger": {min_scenarios_flagged, start_iter},
          "w_damping": {"factor": <0..1>},      # if w_damping/both
          "slam": {"directives_file": <path>,   # if slam/both
                   "iters_between_slams": <int >= 1>},
          "detect": { ... }   # optional detection config (else a default is used)
        }

    Raises ``ValueError`` on an unknown/missing ``action``, a ``factor`` not in
    ``[0, 1)``, a ``slam`` action with no ``directives_file``, a
    non-positive ``iters_between_slams``, or a bad trigger value.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"{where}: top level must be a JSON object")
    action = cfg.get("action")
    if action not in VALID_ACTIONS:
        raise ValueError(
            f"{where}: 'action' {action!r} not in {VALID_ACTIONS}")

    trigger_block = cfg.get("trigger")
    if trigger_block is not None and not isinstance(trigger_block, dict):
        raise ValueError(f"{where}: 'trigger' must be a JSON object")

    trig = dict(_TRIGGER_DEFAULTS)
    trig.update(trigger_block or {})
    trigger = {
        "min_scenarios_flagged": int(trig["min_scenarios_flagged"]),
        "start_iter": int(trig["start_iter"]),
    }
    if trigger["min_scenarios_flagged"] < 1:
        raise ValueError(f"{where}: trigger.min_scenarios_flagged must be >= 1")

    out = {"action": action, "trigger": trigger}

    if action in ("w_damping", "both"):
        wd = dict(_W_DAMPING_DEFAULTS)
        wd.update(cfg.get("w_damping") or {})
        wd["factor"] = float(wd["factor"])
        if not (0.0 <= wd["factor"] < 1.0):
            raise ValueError(
                f"{where}: w_damping.factor must be in [0, 1) to damp the W "
                f"step (1.0 is a no-op; got {wd['factor']})")
        out["w_damping"] = wd

    if action in ("slam", "both"):
        slam = dict(_SLAM_DEFAULTS)
        slam.update(cfg.get("slam") or {})
        directives_file = slam.get("directives_file")
        if not directives_file:
            raise ValueError(
                f"{where}: action {action!r} requires "
                "'slam': {{'directives_file': <path>}}")
        iters_between = int(slam["iters_between_slams"])
        if iters_between < 1:
            raise ValueError(
                f"{where}: slam.iters_between_slams must be >= 1 "
                f"(got {iters_between}; 1 means a slam may occur every "
                "iteration)")
        out["slam"] = {"directives_file": directives_file,
                       "iters_between_slams": iters_between}

    if "detect" in cfg:
        out["detect"] = validate_detect_config(
            cfg["detect"], where=f"{where} (detect block)")
    return out


def slam_due(phiter, last_slam_iter, iters_between_slams):
    """Whether a slam may occur at ``phiter`` given the cooldown: true when no
    slam has happened yet (``last_slam_iter`` is None) or at least
    ``iters_between_slams`` iterations have passed since the last successful
    one.  The cooldown exists because the detectors judge a trailing history
    window, which keeps signaling oscillation for a while after a fix -- this
    gives the fix time to re-settle the others (WW §2.1) before concluding that
    another variable must also be fixed.  A pure function of rank-identical
    inputs, so the verdict is identical on every rank."""
    if last_slam_iter is None:
        return True
    return phiter - last_slam_iter >= iters_between_slams


def w_damped(w, rho, xdiff, factor):
    """W after one damping event: undo the fraction ``1 - factor`` of the dual
    step ``rho * xdiff`` that ``Update_W`` just applied.

    ``Update_W`` set ``W += rho * (x - xbar)``; had it used ``factor * rho``
    instead, ``W`` would be smaller by ``(1 - factor) * rho * xdiff``.  This
    rescales that most recent increment to what a lower rho would have produced,
    leaving the proximal rho untouched -- damping the dual (``w``) step that
    Watson--Woodruff §2.1 identifies as "shoot[ing] past" the optimal weight.
    ``factor`` is in ``[0, 1)``: 0 fully cancels the step, near 1 barely damps."""
    return w - (1.0 - factor) * rho * xdiff


# --------------------------------------------------------------------------- #
# The extension
# --------------------------------------------------------------------------- #
# CSV columns for the per-nonant aggregate report (§7 of the design).
_AGG_COLUMNS = [
    "iteration", "node", "variable", "method",
    "num_scenarios_total", "num_scenarios_flagged",
    "max_w_crossings", "max_diff_crossings", "max_diffs_ratio", "cycle_period",
]
_PER_SCEN_COLUMNS = [
    "iteration", "node", "scenario", "variable", "method",
    "w_crossings", "diff_crossings", "diffs_ratio", "w_value",
]


class WOscillationMonitor(Extension):
    """Detect, report, and optionally interrupt W-vector oscillation; see the
    module docstring.

    Constructed from ``opt.options["w_oscillation_options"]``, a dict with:

    ``detect_config`` (dict) or ``detect_json`` (path)
        the parsed detection control dict, or a path parsed with
        :func:`parse_detect_config`.
    ``interrupt_config`` (dict) or ``interrupt_json`` (path)
        (optional) the parsed interruption control dict, or a path parsed with
        :func:`parse_interrupt_config`.  Its presence turns on interruption.
    ``verbose`` (bool, default False)
        print rank-0 summary lines.

    At least one of detection / interruption must be configured.  When only
    interruption is configured, the detection config is taken from the
    interrupt file's optional ``detect`` block, else a default
    (:func:`default_detect_config`).
    """

    def __init__(self, spobj):
        super().__init__(spobj)
        opts = spobj.options.get("w_oscillation_options", {})

        # Detection config (direct dict, by path, or -- below -- derived from
        # the interrupt config).
        self.cfg = None
        if "detect_config" in opts:
            self.cfg = validate_detect_config(opts["detect_config"])
        elif opts.get("detect_json"):
            self.cfg = parse_detect_config(opts["detect_json"])

        # Interruption config (PR2); None when only detecting.
        self._interrupt_cfg = None
        if "interrupt_config" in opts:
            self._interrupt_cfg = validate_interrupt_config(
                opts["interrupt_config"])
        elif opts.get("interrupt_json"):
            self._interrupt_cfg = parse_interrupt_config(opts["interrupt_json"])

        # Interruption needs the detection *engine* to know which nonants are
        # cycling, but writing the cycling *report* (CSV) is opt-in.  The report
        # is written only when detection was explicitly requested -- via
        # --detect-W-oscillations / a detect_config, or a "detect" block in the
        # interrupt file.  A pure --interrupt-W-oscillations run runs the engine
        # (to drive the actions) but writes no report; its activity is announced
        # with a global_toc instead (see _apply_interruption).
        self._report_enabled = self.cfg is not None or (
            self._interrupt_cfg is not None and "detect" in self._interrupt_cfg)

        # Interruption implies detection: derive a detection config if none was
        # supplied directly.
        if self.cfg is None:
            if self._interrupt_cfg is not None and "detect" in self._interrupt_cfg:
                self.cfg = self._interrupt_cfg["detect"]
            elif self._interrupt_cfg is not None:
                self.cfg = default_detect_config()
            else:
                raise ValueError(
                    "WOscillationMonitor requires a detection or interruption "
                    "control file (none supplied)")

        self.verbose = opts.get("verbose", False)

        self._methods = self.cfg["methods"]
        # how many checks of history to retain
        self._history = self.cfg["max_history"]
        if "w_hash_recurrence" in self._methods:
            self._history = max(self._history,
                                self._methods["w_hash_recurrence"]["window"] + 1)

        # Built in pre_iter0():
        self._ndn_i = []        # ordered monitored (ndn, i)
        self._names = []        # parallel xvar.name
        self._nodes = []        # parallel node name
        self._node_pos = {}     # ndn -> list of positions j into the lists above
        self._n_scen_total = {} # ndn -> total scenarios passing through node
        # per local scenario: list (ring) of W vectors over the monitored nonants
        self._traj = {}         # sname -> list[np.ndarray]
        self._recur = {}        # j -> RecurrenceTracker (method B)
        self._reported = set()  # (method, j) already reported (on_detect mode)

        self._agg_writer = None
        self._agg_file = None
        self._scen_writer = None
        self._scen_file = None
        self._n_flag_events = 0

        # Interruption state (PR2), built in pre_iter0() / updated in miditer().
        self._slammer = None        # internal Slammer for the slam action
        self._last_slam_iter = None # iteration of the last successful slam
                                    # (rank-identical; drives the slam cooldown)
        self._n_action_events = 0
        # j -> max scenarios flagging nonant j on the most recent evaluation
        # (rank-identical; the interrupter reads this).
        self._flagged_counts = {}

    @property
    def _is_writer(self):
        return self.opt.cylinder_rank == 0

    # ------------------------------------------------------------------ #
    def pre_iter0(self):
        """Build the monitored-nonant catalog (node-grouped), per-node scenario
        counts, per-method state, and open the CSV(s) on rank 0."""
        rep = self.opt.local_scenarios[self.opt.local_scenario_names[0]]
        surrogates = getattr(rep._mpisppy_data, "all_surrogate_nonants", set())
        for ndn_i, xvar in rep._mpisppy_data.nonant_indices.items():
            if xvar in surrogates:
                continue  # zero-prob surrogate nonants have masked W
            j = len(self._ndn_i)
            self._ndn_i.append(ndn_i)
            self._names.append(xvar.name)
            self._nodes.append(ndn_i[0])
            self._node_pos.setdefault(ndn_i[0], []).append(j)

        # Total scenarios at each node (reduced once; identical on every rank).
        for ndn, positions in self._node_pos.items():
            local = sum(1 for s in self.opt.local_scenarios.values()
                        if any(nd.name == ndn for nd in s._mpisppy_node_list))
            loc = np.array([local], "i")
            glob = np.zeros(1, "i")
            self.opt.comms[ndn].Allreduce([loc, MPI.INT], [glob, MPI.INT],
                                          op=MPI.SUM)
            self._n_scen_total[ndn] = int(glob[0])

        # Stable, rank-identical per-scenario index for the signature hash:
        # position in the global scenario ordering (identical on every rank).
        self._scen_idx = {name: i for i, name
                          in enumerate(self.opt.all_scenario_names)}

        for sname in self.opt.local_scenario_names:
            self._traj[sname] = []
        if "w_hash_recurrence" in self._methods:
            hp = self._methods["w_hash_recurrence"]
            for j in range(len(self._ndn_i)):
                self._recur[j] = RecurrenceTracker(window=hp["window"],
                                                   min_period=hp["min_period"])

        # For the slam action, drive the existing Slammer action layer (the
        # consumer the slamming design anticipated).  Construct it directly (not
        # via the extension machinery) and call its pre_iter0() here, on every
        # rank symmetrically; its iteration-count trigger is never used -- the
        # interrupter calls slam_nonant() on exactly the flagged nonants.
        if self._interrupt_cfg is not None \
                and self._interrupt_cfg["action"] in ("slam", "both"):
            self._slammer = Slammer(self.opt, options={
                "directives_file": self._interrupt_cfg["slam"]["directives_file"],
                "verbose": self.verbose,
            })
            self._slammer.pre_iter0()

        if self._is_writer and self._report_enabled:
            self._agg_file = open(self.cfg["output_csv"], "w", newline="")
            self._agg_writer = csv.writer(self._agg_file)
            self._agg_writer.writerow(_AGG_COLUMNS)
            self._agg_file.flush()
            if self.cfg["per_scenario_csv"]:
                self._scen_file = open(self.cfg["per_scenario_csv"], "w",
                                       newline="")
                self._scen_writer = csv.writer(self._scen_file)
                self._scen_writer.writerow(_PER_SCEN_COLUMNS)
                self._scen_file.flush()

    # ------------------------------------------------------------------ #
    def _capture(self):
        """Append the current (post-update) W vector for each local scenario."""
        for sname, s in self.opt.local_scenarios.items():
            W = s._mpisppy_model.W
            vec = np.fromiter((W[k]._value for k in self._ndn_i), dtype="d",
                              count=len(self._ndn_i))
            buf = self._traj[sname]
            buf.append(vec)
            if len(buf) > self._history:
                buf.pop(0)

    def _detect_due(self, phiter):
        """Whether to run the detectors for *reporting* at ``phiter``: at or
        after ``warmup_iters``, then every ``check_every`` iterations.  A pure
        function of the iteration counter, so it is identical on every rank."""
        w = self.cfg["warmup_iters"]
        if phiter < w:
            return False
        return (phiter - w) % self.cfg["check_every"] == 0

    def _act_due(self, phiter):
        """Whether to *interrupt* at ``phiter`` (interruption configured): at or
        after the trigger's ``start_iter``.  W-damping has no inter-action
        cadence -- once started, it acts every iteration a nonant is still
        flagged; the slam action is additionally throttled by its own cooldown
        (:func:`slam_due`, applied in :meth:`_apply_interruption`).  Pure
        function of the counter -> identical on every rank."""
        return phiter >= self._interrupt_cfg["trigger"]["start_iter"]

    def miditer(self):
        self._capture()
        phiter = self.opt._PHIter
        # The report is opt-in (self._report_enabled); a detection check only
        # matters when reporting is on.  In "final" report mode all rows are
        # deferred to the single end-of-run evaluation in post_everything.
        # The engine still runs whenever an interruption action is due, to
        # find the cycling nonants to act on.
        report_due = (self._report_enabled
                      and self.cfg["report_mode"] != "final"
                      and self._detect_due(phiter))
        act_due = self._interrupt_cfg is not None and self._act_due(phiter)
        if not (report_due or act_due):
            return
        # Evaluate the detectors (always populates the rank-identical flagged
        # counts the interrupter reads; emits report rows only when a report is
        # due).  The decision to evaluate is a pure function of phiter, so the
        # collective reductions inside fire symmetrically on every rank.
        flagged = self._evaluate(phiter, emit=report_due)
        if act_due:
            self._apply_interruption(phiter, flagged)

    # ------------------------------------------------------------------ #
    def _evaluate(self, phiter, emit=True):
        """Run the active detectors, reduce across scenarios per node, and (if
        ``emit``) write report rows for nonants meeting the reporting threshold.

        Returns the per-nonant flagged-scenario counts ``{j: count}`` (the
        rank-identical SUM reduction; for method B the participating scenario
        count), regardless of ``emit``, for the interrupter's own threshold."""
        nJ = len(self._ndn_i)
        self._flagged_counts = {}
        if "zero_crossings" in self._methods:
            self._eval_zero_crossings(phiter, nJ, emit)
        if "w_hash_recurrence" in self._methods:
            self._eval_w_hash(phiter, nJ, emit)
        return self._flagged_counts

    def _record_flagged(self, j, count):
        """Record (max over methods) how many scenarios flagged nonant ``j``."""
        if count > self._flagged_counts.get(j, 0):
            self._flagged_counts[j] = count

    def _threshold(self, ndn):
        frac = self.cfg["min_frac_to_report"]
        if frac is not None:
            return max(1, int(np.ceil(frac * self._n_scen_total[ndn])))
        return int(self.cfg["min_scenarios_to_report"])

    def _eval_zero_crossings(self, phiter, nJ, emit=True):
        p = self._methods["zero_crossings"]
        # local per-nonant accumulators
        n_flag = np.zeros(nJ, "i")
        mx_w = np.zeros(nJ, "i")
        mx_d = np.zeros(nJ, "i")
        mx_r = np.zeros(nJ, "d")
        scen_rows = []  # per-scenario detail (gathered if requested)
        for sname, buf in self._traj.items():
            if len(buf) < 2:
                continue
            arr = np.array(buf)  # (T, nJ)
            for j in range(nJ):
                res = zero_crossings_detect(
                    arr[:, j], tol=p["tol"], window=p["window"],
                    thresh_w_crossings=p["thresh_w_crossings"],
                    thresh_diff_crossings=p["thresh_diff_crossings"],
                    thresh_diffs_ratio=p["thresh_diffs_ratio"])
                if res["flagged"]:
                    n_flag[j] += 1
                    mx_w[j] = max(mx_w[j], res["w_crossings"])
                    mx_d[j] = max(mx_d[j], res["diff_crossings"])
                    mx_r[j] = max(mx_r[j], res["diffs_ratio"])
                    if self.cfg["per_scenario_csv"]:
                        scen_rows.append((
                            phiter, self._nodes[j], sname, self._names[j],
                            "zero_crossings", res["w_crossings"],
                            res["diff_crossings"], round(res["diffs_ratio"], 6),
                            round(float(arr[-1, j]), 9)))

        g_flag, g_w, g_d, g_r = self._reduce_per_node(
            nJ, [(n_flag, MPI.SUM), (mx_w, MPI.MAX),
                 (mx_d, MPI.MAX), (mx_r, MPI.MAX)])
        for j in range(nJ):
            self._record_flagged(j, int(g_flag[j]))
            if emit and g_flag[j] >= self._threshold(self._nodes[j]):
                self._emit_agg(phiter, j, "zero_crossings",
                               int(g_flag[j]),
                               max_w_crossings=int(g_w[j]),
                               max_diff_crossings=int(g_d[j]),
                               max_diffs_ratio=round(float(g_r[j]), 6))
        if emit:
            self._emit_scen(scen_rows)

    def _eval_w_hash(self, phiter, nJ, emit=True):
        p = self._methods["w_hash_recurrence"]
        q = p["quantum"]
        # local partial signature per nonant (sum of identity-mixed hashes)
        partial = np.zeros(nJ, dtype=np.uint64)
        for sname, s in self.opt.local_scenarios.items():
            sidx = self._scen_idx[sname]
            W = s._mpisppy_model.W
            for j, k in enumerate(self._ndn_i):
                term = signature_term(sidx, W[k]._value, q)
                partial[j] = (int(partial[j]) + term) & _MASK64
        glob = self._reduce_signatures(nJ, partial)

        scen_rows = []
        for j in range(nJ):
            flagged, period = self._recur[j].push(int(glob[j]))
            if flagged and period >= self._threshold_period():
                # A recurring vector implicates every scenario at the node.
                self._record_flagged(j, self._n_scen_total[self._nodes[j]])
                if emit:
                    self._emit_agg(phiter, j, "w_hash_recurrence",
                                   self._n_scen_total[self._nodes[j]],
                                   cycle_period=period)
                    if self.cfg["per_scenario_csv"]:
                        for sname, s in self.opt.local_scenarios.items():
                            scen_rows.append((
                                phiter, self._nodes[j], sname, self._names[j],
                                "w_hash_recurrence", "", "", "",
                                round(float(
                                    s._mpisppy_model.W[self._ndn_i[j]]._value),
                                    9)))
        if emit:
            self._emit_scen(scen_rows)

    def _threshold_period(self):
        return self._methods["w_hash_recurrence"]["min_period"]

    # ------------------------------------------------------------------ #
    # Interruption (PR2): act on the flagged nonants to break the oscillation.
    # ------------------------------------------------------------------ #
    def _apply_interruption(self, phiter, flagged):
        """Act on the nonants flagged by at least ``min_scenarios_flagged``
        scenarios: damp W on **every** such nonant and/or slam **one** of them,
        per the configured action.  Slamming is additionally throttled by the
        ``iters_between_slams`` cooldown (:func:`slam_due`) -- the detectors'
        trailing-history flags outlive a fix, so without the cooldown a slam
        would land every iteration until the history flushes.

        ``flagged``, the eligibility tests, and the cooldown state are
        rank-identical, so all ranks act on the same nonants in the same order
        -- this matters because the slam action may trigger a per-node
        ``Allreduce`` (the min/max extremum), which must be reached
        symmetrically."""
        thresh = self._interrupt_cfg["trigger"]["min_scenarios_flagged"]
        targets = [j for j in range(len(self._ndn_i))
                   if flagged.get(j, 0) >= thresh]
        if not targets:
            return
        action = self._interrupt_cfg["action"]
        damp = action in ("w_damping", "both")
        slam = action in ("slam", "both") and slam_due(
            phiter, self._last_slam_iter,
            self._interrupt_cfg["slam"]["iters_between_slams"])
        if not (damp or slam):
            return  # slam-only action, cooling down: nothing to do or announce
        if damp:
            self._damp_w(targets)
        n_slam = self._slam_targets(targets) if slam else 0
        if n_slam:
            # Start the cooldown only on a *successful* slam; a no-candidate
            # attempt (n_slam == 0) retries on the next flagged iteration.
            self._last_slam_iter = phiter
        self._n_action_events += 1
        # Announce the interruption activity unconditionally (rank-0 gated) --
        # this is the only output a pure --interrupt-W-oscillations run produces;
        # the cycling report (CSV) is opt-in (self._report_enabled).
        bits = []
        if damp:
            # len(targets) is rank-identical (a rank's *local* touched count
            # can be smaller under node-split multistage distributions).
            bits.append(f"damped W on {len(targets)} nonant(s)")
        if slam:
            bits.append(f"slammed {n_slam} nonant(s)")
        elif action == "both":
            bits.append("slam cooling down")
        global_toc(
            f"W-oscillation interruption [iter {phiter}]: {len(targets)} "
            f"nonant(s) flagged; " + "; ".join(bits),
            self.opt.cylinder_rank == 0)

    def _damp_w(self, targets):
        """Damp the dual weight on **every** flagged nonant, in every local
        scenario, by rescaling the increment ``Update_W`` just applied:
        ``W -= (1 - factor) * rho * (x - xbar)`` (see :func:`w_damped`).

        W is a mutable Pyomo Param in the objective, so changing its value is
        picked up by the per-iteration ``set_objective`` that ``solve_one``
        issues for persistent solvers -- no explicit solver push is needed (the
        same mechanism the rho-updating extensions rely on).  The proximal rho
        is left unchanged; only the dual step is damped."""
        factor = self._interrupt_cfg["w_damping"]["factor"]
        for j in targets:
            ndn_i = self._ndn_i[j]
            for s in self.opt.local_scenarios.values():
                W = s._mpisppy_model.W
                if ndn_i not in W:
                    continue  # scenario does not pass through this node
                rho = pyo.value(s._mpisppy_model.rho[ndn_i])
                xdiff = s._mpisppy_data.nonant_indices[ndn_i]._value \
                    - s._mpisppy_model.xbars[ndn_i]._value
                W[ndn_i]._value = w_damped(W[ndn_i]._value, rho, xdiff, factor)

    def _slam_targets(self, targets):
        """Slam **at most one** flagged nonant this iteration -- the
        highest-priority one that can actually be slammed -- via the Slammer
        action layer.  Fixing is drastic, and fixing just one cycling variable
        often re-settles the others (WW §2.1), so we reuse Slammer's own
        priority-ranked, one-per-call :meth:`~mpisppy.extensions.slammer.Slammer._slam_one`
        restricted to the flagged set.  Return the number slammed (0 or 1)."""
        candidates = {self._ndn_i[j] for j in targets}
        return self._slammer._slam_one(candidates=candidates)

    # ------------------------------------------------------------------ #
    def _reduce_per_node(self, nJ, arrays_and_ops):
        """Allreduce each (array, op) over the per-node communicators, returning
        rank-identical global arrays of length ``nJ``."""
        outs = [np.zeros(nJ, a.dtype) for a, _ in arrays_and_ops]
        for ndn, positions in self._node_pos.items():
            comm = self.opt.comms[ndn]
            pos = np.array(positions)
            for (a, op), out in zip(arrays_and_ops, outs):
                try:
                    mpitype = _NUMPY_TO_MPI[a.dtype]
                except KeyError:
                    raise ValueError(
                        f"_reduce_per_node got array of unsupported dtype "
                        f"{a.dtype}; add it to _NUMPY_TO_MPI.")
                loc = np.ascontiguousarray(a[pos])
                res = np.zeros(len(pos), a.dtype)
                comm.Allreduce([loc, mpitype], [res, mpitype], op=op)
                out[pos] = res
        return outs

    def _reduce_signatures(self, nJ, partial):
        """Sum-reduce the uint64 partial signatures over the per-node comms
        (modular mod 2^64), so the result is independent of scenario->rank map."""
        out = np.zeros(nJ, dtype=np.uint64)
        for ndn, positions in self._node_pos.items():
            comm = self.opt.comms[ndn]
            pos = np.array(positions)
            loc = np.ascontiguousarray(partial[pos])
            res = np.zeros(len(pos), dtype=np.uint64)
            comm.Allreduce([loc, MPI.UINT64_T], [res, MPI.UINT64_T], op=MPI.SUM)
            out[pos] = res
        return out

    # ------------------------------------------------------------------ #
    def _emit_agg(self, phiter, j, method, num_flagged, **stats):
        mode = self.cfg["report_mode"]
        if mode == "on_detect":
            if (method, j) in self._reported:
                return
            self._reported.add((method, j))
        self._n_flag_events += 1
        if not self._is_writer:
            return
        row = {
            "iteration": phiter, "node": self._nodes[j],
            "variable": self._names[j], "method": method,
            "num_scenarios_total": self._n_scen_total[self._nodes[j]],
            "num_scenarios_flagged": num_flagged,
            "max_w_crossings": "", "max_diff_crossings": "",
            "max_diffs_ratio": "", "cycle_period": "",
        }
        row.update(stats)
        self._agg_writer.writerow([row[c] for c in _AGG_COLUMNS])
        self._agg_file.flush()

    def _emit_scen(self, scen_rows):
        if not self.cfg["per_scenario_csv"]:
            return
        gathered = self.opt.comms["ROOT"].gather(scen_rows, root=0)
        if self._is_writer and gathered is not None:
            for chunk in gathered:
                for row in chunk:
                    self._scen_writer.writerow(row)
            self._scen_file.flush()

    # ------------------------------------------------------------------ #
    def post_everything(self):
        if self._report_enabled and self.cfg["report_mode"] == "final":
            self._evaluate(self.opt._PHIter)
        if self._is_writer:
            if self._agg_file is not None:
                self._agg_file.close()
            if self._scen_file is not None:
                self._scen_file.close()
        if self.verbose and self.opt.cylinder_rank == 0:
            if self._report_enabled:
                msg = (f"(rank0) WOscillationMonitor: {self._n_flag_events} "
                       f"oscillation report event(s); see "
                       f"{self.cfg['output_csv']}")
            else:
                msg = "(rank0) WOscillationMonitor: report disabled"
            if self._interrupt_cfg is not None:
                msg += (f"; {self._n_action_events} interruption action "
                        f"event(s) [{self._interrupt_cfg['action']}]")
            print(msg)
