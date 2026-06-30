###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Detect oscillation / cycling in the PH dual weight (W) vector.

This is a *hub* extension that observes the W vector while a synchronous PH hub
runs and reports detected oscillation to a CSV.  It is **pure observation**: it
attaches no rho or fixing changes, so a run with ``--detect-W-oscillations``
produces the same optimization trajectory as one without.  Interrupting the
oscillation (rho reduction / slamming) is a separate piece of work.

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

import mpisppy.MPI as MPI
from mpisppy.extensions.extension import Extension


# Method names recognized in the JSON "methods" block.
VALID_METHODS = ("zero_crossings", "w_hash_recurrence")
VALID_REPORT_MODES = ("on_detect", "final", "every_check")

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
    """Detect (and report) oscillation in the W vector; see the module docstring.

    Constructed from ``opt.options["w_oscillation_options"]``, a dict with:

    ``detect_config`` (dict) or ``detect_json`` (path)
        the parsed control dict, or a path parsed with
        :func:`parse_detect_config`.
    ``verbose`` (bool, default False)
        print a rank-0 summary line at the end.
    """

    def __init__(self, spobj):
        super().__init__(spobj)
        opts = spobj.options.get("w_oscillation_options", {})
        if "detect_config" in opts:
            self.cfg = validate_detect_config(opts["detect_config"])
        else:
            self.cfg = parse_detect_config(opts["detect_json"])
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

        if self._is_writer:
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

    def miditer(self):
        self._capture()
        phiter = self.opt._PHIter
        if phiter < self.cfg["warmup_iters"]:
            return
        if (phiter - self.cfg["warmup_iters"]) % self.cfg["check_every"] != 0:
            return
        self._evaluate_and_report(phiter)

    # ------------------------------------------------------------------ #
    def _evaluate_and_report(self, phiter):
        """Run the active detectors, reduce across scenarios per node, and emit
        rows for nonants that meet the reporting threshold."""
        nJ = len(self._ndn_i)
        if "zero_crossings" in self._methods:
            self._eval_zero_crossings(phiter, nJ)
        if "w_hash_recurrence" in self._methods:
            self._eval_w_hash(phiter, nJ)

    def _threshold(self, ndn):
        frac = self.cfg["min_frac_to_report"]
        if frac is not None:
            return max(1, int(np.ceil(frac * self._n_scen_total[ndn])))
        return int(self.cfg["min_scenarios_to_report"])

    def _eval_zero_crossings(self, phiter, nJ):
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
            if g_flag[j] >= self._threshold(self._nodes[j]):
                self._emit_agg(phiter, j, "zero_crossings",
                               int(g_flag[j]),
                               max_w_crossings=int(g_w[j]),
                               max_diff_crossings=int(g_d[j]),
                               max_diffs_ratio=round(float(g_r[j]), 6))
        self._emit_scen(scen_rows)

    def _eval_w_hash(self, phiter, nJ):
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
                self._emit_agg(phiter, j, "w_hash_recurrence",
                               self._n_scen_total[self._nodes[j]],
                               cycle_period=period)
                if self.cfg["per_scenario_csv"]:
                    for sname, s in self.opt.local_scenarios.items():
                        scen_rows.append((
                            phiter, self._nodes[j], sname, self._names[j],
                            "w_hash_recurrence", "", "", "",
                            round(float(s._mpisppy_model.W[self._ndn_i[j]]._value),
                                  9)))
        self._emit_scen(scen_rows)

    def _threshold_period(self):
        return self._methods["w_hash_recurrence"]["min_period"]

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
        if self.cfg["report_mode"] == "final":
            self._evaluate_and_report(self.opt._PHIter)
        if self._is_writer:
            if self._agg_file is not None:
                self._agg_file.close()
            if self._scen_file is not None:
                self._scen_file.close()
        if self.verbose and self.opt.cylinder_rank == 0:
            print(f"(rank0) WOscillationMonitor: {self._n_flag_events} "
                  f"oscillation report event(s); see {self.cfg['output_csv']}")
