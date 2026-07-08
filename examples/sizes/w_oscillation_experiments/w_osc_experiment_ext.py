###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Experiment PH extension: measure W-oscillation and try to break it.

This is a **research harness**, not a product feature. It reuses the merged
detection primitives from ``mpisppy.extensions.w_oscillation`` (PR1) to flag
cycling nonants each iteration, optionally applies a configured *intervention*
to those nonants, and records two metrics per iteration:

* ``zc_flagged`` -- how many nonants the zero-crossings detector still flags.
  Scale-free (it counts sign changes), so it can be fooled by anything that
  freezes W even when nothing has converged.
* ``primal_gap`` = ``sum_s p_s |x_s - xbar|`` -- the ground truth: it shrinks
  only when the scenarios actually agree.

Interventions (``intervention`` in ``opt.options["wosc_experiment_options"]``):

``none``        observe only (the plain baseline).
``w_damping``   rescale the latest dual (W) step on flagged nonants by ``factor``.
``rho_reduce``  multiply flagged nonants' rho by ``factor`` each acting iteration,
                floored at ``floorfrac * original_rho`` (rho must stay > 0).
``w_reset``     one-shot: zero ALL W and scale ALL rho by ``factor`` (keep xbar).
``fix``         the slam analogue: fix **one** flagged nonant per
                ``iters_between_slams`` cooldown to its per-scenario max (the
                Watson-Woodruff §2.4 / Slammer ``max`` remedy), solver-correctly
                (``update_var`` for persistent solvers). Detector-driven, so it
                targets the *cycling* nonants -- which is what makes it work.

Mutating W/rho ``Param`` values is picked up by the per-iteration
``set_objective`` that persistent solvers already issue (the mechanism the
rho-updating extensions rely on), so those interventions need no explicit solver
push; ``fix`` pushes the fixed var to the solver explicitly.

The harness runs **serially** (one rank); detection and the primal gap are
computed over the local scenarios (all of them), so no MPI reductions are
needed. See ``README.md`` for the findings this harness produced.
"""

import csv

import numpy as np

from mpisppy.extensions.extension import Extension
from mpisppy.extensions.w_oscillation import zero_crossings_detect
from mpisppy.utils.sputils import is_persistent


class WOscExperimentMonitor(Extension):
    """Detect W-oscillation, optionally intervene, and record zc + primal gap."""

    def __init__(self, spobj):
        super().__init__(spobj)
        opts = self.opt.options["wosc_experiment_options"]
        self.intervention = opts.get("intervention", "none")
        self.factor = float(opts.get("factor", 0.5))
        self.floorfrac = float(opts.get("floorfrac", 1e-3))
        self.start_iter = int(opts.get("start_iter", 10))
        self.window = int(opts.get("window", 20))
        self.min_flagged = int(opts.get("min_scenarios_flagged", 1))
        self.iters_between_slams = int(opts.get("iters_between_slams", 3))
        self.out_csv = opts["out_csv"]
        self.zc_params = {
            "tol": opts.get("tol", 1e-6),
            "window": self.window,
            "thresh_w_crossings": opts.get("thresh_w_crossings", 2),
            "thresh_diff_crossings": opts.get("thresh_diff_crossings", 3),
            "thresh_diffs_ratio": opts.get("thresh_diffs_ratio", 0.2),
        }
        self._traj = {}          # sname -> list of W vectors (ring buffer)
        self._rows = []          # (iter, zc_flagged, primal_gap, mean_abs_W)
        self._orig_rho = {}      # (id(scenario), ndn_i) -> original rho
        self._reset_fired = False
        self._fixed = set()      # ndn_i already fixed (fix intervention)
        self._last_fix_iter = None

    def pre_iter0(self):
        rep = next(iter(self.opt.local_scenarios.values()))
        surrogates = getattr(rep._mpisppy_data, "all_surrogate_nonants", set())
        self._ndn_i = [ndn_i for ndn_i in rep._mpisppy_data.nonant_indices
                       if ndn_i not in surrogates]
        for sname in self.opt.local_scenarios:
            self._traj[sname] = []

    def miditer(self):
        phiter = self.opt._PHIter
        self._capture()
        flagged = self._flagged_nonants()
        # record the state ENTERING this iteration (before any intervention),
        # uniformly across arms, so the trajectories are comparable.
        self._record(phiter, flagged)
        if phiter >= self.start_iter and flagged:
            self._intervene(flagged, phiter)

    def _capture(self):
        for sname, s in self.opt.local_scenarios.items():
            W = s._mpisppy_model.W
            vec = np.fromiter((W[k]._value for k in self._ndn_i), dtype="d",
                              count=len(self._ndn_i))
            buf = self._traj[sname]
            buf.append(vec)
            if len(buf) > self.window:
                buf.pop(0)

    def _flagged_nonants(self):
        nJ = len(self._ndn_i)
        counts = np.zeros(nJ, dtype=int)
        for buf in self._traj.values():
            if len(buf) < 2:
                continue
            arr = np.array(buf)  # (T, nJ)
            for j in range(nJ):
                if zero_crossings_detect(arr[:, j], **self.zc_params)["flagged"]:
                    counts[j] += 1
        return {j for j in range(nJ) if counts[j] >= self.min_flagged}

    def _record(self, phiter, flagged):
        gap = wabs = 0.0
        n = 0
        for s in self.opt.local_scenarios.values():
            p = getattr(s, "_mpisppy_probability", 1.0)
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                xbar = s._mpisppy_model.xbars[ndn_i]._value
                gap += p * abs(xvar._value - xbar)
                wabs += abs(s._mpisppy_model.W[ndn_i]._value)
                n += 1
        self._rows.append((phiter, len(flagged), gap, wabs / n if n else 0.0))

    def _intervene(self, flagged, phiter):
        if self.intervention == "none":
            return
        if self.intervention == "w_reset":
            self._w_reset()
            return
        if self.intervention == "fix":
            self._fix_one(flagged, phiter)
            return
        for j in flagged:
            ndn_i = self._ndn_i[j]
            for s in self.opt.local_scenarios.values():
                model = s._mpisppy_model
                if ndn_i not in model.W:
                    continue
                if self.intervention == "w_damping":
                    rho = model.rho[ndn_i]._value
                    xdiff = (s._mpisppy_data.nonant_indices[ndn_i]._value
                             - model.xbars[ndn_i]._value)
                    model.W[ndn_i]._value -= (1.0 - self.factor) * rho * xdiff
                elif self.intervention == "rho_reduce":
                    key = (id(s), ndn_i)
                    self._orig_rho.setdefault(key, model.rho[ndn_i]._value)
                    floor = self.floorfrac * self._orig_rho[key]
                    model.rho[ndn_i]._value = max(
                        model.rho[ndn_i]._value * self.factor, floor)

    def _w_reset(self):
        """One-shot: zero all W and scale all rho (keep xbar) -- a warm restart."""
        if self._reset_fired:
            return
        for s in self.opt.local_scenarios.values():
            model = s._mpisppy_model
            for ndn_i in model.rho:
                key = (id(s), ndn_i)
                self._orig_rho.setdefault(key, model.rho[ndn_i]._value)
                if ndn_i in model.W:
                    model.W[ndn_i]._value = 0.0
                floor = self.floorfrac * self._orig_rho[key]
                model.rho[ndn_i]._value = max(
                    model.rho[ndn_i]._value * self.factor, floor)
        self._reset_fired = True

    def _fix_one(self, flagged, phiter):
        """Slam analogue: fix one flagged nonant per cooldown to its per-scenario
        max, in every scenario (so they agree), pushing the fix to persistent
        solvers -- the same mechanism Slammer uses (its ``_fix_everywhere``)."""
        if (self._last_fix_iter is not None
                and phiter - self._last_fix_iter < self.iters_between_slams):
            return
        cand = sorted(j for j in flagged
                      if self._ndn_i[j] not in self._fixed)
        if not cand:
            return
        ndn_i = self._ndn_i[cand[0]]
        value = max(s._mpisppy_data.nonant_indices[ndn_i]._value
                    for s in self.opt.local_scenarios.values())
        for s in self.opt.local_scenarios.values():
            xvar = s._mpisppy_data.nonant_indices[ndn_i]
            xvar.fix(value)
            if is_persistent(s._solver_plugin):
                s._solver_plugin.update_var(xvar)
        self._fixed.add(ndn_i)
        self._last_fix_iter = phiter

    def post_everything(self):
        if self.opt.cylinder_rank != 0:
            return
        with open(self.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "zc_flagged", "primal_gap", "mean_abs_W"])
            w.writerows(self._rows)


# A module-level instance is not created; the driver passes the class to
# vanilla.extension_adder. (The generic --user-defined-extensions loader, which
# scans for instances, is not used here.)
