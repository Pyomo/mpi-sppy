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
``prox_boost``  global: on a detected cycle at/after ``start_iter``, scale the
                quadratic penalty by ``boost_factor``. Scheduling knobs:
                ``boost_iters`` (window length; >> budget => held to the end),
                ``refire_cooldown`` (0 one-shot; > 0 re-open a window on each
                re-detected cycle after that many idle iterations), and
                ``escalate_mult`` (> 1 => hold and ramp the multiplier x this
                every ``escalate_every`` iters while the cycle persists, capped
                at ``escalate_cap``, persistence judged by ``escalate_on`` =
                gap|zc). Prox-*only*: it scales the ``prox_on`` coefficient on
                ``ProxExpr``, so the W dual step (which uses ``rho`` directly,
                not ``prox_on``) is untouched -- a tightening of the anchor to
                xbar without inflating the dual price. Distinct from the
                rho-level sweep, which moves prox *and* dual together.

Mutating W/rho/prox_on ``Param`` values is picked up by the per-iteration
``set_objective`` that persistent solvers already issue (the mechanism the
rho-updating extensions rely on), so those interventions need no explicit solver
push; ``fix`` pushes the fixed var to the solver explicitly.

The harness runs **serially** (one rank); detection and the primal gap are
computed over the local scenarios (all of them), so no MPI reductions are
needed. See ``README.md`` for the findings this harness produced.
"""

import csv
import json

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
        self.boost_factor = float(opts.get("boost_factor", 10.0))
        self.boost_iters = int(opts.get("boost_iters", 5))
        # prox_boost re-fire: 0 => one-shot (a single window); > 0 => re-open a
        # window each time a cycle is re-detected, waiting this many iterations
        # after the previous window closed (a cooldown, like the fix arm).
        self.refire_cooldown = int(opts.get("refire_cooldown", 0))
        # prox_boost escalate: > 1 => held-and-ramping. Turn the boost on and,
        # every escalate_every iterations that the cycle still persists, multiply
        # the current multiplier by escalate_mult (capped at escalate_cap) --
        # keep raising the penalty until it is strong enough to force consensus.
        # Persistence is judged by escalate_on: "gap" (primal gap, the ground
        # truth) or "zc" (the detector, which keeps flagging a low-amplitude
        # residual and so ramps to the cap).
        self.escalate_mult = float(opts.get("escalate_mult", 1.0))
        self.escalate_every = int(opts.get("escalate_every", 5))
        self.escalate_cap = float(opts.get("escalate_cap", 1e6))
        self.escalate_on = opts.get("escalate_on", "gap")
        self.escalate_gap_thresh = float(opts.get("escalate_gap_thresh", 5.0))
        # when True, at the end of the run measure the quality of what the arm
        # produced: the expected cost of committing to the consensus xbar
        # (x-quality, inner bound) and the Lagrangian bound from the final W
        # (W-quality, outer bound). Written to a {out_csv}.bounds.json sidecar.
        self.measure_quality = bool(opts.get("measure_quality", False))
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
        self._boost_fired = False   # prox_boost: has any window opened yet?
        self._boost_end_iter = None  # phiter at which to revert the open window
        self._boost_last_end = None  # phiter the previous window closed (cooldown)
        self._orig_prox_on = None    # prox_on value to restore after the boost
        self._boost_current = None   # current multiplier (escalate mode)
        self._last_escalate_iter = None  # phiter of the last escalation step

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
        # prox_boost owns a multi-iteration lifecycle (the window outlives any
        # single detection), so it is managed every iteration, not only when
        # something is flagged.
        if self.intervention == "prox_boost":
            self._manage_prox_boost(flagged, phiter)
        elif phiter >= self.start_iter and flagged:
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
        rep = None
        for s in self.opt.local_scenarios.values():
            rep = s
            p = getattr(s, "_mpisppy_probability", 1.0)
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                xbar = s._mpisppy_model.xbars[ndn_i]._value
                gap += p * abs(xvar._value - xbar)
                wabs += abs(s._mpisppy_model.W[ndn_i]._value)
                n += 1
        # the live prox multiplier (1x normally; boosted/ramping under prox_boost)
        prox_mult = getattr(rep._mpisppy_model, "prox_on", None)
        prox_mult = prox_mult._value if prox_mult is not None else 1.0
        self._rows.append((phiter, len(flagged), gap,
                           wabs / n if n else 0.0, prox_mult))

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

    def _manage_prox_boost(self, flagged, phiter):
        """Global prox tightening, temporary and (optionally) repeating. When a
        cycle is detected at/after ``start_iter``, scale the quadratic penalty by
        ``boost_factor`` for ``boost_iters`` iterations, then revert to 1x. With
        ``refire_cooldown == 0`` this fires exactly once (one-shot); with
        ``refire_cooldown > 0`` a fresh window re-opens whenever a cycle is
        re-detected at least that many iterations after the previous window
        closed. Prox-only: ``prox_on`` multiplies ``ProxExpr`` in the objective
        but the W dual step uses ``rho`` directly, so the dual price is
        unaffected -- a repeated tightening of the anchor to xbar, not a dual
        move. The persistent solver's per-iteration ``set_objective`` re-reads
        ``prox_on``, so no explicit solver push is needed."""
        if self.escalate_mult > 1.0:              # held-and-ramping mode
            self._manage_prox_escalate(flagged, phiter)
            return
        if self._boost_end_iter is not None:      # a window is currently open
            if phiter >= self._boost_end_iter:
                self._set_prox_multiplier(self._orig_prox_on)
                self._boost_end_iter = None
                self._boost_last_end = phiter     # start the cooldown clock
            return
        if phiter < self.start_iter or not flagged:
            return
        if self.refire_cooldown <= 0 and self._boost_fired:
            return                                # one-shot already spent
        if (self._boost_last_end is not None
                and phiter - self._boost_last_end < self.refire_cooldown):
            return                                # still cooling down
        if self._orig_prox_on is None:            # capture 1x once, reuse it
            rep = next(iter(self.opt.local_scenarios.values()))
            self._orig_prox_on = rep._mpisppy_model.prox_on._value
        self._set_prox_multiplier(self.boost_factor)
        self._boost_fired = True
        self._boost_end_iter = phiter + self.boost_iters

    def _manage_prox_escalate(self, flagged, phiter):
        """Escalating held prox boost: on the first detected cycle at/after
        ``start_iter``, turn the boost on at ``boost_factor`` and hold it, then
        every ``escalate_every`` iterations that the cycle still persists,
        multiply the current multiplier by ``escalate_mult`` (capped at
        ``escalate_cap``). Keep raising the penalty until it is strong enough to
        force consensus. Prox-only throughout -- ``rho`` (the dual step) is never
        touched. Persistence is judged by ``escalate_on``: ``gap`` (primal gap,
        the ground truth -- stops once the cycle is actually broken) or ``zc``
        (the scale-free detector -- keeps flagging the low-amplitude residual and
        so ramps to the cap)."""
        if not self._boost_fired:
            if phiter >= self.start_iter and flagged:
                rep = next(iter(self.opt.local_scenarios.values()))
                self._orig_prox_on = rep._mpisppy_model.prox_on._value
                self._boost_current = self.boost_factor
                self._set_prox_multiplier(self._boost_current)
                self._boost_fired = True
                self._last_escalate_iter = phiter
            return
        if phiter - self._last_escalate_iter < self.escalate_every:
            return
        self._last_escalate_iter = phiter
        if self._boost_current >= self.escalate_cap:
            return
        persists = (flagged if self.escalate_on == "zc"
                    else self._rows[-1][2] > self.escalate_gap_thresh)
        if persists:
            self._boost_current = min(self._boost_current * self.escalate_mult,
                                      self.escalate_cap)
            self._set_prox_multiplier(self._boost_current)

    def _set_prox_multiplier(self, value):
        """Set the prox_on Param (the coefficient on ProxExpr) on every scenario.
        prox_on is declared Binary, so assign to ``._value`` to bypass domain
        validation; the objective just reads it as a scalar coefficient."""
        for s in self.opt.local_scenarios.values():
            s._mpisppy_model.prox_on._value = value

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

    def _measure_solution_quality(self):
        """At the end of the run, score what the arm actually produced:

        * inner bound (x-quality) -- fix every scenario to the consensus xbar,
          re-optimize the recourse, and take the expected cost. This is the true
          cost of *committing to the consensus decision* (a valid upper bound).
        * outer bound (W-quality) -- the Lagrangian bound from the final W
          (``post_solve_bound``: unfix, prox off, W on, solve, ``Ebound``). Tight
          only if the duals reached good values.

        Both reuse the hub's own PHBase machinery. Returns a dict; either bound is
        ``None`` if its solve fails (e.g. a rounded xbar infeasible somewhere)."""
        opt = self.opt
        rep = next(iter(opt.local_scenarios.values()))
        cache = {}
        for node in rep._mpisppy_node_list:
            ndn = node.name
            nlen = rep._mpisppy_data.nlens[ndn]
            cache[ndn] = [rep._mpisppy_model.xbars[(ndn, i)]._value
                          for i in range(nlen)]
        inner = None
        try:
            opt._save_nonants()
            opt._fix_nonants(cache)
            opt.solve_loop(dis_W=True, dis_prox=True, gripe=True, warmstart=True)
            inner = float(opt.Eobjective())
        except Exception as e:                      # noqa: BLE001 (research code)
            print(f"inner-bound solve failed: {e}")
        outer = None
        try:                                        # unfixes internally
            outer = float(opt.post_solve_bound())
        except Exception as e:                      # noqa: BLE001
            print(f"outer-bound solve failed: {e}")
        return {"inner": inner, "outer": outer}

    def post_everything(self):
        if self.opt.cylinder_rank != 0:
            return
        with open(self.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "zc_flagged", "primal_gap", "mean_abs_W",
                        "prox_mult"])
            w.writerows(self._rows)
        if self.measure_quality:
            q = self._measure_solution_quality()
            with open(self.out_csv.rsplit(".csv", 1)[0] + ".bounds.json",
                      "w") as f:
                json.dump(q, f)


# A module-level instance is not created; the driver passes the class to
# vanilla.extension_adder. (The generic --user-defined-extensions loader, which
# scans for instances, is not used here.)
