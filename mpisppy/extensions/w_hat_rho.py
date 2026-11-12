###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""rho_i ~ |W-hat_i| / |x_i - xbar_i|, W-hat from symmetric-NAC LP-EF duals.

Two-stage only. With w_hat_rho_ef_num_scens=N the EF uses the first N
scenarios and the order stat of |W-hat| over them replaces |W-hat_s|.
Pin mode (w_hat_rho_pin, or w_hat_rho_xhat_path): W-hat_s from the duals
of x_s == xhat in each (LP-relaxed) local scenario, centered; xhat from
the npy file, else the xbar of the (first-N) LP-EF.
"""

import numpy as np
import pyomo.environ as pyo

import mpisppy.extensions.dyn_rho_base
import mpisppy.MPI as MPI
from mpisppy import global_toc
from mpisppy.utils.rho_utils import assign_rho_with_fallback
from mpisppy.utils.sputils import find_active_objective


def _order_stat(vmin, vmean, vmax, a):
    if a <= 0.5:
        return vmin + 2.0 * a * (vmean - vmin)
    return (2.0 * vmean - vmax) + 2.0 * a * (vmax - vmean)


def _relax_integers(m):
    saved = []
    for v in m.component_data_objects(pyo.Var, descend_into=True):
        if v.is_integer() and not v.fixed:
            lb, ub = v.bounds
            saved.append((v, v.domain, v.lower, v.upper))
            v.domain = pyo.Reals
            v.setlb(lb)
            v.setub(ub)
    return saved


def _restore_integers(saved):
    for v, dom, lb, ub in saved:
        v.domain = dom
        v.setlb(lb)
        v.setub(ub)


def pinned_lp_w_hat(scens, xhat, solver_name="gurobi", solver_options=None):
    """{sname: -(duals of x_s == xhat)} for the LP-relaxed scenarios (uncentered).
    Restores the models; nonant values are left at xhat."""
    solver = pyo.SolverFactory(solver_name.replace("_persistent", "_direct"))
    for k, v in (solver_options or {}).items():
        solver.options[k] = v
    w = {}
    for sn, s in scens.items():
        vl = s._mpisppy_node_list[0].nonant_vardata_list
        s._w_hat_rho_pin = pyo.Constraint(
            range(len(vl)), rule=lambda m, j: vl[j] == float(xhat[j]))
        had_dual = s.component("dual") is not None
        if not had_dual:
            s.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        saved = _relax_integers(s)
        try:
            tc = solver.solve(s).solver.termination_condition
            if tc != pyo.TerminationCondition.optimal:
                raise RuntimeError(f"w_hat_rho: pinned LP for {sn} terminated with {tc}")
            w[sn] = np.array([-s.dual[s._w_hat_rho_pin[j]] for j in range(len(vl))])
        finally:
            _restore_integers(saved)
            s.del_component("_w_hat_rho_pin")
            if not had_dual:
                s.del_component("dual")
    return w


def nac_ef_w_hat(scenario_names, scenario_creator, scenario_creator_kwargs=None,
                 solver_name="gurobi", solver_options=None, return_probs=False,
                 return_xbar=False):
    """{sname: W-hat array} from the LP-EF with x_s == xbar.
    Probabilities are renormalized over scenario_names."""
    kwargs = scenario_creator_kwargs or {}
    scens = {sn: scenario_creator(sn, **kwargs) for sn in scenario_names}
    S = len(scens)
    probs = {}
    for sn, s in scens.items():
        p = getattr(s, "_mpisppy_probability", "uniform")
        probs[sn] = 1.0 / S if p in (None, "uniform") else float(p)
    tot = sum(probs.values())
    probs = {sn: p / tot for sn, p in probs.items()}

    ef = pyo.ConcreteModel()
    obj = 0.0
    sense = None
    for sn, s in scens.items():
        if len(s._mpisppy_node_list) != 1:
            raise RuntimeError("w_hat_rho is two-stage only")
        o = find_active_objective(s)
        sense = o.sense if sense is None else sense
        if o.sense != sense:
            raise RuntimeError("w_hat_rho: mixed objective senses")
        obj = obj + probs[sn] * o.expr
        o.deactivate()
        ef.add_component(sn, s)
    ef.EF_Obj = pyo.Objective(expr=obj, sense=sense)

    n = len(scens[scenario_names[0]]._mpisppy_node_list[0].nonant_vardata_list)
    ef.xbar = pyo.Var(range(n))
    ef.nac = pyo.Constraint(pyo.Any)
    for sn, s in scens.items():
        vl = s._mpisppy_node_list[0].nonant_vardata_list
        if len(vl) != n:
            raise RuntimeError(f"w_hat_rho: nonant count differs in {sn}")
        for j, v in enumerate(vl):
            ef.nac[sn, j] = v == ef.xbar[j]

    pyo.TransformationFactory("core.relax_integer_vars").apply_to(ef)
    ef.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    solver = pyo.SolverFactory(solver_name.replace("_persistent", "_direct"))
    for k, v in (solver_options or {}).items():
        solver.options[k] = v
    res = solver.solve(ef)
    tc = res.solver.termination_condition
    if tc != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"w_hat_rho: LP-EF terminated with {tc}")

    w = {sn: np.array([-ef.dual[ef.nac[sn, j]] / probs[sn] for j in range(n)])
         for sn in scenario_names}
    out = (w,)
    if return_probs:
        out += (probs,)
    if return_xbar:
        out += (np.array([pyo.value(ef.xbar[j]) for j in range(n)]),)
    return out if len(out) > 1 else w


class WHatRho(mpisppy.extensions.dyn_rho_base.Dyn_Rho_extension_base):
    """options['w_hat_rho_options']: cfg, scenario_creator_kwargs."""

    def __init__(self, opt):
        o = opt.options["w_hat_rho_options"]
        cfg = o["cfg"]
        super().__init__(opt, cfg)
        self.ph = opt
        self.creator_kwargs = o.get("scenario_creator_kwargs")
        self.multiplier = cfg.get("w_hat_rho_multiplier", 1.0)
        self.alpha = cfg.get("w_hat_rho_order_stat", 0.5)
        self.rel_bound = cfg.get("w_hat_rho_relative_bound", 1e-6)
        self.ef_num_scens = cfg.get("w_hat_rho_ef_num_scens", None)
        self.xhat_path = cfg.get("w_hat_rho_xhat_path", None)
        self.pin = bool(cfg.get("w_hat_rho_pin", False)) or self.xhat_path is not None
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("w_hat_rho_order_stat must be in [0, 1]")
        self._fallback_rho = opt.options["defaultPHrho"]
        self._rho_report_state = {}

    def _small(self):
        N = self.ef_num_scens
        return (not self.pin) and N is not None and N < len(self.ph.all_scenario_names)

    def _ef(self):
        names = self.ph.all_scenario_names
        kw = dict(self.creator_kwargs or {})
        N = self.ef_num_scens
        if N is not None and N < len(names):
            if N < 1:
                raise ValueError("w_hat_rho_ef_num_scens must be >= 1")
            names = names[:N]
            if "num_scens" in kw:
                kw["num_scens"] = N
        w, probs, xbar = nac_ef_w_hat(names, self.ph.scenario_creator, kw,
                                      solver_name=self.cfg.solver_name,
                                      return_probs=True, return_xbar=True)
        return names, w, probs, xbar

    def _pin_w_hat(self):
        """{local sname: centered pinned-LP W-hat}."""
        ph = self.ph
        xhat = None
        if ph.cylinder_rank == 0:
            if self.xhat_path is not None:
                from mpisppy.confidence_intervals.ciutils import read_xhat
                xhat = np.asarray(read_xhat(self.xhat_path)["ROOT"], dtype=float)
            else:
                xhat = self._ef()[3]
        xhat = ph.mpicomm.bcast(xhat, root=0)
        ph._save_nonants()
        w = pinned_lp_w_hat(ph.local_scenarios, xhat, solver_name=self.cfg.solver_name)
        ph._restore_nonants()
        loc = sum(s._mpisppy_data.prob_coeff["ROOT"] * w[sn]
                  for sn, s in ph.local_scenarios.items())
        mean = np.zeros_like(loc)
        ph.comms["ROOT"].Allreduce([loc, MPI.DOUBLE], [mean, MPI.DOUBLE], op=MPI.SUM)
        return {sn: v - mean for sn, v in w.items()}

    def _w_hat(self):
        """Full/pin: {sname: W-hat}. Small: order stat of |W-hat| (array)."""
        ph = self.ph
        if self.pin:
            return self._pin_w_hat()
        out = None
        if ph.cylinder_rank == 0:
            names, w, probs, _ = self._ef()
            if self._small():
                A = np.abs(np.array([w[sn] for sn in names]))
                pv = np.array([probs[sn] for sn in names])
                out = _order_stat(A.min(axis=0), pv @ A, A.max(axis=0), self.alpha)
            else:
                out = w
        return ph.mpicomm.bcast(out, root=0)

    def compute_and_update_rho(self):
        ph = self.ph
        w = self._w_hat()
        small = self._small()
        n = None
        loc_min = loc_max = loc_mean = None
        for sn, s in ph.local_scenarios.items():
            node = s._mpisppy_node_list[0]
            x = np.array([v._value for v in node.nonant_vardata_list])
            xbar = np.array([s._mpisppy_model.xbars["ROOT", i]._value
                             for i in range(len(x))])
            d = np.abs(x - xbar)
            dmax = d.max() if d.max() > 0 else 1.0
            floor = self.rel_bound * np.abs(x)
            d = np.where(d <= floor, np.maximum(dmax, floor), d)
            r = (w if small else np.abs(w[sn])) / d
            p = s._mpisppy_data.prob_coeff["ROOT"]
            if n is None:
                n = len(r)
                loc_min = np.full(n, np.inf)
                loc_max = np.full(n, -np.inf)
                loc_mean = np.zeros(n)
            np.minimum(loc_min, r, out=loc_min)
            np.maximum(loc_max, r, out=loc_max)
            loc_mean += p * r

        gmin, gmax, gmean = np.zeros(n), np.zeros(n), np.zeros(n)
        comm = ph.comms["ROOT"]
        comm.Allreduce([loc_min, MPI.DOUBLE], [gmin, MPI.DOUBLE], op=MPI.MIN)
        comm.Allreduce([loc_max, MPI.DOUBLE], [gmax, MPI.DOUBLE], op=MPI.MAX)
        comm.Allreduce([loc_mean, MPI.DOUBLE], [gmean, MPI.DOUBLE], op=MPI.SUM)
        rho = self.multiplier * _order_stat(gmin, gmean, gmax, self.alpha)

        assign_rho_with_fallback(ph, self._fallback_rho, self.__class__.__name__,
                                 self._rho_report_state,
                                 lambda s, ndn_i: rho[ndn_i[1]],
                                 reason="a zero W-hat")
        global_toc(f"Rho values updated by {self.__class__.__name__} Extension",
                   ph.cylinder_rank == 0)

    def post_iter0(self):
        if any(len(s._mpisppy_node_list) != 1
               for s in self.ph.local_scenarios.values()):
            raise RuntimeError("w_hat_rho is two-stage only")
        self.compute_and_update_rho()

    def miditer(self):
        pass
