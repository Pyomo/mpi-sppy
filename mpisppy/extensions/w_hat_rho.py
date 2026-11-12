###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""rho_i ~ |W-hat_i| / |x_i - xbar_i|, W-hat from symmetric-NAC LP-EF duals.

Two-stage only.
"""

import numpy as np
import pyomo.environ as pyo

import mpisppy.extensions.dyn_rho_base
import mpisppy.MPI as MPI
from mpisppy import global_toc
from mpisppy.utils.rho_utils import assign_rho_with_fallback
from mpisppy.utils.sputils import find_active_objective


def nac_ef_w_hat(scenario_names, scenario_creator, scenario_creator_kwargs=None,
                 solver_name="gurobi", solver_options=None):
    """{sname: W-hat array} from the LP-EF with x_s == xbar."""
    kwargs = scenario_creator_kwargs or {}
    scens = {sn: scenario_creator(sn, **kwargs) for sn in scenario_names}
    S = len(scens)
    probs = {}
    for sn, s in scens.items():
        p = getattr(s, "_mpisppy_probability", "uniform")
        probs[sn] = 1.0 / S if p in (None, "uniform") else float(p)

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

    return {sn: np.array([-ef.dual[ef.nac[sn, j]] / probs[sn] for j in range(n)])
            for sn in scenario_names}


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
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("w_hat_rho_order_stat must be in [0, 1]")
        self._fallback_rho = opt.options["defaultPHrho"]
        self._rho_report_state = {}

    def _w_hat(self):
        ph = self.ph
        w = None
        if ph.cylinder_rank == 0:
            w = nac_ef_w_hat(ph.all_scenario_names, ph.scenario_creator,
                             self.creator_kwargs,
                             solver_name=self.cfg.solver_name)
        return ph.mpicomm.bcast(w, root=0)

    def compute_and_update_rho(self):
        ph = self.ph
        w = self._w_hat()
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
            r = np.abs(w[sn]) / d
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
        a = self.alpha
        if a <= 0.5:
            rho = gmin + 2.0 * a * (gmean - gmin)
        else:
            rho = (2.0 * gmean - gmax) + 2.0 * a * (gmax - gmean)
        rho = self.multiplier * rho

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
