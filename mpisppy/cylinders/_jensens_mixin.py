###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Mixin used by spokes that support --*-try-jensens-first.

Two-stage only. See doc/jensens_bound_design.md for the full design,
including the convexity precondition that governs when the outer-bound
path is valid.

The mixin only provides helpers; the caller (each spoke's main()) is
responsible for deciding when to call them and what to do with the
results (send_bound for outer-bound spokes, evaluate + update_if_improving
for xhat spokes).
"""

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

import mpisppy.utils.sputils as sputils


class _JensensMixin:

    def _jensens_enabled(self):
        return "jensens" in self.opt.options

    def _jensens_build_ev(self):
        """Build and return the expected-value scenario model.

        Shared by outer-bound and inner-bound (xhat) paths. The two-stage
        precondition is enforced here; the integer-safety check is NOT
        run here because the xhat path tolerates integer recourse.
        Outer-bound callers must call _jensens_assert_safe_for_outer_bound
        on the returned model before using it for a bound.
        """
        j = self.opt.options["jensens"]
        ev_creator = j["expected_value_creator"]
        if ev_creator is None:
            raise RuntimeError(
                "try-jensens-first was requested but no "
                "expected_value_creator was threaded into the spoke. "
                "This is a cfg_vanilla wiring bug."
            )
        kwargs = j.get("scenario_creator_kwargs") or {}
        sname = self.opt.all_scenario_names[0]
        ev_model = ev_creator(sname, **kwargs)
        if not hasattr(ev_model, "_mpisppy_node_list"):
            raise RuntimeError(
                "expected_value_creator must return a model with "
                "_mpisppy_node_list attached (via sputils.attach_root_node)."
            )
        if len(ev_model._mpisppy_node_list) != 1:
            raise RuntimeError(
                "Jensen's bound is two-stage only; expected_value_creator "
                f"returned a model with {len(ev_model._mpisppy_node_list)} "
                "tree nodes."
            )
        return ev_model

    def _jensens_assert_safe_for_outer_bound(self, ev_model):
        """Raise if the EV model has integer/binary Vars outside the
        nonant set. Outer-bound (lower-bounder) path only. The xhat
        (inner-bound) path does NOT call this — see design doc §1."""
        sputils.assert_jensen_integer_safe(ev_model)

    def _jensens_solve(self, ev_model):
        """Solve ev_model with the spoke's configured solver.

        Returns (obj_value, nonant_values) where nonant_values is a list
        in the order of ev_model._mpisppy_node_list[0].nonant_vardata_list
        (i.e. ROOT).
        """
        solver_name = self.opt.options["solver_name"]
        solver_options = self.opt.options.get("iter0_solver_options") or {}
        solver = pyo.SolverFactory(solver_name)
        for k, v in solver_options.items():
            solver.options[k] = v
        results = solver.solve(ev_model, tee=False)
        tc = results.solver.termination_condition
        if tc not in (TerminationCondition.optimal,
                      TerminationCondition.feasible):
            raise RuntimeError(
                f"Jensen's EV solve failed with termination_condition={tc}"
            )
        obj = pyo.value(sputils.find_active_objective(ev_model))
        root = ev_model._mpisppy_node_list[0]
        nonant_values = [pyo.value(v) for v in root.nonant_vardata_list]
        return obj, nonant_values

    def _jensens_pack_nonant_cache(self, nonant_values):
        """Pack EV nonant values into the dict-by-nodename cache format
        consumed by Xhat_Eval._fix_nonants / .evaluate. Two-stage: ROOT only."""
        return {"ROOT": np.array(nonant_values, dtype='d')}
