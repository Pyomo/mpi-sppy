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


def assert_jensen_integer_safe(scenario):
    """Raise if any integer/binary Var is outside the nonant set.

    Necessary (not sufficient) check that recourse is convex in the
    random parameters. Jensen's bound is only valid under that
    assumption; convexity in the random parameters cannot be checked
    statically, but the presence of integer recourse is a common
    failure mode that is easy to detect.

    Callers: only the lower-bounder (outer-bound) Jensen's path. Inner
    bounds (xhatters) tolerate integer recourse because the average-
    scenario solution is only used as a candidate xhat, which is then
    honestly evaluated across the real scenarios.

    Reads nonants from scenario._mpisppy_node_list (set by
    attach_root_node / attach_nodes), so this works on a model that has
    NOT yet been processed by SPBase — which is exactly the state an
    average scenario is in when this check runs.
    """
    nonant_ids = set()
    for node in scenario._mpisppy_node_list:
        for v in node.nonant_vardata_list:
            nonant_ids.add(id(v))
    for v in scenario.component_data_objects(pyo.Var, descend_into=True):
        if id(v) in nonant_ids:
            continue
        if v.is_integer() or v.is_binary():
            raise RuntimeError(
                f"Jensen's bound requires convex recourse, but non-nonant "
                f"integer/binary Var found: {v.name}. Disable the "
                f"--*-try-jensens-first flag, or reformulate."
            )


class _JensensMixin:

    def _jensens_enabled(self):
        return "jensens" in self.opt.options

    def _jensens_build_avg(self):
        """Build and return the average scenario model.

        Shared by outer-bound and inner-bound (xhat) paths. The two-stage
        precondition is enforced here; the integer-safety check is NOT
        run here because the xhat path tolerates integer recourse.
        Outer-bound callers must call _jensens_assert_safe_for_outer_bound
        on the returned model before using it for a bound.
        """
        j = self.opt.options["jensens"]
        avg_creator = j["average_scenario_creator"]
        if avg_creator is None:
            raise RuntimeError(
                "try-jensens-first was requested but no "
                "average_scenario_creator was threaded into the spoke. "
                "This is a cfg_vanilla wiring bug."
            )
        kwargs = j.get("scenario_creator_kwargs") or {}
        sname = self.opt.all_scenario_names[0]
        avg_scenario = avg_creator(sname, **kwargs)
        if not hasattr(avg_scenario, "_mpisppy_node_list"):
            raise RuntimeError(
                "average_scenario_creator must return a model with "
                "_mpisppy_node_list attached (via sputils.attach_root_node)."
            )
        if len(avg_scenario._mpisppy_node_list) != 1:
            raise RuntimeError(
                "Jensen's bound is two-stage only; average_scenario_creator "
                f"returned a model with {len(avg_scenario._mpisppy_node_list)} "
                "tree nodes."
            )
        return avg_scenario

    def _jensens_assert_safe_for_outer_bound(self, avg_scenario):
        """Raise if the average scenario has integer/binary Vars outside
        the nonant set. Outer-bound (lower-bounder) path only. The xhat
        (inner-bound) path does NOT call this — see design doc §1."""
        assert_jensen_integer_safe(avg_scenario)

    def _jensens_solve(self, avg_scenario):
        """Solve avg_scenario with the spoke's configured solver.

        Returns (outer_bound, nonant_values).

        outer_bound is the solver's best dual bound (results.problem.lower_bound
        for minimize, .upper_bound for maximize) — NOT the incumbent
        objective value. With the dual bound, a non-zero MIP gap on this
        solve does not invalidate Jensen's outer bound; with the incumbent
        it would. For an LP the two coincide.

        nonant_values is a list in the order of
        avg_scenario._mpisppy_node_list[0].nonant_vardata_list (ROOT).

        Uses iterk_solver_options because the average-scenario solve plays
        the role of a production-tolerance solve, not a first-iteration one.
        """
        solver_name = self.opt.options["solver_name"]
        solver_options = self.opt.options.get("iterk_solver_options") or {}
        solver = pyo.SolverFactory(solver_name)
        for k, v in solver_options.items():
            solver.options[k] = v
        if sputils.is_persistent(solver):
            solver.set_instance(avg_scenario)
            results = solver.solve(tee=False)
        else:
            results = solver.solve(avg_scenario, tee=False)
        tc = results.solver.termination_condition
        if tc not in (TerminationCondition.optimal,
                      TerminationCondition.feasible):
            raise RuntimeError(
                f"Jensen's average-scenario solve failed with termination_condition={tc}"
            )
        sense = sputils.find_active_objective(avg_scenario).sense
        if sense == pyo.minimize:
            outer_bound = results.problem.lower_bound
        else:
            outer_bound = results.problem.upper_bound
        if outer_bound is None or outer_bound != outer_bound:
            raise RuntimeError(
                "Jensen's average-scenario solve did not report a finite dual bound "
                f"(results.problem.{'lower' if sense == pyo.minimize else 'upper'}"
                "_bound). Cannot send a valid outer bound."
            )
        root = avg_scenario._mpisppy_node_list[0]
        nonant_values = [pyo.value(v) for v in root.nonant_vardata_list]
        return outer_bound, nonant_values

    def _jensens_pack_nonant_cache(self, nonant_values):
        """Pack average-scenario nonant values into the dict-by-nodename
        cache format consumed by Xhat_Eval._fix_nonants / .evaluate.
        Two-stage: ROOT only."""
        return {"ROOT": np.array(nonant_values, dtype='d')}
