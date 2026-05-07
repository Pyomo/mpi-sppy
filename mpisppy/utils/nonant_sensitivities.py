###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import numpy as np

import pyomo.environ as pyo
from pyomo.contrib.pynumero.linalg.scipy_interface import ScipyLU

from mpisppy.utils.kkt.interface import InteriorPointInterface


def _bundle_consensus_groups(bundle):
    """Map each bundle nonant position to the per-sub-scenario nonant Vars
    coupled to it by NA equality constraints.

    For a proper bundle (a Pyomo EF model with ``_ef_scenario_names``),
    ``sputils.create_EF`` picks the first sub-scenario's nonant Var as the
    bundle's ref Var and ties the rest by ``x_sub_j == ref`` constraints.
    The bundle objective references each sub-scenario's *own* nonant Vars
    rather than the ref. Probing the KKT sensitivity at the ref Var alone
    therefore captures only the representative sub-scenario; correct
    behavior is to perturb the consensus direction — every per-sub-scenario
    nonant Var at that bundle position together. See issue #673.

    Returns ``{(ndn, k): [list of per-sub-scenario Var objects]}``. The
    construction mirrors the bundle branch of ``sputils.nonant_cost_coeffs``
    and the gradient aggregation in ``mpisppy.extensions.grad_rho``.
    """
    per_scen_to_bundle = {}
    counters = {}
    for (ndn, per_scen_i) in bundle.ref_vars.keys():
        if (ndn, per_scen_i) in bundle.ref_surrogate_vars:
            continue
        counters.setdefault(ndn, 0)
        per_scen_to_bundle[(ndn, per_scen_i)] = (ndn, counters[ndn])
        counters[ndn] += 1

    groups = {}
    for scenario_name in bundle._ef_scenario_names:
        scenario = bundle.component(scenario_name)
        for node in scenario._mpisppy_node_list:
            ndn = node.name
            for per_scen_i, v in enumerate(node.nonant_vardata_list):
                bundle_key = per_scen_to_bundle.get((ndn, per_scen_i))
                if bundle_key is not None:
                    groups.setdefault(bundle_key, []).append(v)
    return groups


def nonant_sensitivies(s):
    """ Compute the sensitivities of noants (w.r.t. the Lagrangian for s)
        Args:
            s: (Pyomo ConcreteModel): the scenario
        Returns:
            nonant_sensis (dict): [ndn_i]: sensitivity for the Var
    """

    # first, solve the subproblems with Ipopt,
    # and gather sensitivity information
    ipopt = pyo.SolverFactory("ipopt")
    solution_cache = pyo.ComponentMap()
    for var in s.component_data_objects(pyo.Var):
        solution_cache[var] = var._value
    relax_int = pyo.TransformationFactory('core.relax_integer_vars')
    relax_int.apply_to(s)

    assert hasattr(s, "_relaxed_integer_vars")

    # add the needed suffixes / remove later
    s.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    s.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    s.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    results = ipopt.solve(s)
    pyo.assert_optimal_termination(results)

    kkt_builder = InteriorPointInterface(s)
    kkt_builder.set_barrier_parameter(1e-9)
    kkt_builder.set_bounds_relaxation_factor(1e-8)
    #rhs = kkt_builder.evaluate_primal_dual_kkt_rhs()
    #print(f"{rhs}")
    #print(f"{rhs.flatten()}")
    kkt = kkt_builder.evaluate_primal_dual_kkt_matrix()

    # print(f"{kkt=}")
    # could do better than SuperLU
    kkt_lu = ScipyLU()
    # always regularize equality constraints
    kkt_builder.regularize_equality_gradient(kkt=kkt, coef=-1e-8, copy_kkt=False)
    # always regularize the Hessian too
    kkt_builder.regularize_hessian(kkt=kkt, coef=1e-8, copy_kkt=False)
    kkt_lu.do_numeric_factorization(kkt, raise_on_error=True)

    grad_vec = np.zeros(kkt.shape[1])
    grad_vec[0:kkt_builder._nlp.n_primals()] = kkt_builder._nlp.evaluate_grad_objective()

    grad_vec_kkt_inv = kkt_lu._lu.solve(grad_vec, "T")

    consensus_groups = (
        _bundle_consensus_groups(s)
        if hasattr(s, "_ef_scenario_names")
        else None
    )

    nonant_sensis = dict()
    for ndn_i, v in s._mpisppy_data.nonant_indices.items():
        if v.fixed:
            # Modeler fixed  -- reporting 0.
            # +infy probably makes more conceptual sense, but 0 seems safer.
            nonant_sensis[ndn_i] = 0.0
            continue

        y_vec = np.zeros(kkt.shape[0])
        if consensus_groups is None:
            y_vec[kkt_builder._nlp._vardata_to_idx[v]] = 1.0
        else:
            # Bundle: probe the consensus direction so the perturbation
            # matches the NA equality constraints that couple per-sub-scenario
            # nonants at this bundle position. Without this, the formula
            # would only see the representative ref Var (issue #673).
            wrote_any = False
            for sub_v in consensus_groups.get(ndn_i, ()):
                if sub_v.fixed:
                    continue
                y_vec[kkt_builder._nlp._vardata_to_idx[sub_v]] = 1.0
                wrote_any = True
            if not wrote_any:
                nonant_sensis[ndn_i] = 0.0
                continue

        x_denom = y_vec.T @ kkt_lu._lu.solve(y_vec)
        x = (-1 / x_denom)
        e_x = x * y_vec

        sensitivity = grad_vec_kkt_inv @ -e_x
        nonant_sensis[ndn_i] = sensitivity

    relax_int.apply_to(s, options={"undo":True})
    assert not hasattr(s, "_relaxed_integer_vars")
    del s.ipopt_zL_out
    del s.ipopt_zU_out
    del s.dual
    for var, val in solution_cache.items():
        var._value = val

    return nonant_sensis
