###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Chance constraints (PySP-style sample-average approximation), applied as a
# transform on an already-built extensive form (EF).
#
# The user defines, in each scenario model, a binary indicator variable z_s with
# the convention
#
#     z_s = 1   <=>   the risky constraint is SATISFIED in scenario s
#
# plus their own big-M constraints linking z_s to satisfaction (exactly as in
# PySP -- this module creates no indicator and no linking constraint).  Given
# those indicators, the SAA chance constraint at confidence level 1 - alpha is
# the single aggregating inequality
#
#     Sum_s p_s * z_s  >=  1 - alpha
#
# i.e. E[z] >= 1 - alpha, i.e. the violation probability is at most alpha.
#
# Unlike CVaR (mpisppy/utils/cvar.py), this aggregator couples a variable from
# every scenario, so it does NOT separate and is supported only for the EF.
# See doc/designs/chance_constraint_design.md.

import warnings

import pyomo.environ as pyo
from mpisppy import MPI


def add_chance_constraint(ef_model, *, cc_indicator_var_name, cc_alpha):
    """Add a PySP-style SAA chance constraint to an already-built EF model.

    For a scalar indicator variable, adds to ``ef_model`` the single constraint

        Sum_s p_s * z_s  >=  (1 - cc_alpha) * Sum_s p_s

    where ``z_s = getattr(scenario_s, cc_indicator_var_name)``.  The ``Sum_s p_s``
    factor on the right is the total probability of the scenarios in this model
    (1.0 for a full EF); carrying it makes the constraint correct even for
    normalized / bundled EFs and reduces to PySP's plain ``>= 1 - alpha`` when
    the probabilities sum to one.

    For an indexed indicator variable, adds one such constraint per index of the
    variable (one chance constraint per index, joint over scenarios).

    The user is responsible for defining ``z_s`` (binary) and the big-M
    constraints linking it to satisfaction, exactly as in PySP.  This function
    adds only the aggregator.

    Args:
        ef_model (Pyomo ConcreteModel): an assembled extensive form, i.e.
            ``ExtensiveForm.ef`` or the result of ``sputils.create_EF``.  It must
            carry ``_ef_scenario_names`` and expose each scenario as a sub-block
            with a ``_mpisppy_probability`` attribute.
        cc_indicator_var_name (str): the name of the per-scenario indicator
            variable (scalar or indexed).
        cc_alpha (float): the allowed violation probability, 0 <= alpha < 1.
            alpha = 0 forces satisfaction in every scenario (a robust constraint).

    Returns:
        pyo.Constraint: the constraint component added to ``ef_model`` (scalar or
        indexed), also reachable as ``ef_model._mpisppy_chance_constraint``.
    """
    if not (0.0 <= cc_alpha < 1.0):
        raise ValueError(f"cc_alpha must satisfy 0 <= alpha < 1 (got {cc_alpha})")

    snames = getattr(ef_model, "_ef_scenario_names", None)
    if not snames:
        raise ValueError(
            "add_chance_constraint needs an assembled EF model carrying "
            "_ef_scenario_names (e.g. ExtensiveForm.ef or sputils.create_EF(...))")

    total_prob = sum(getattr(ef_model, sn)._mpisppy_probability for sn in snames)
    rhs = (1.0 - cc_alpha) * total_prob

    rep = getattr(ef_model, snames[0]).find_component(cc_indicator_var_name)
    if rep is None:
        raise ValueError(
            f"chance-constraint indicator '{cc_indicator_var_name}' not found on "
            f"scenario '{snames[0]}'")

    def _aggregate(index):
        # Sum_s p_s * z_s[index]  (index is None for a scalar indicator)
        expr = 0
        for sn in snames:
            z = getattr(ef_model, sn).find_component(cc_indicator_var_name)
            if z is None:
                raise ValueError(
                    f"chance-constraint indicator '{cc_indicator_var_name}' not "
                    f"found on scenario '{sn}'")
            zs = z if index is None else z[index]
            expr += getattr(ef_model, sn)._mpisppy_probability * zs
        return expr >= rhs

    if rep.is_indexed():
        def _rule(_m, *idx):
            return _aggregate(idx[0] if len(idx) == 1 else idx)
        con = pyo.Constraint(rep.index_set(), rule=_rule)
    else:
        con = pyo.Constraint(expr=_aggregate(None))

    ef_model.add_component("_mpisppy_chance_constraint", con)
    _warn_if_not_binary(rep, cc_indicator_var_name)
    return con


def _warn_if_not_binary(indicator, name):
    """Warn (on rank 0) if the indicator is not binary.

    Exactness of the chance constraint requires binary indicators; a continuous
    "indicator" yields a CVaR-like relaxation rather than a true chance
    constraint.  We warn but proceed, matching PySP's permissiveness.  Gated on
    rank 0 so an HPC run does not print one copy per process.
    """
    if MPI.COMM_WORLD.Get_rank() != 0:
        return
    vardata = next(iter(indicator.values())) if indicator.is_indexed() else indicator
    if not vardata.is_binary():
        warnings.warn(
            f"chance-constraint indicator '{name}' is not binary; the result is "
            "a relaxation, not a true chance constraint")
