###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# A tiny chance-constrained capacity example for generic_cylinders (EF only).
#
# Build capacity x (first stage, cost per unit). Scenario s has demand d_s and a
# binary indicator ``served`` that is 1 only if the capacity covers the demand
# (big-M link: x >= d_s - M*(1 - served)). The chance constraint added by
# mpi-sppy,
#
#     Sum_s p_s * served_s  >=  1 - alpha,
#
# then requires the demand to be met with probability at least 1 - alpha.
#
# With deterministic ramp demands d_i = 10*(i+1) and uniform probabilities, the
# cost-minimizing capacity is the (1 - alpha)-quantile of demand. For example
# num_scens=10, cc_alpha=0.2 => serve the 8 smallest demands => x* = 80.
#
# Run (extensive form only):
#   python -m mpisppy.generic_cylinders --module-name examples/chance_constraint/cc_capacity \
#       --num-scens 10 --EF --solver-name gurobi --cc-indicator-var served --cc-alpha 0.2

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils

UNIT_CAPACITY_COST = 1.0


def _demand(scenario_index):
    # deterministic ramp so the optimum is easy to verify by hand
    return 10.0 * (scenario_index + 1)


def scenario_creator(sname, num_scens=None, **kwargs):
    assert num_scens is not None, "cc_capacity needs num_scens"
    i = sputils.extract_num(sname)
    d = _demand(i)
    big_m = _demand(num_scens - 1)        # max possible demand

    model = pyo.ConcreteModel(name=sname)
    model.x = pyo.Var(bounds=(0.0, big_m))            # first-stage capacity
    model.served = pyo.Var(domain=pyo.Binary)         # 1 == demand met

    # big-M link: served == 1 forces x >= d
    model.serve_link = pyo.Constraint(expr=model.x >= d - big_m * (1 - model.served))

    model.FirstStageCost = pyo.Expression(expr=UNIT_CAPACITY_COST * model.x)
    model.cost = pyo.Objective(expr=model.FirstStageCost, sense=pyo.minimize)

    model._mpisppy_probability = 1.0 / num_scens
    sputils.attach_root_node(model, model.FirstStageCost, [model.x])
    return model


def scenario_names_creator(num_scens, start=None):
    if start is None:
        start = 0
    return [f"scen{i}" for i in range(start, start + num_scens)]


def inparser_adder(cfg):
    cfg.num_scens_required()


def kw_creator(cfg):
    return {"num_scens": cfg.get("num_scens", None)}


def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    # two-stage: defer to the standard creator
    sca = scenario_creator_kwargs.copy()
    sca["num_scens"] = sample_branching_factors[0]
    return scenario_creator(sname, **sca)


def scenario_denouement(rank, scenario_name, scenario):
    if scenario_name == "scen0":
        print(f"capacity x = {pyo.value(scenario.x):.1f}")
