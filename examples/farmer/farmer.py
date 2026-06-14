###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# special for ph debugging DLW Dec 2018
# unlimited crops
# ALL INDEXES ARE ZERO-BASED
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2018 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# special scalable farmer for stress-testing

import pyomo.environ as pyo
import numpy as np
import mpisppy.utils.sputils as sputils


# ===========================================================================
# Underscore helpers (best-practice pattern — see doc/src/jensens.rst):
#
#   _scenario_data : pure-Python random data as a dict, no Pyomo
#   _build_model   : build the Pyomo model from a data dict
#
# scenario_creator and average_scenario_creator both go through these
# helpers so the model-build code lives in exactly one place. The
# random data is generated with a LOCAL np.random.RandomState per call
# (not a module-level global), which is thread-safe and does not rely
# on any caller re-seeding a shared stream.
# ===========================================================================

_BASE_YIELD = {
    'BelowAverageScenario': {'WHEAT': 2.0, 'CORN': 2.4, 'SUGAR_BEETS': 16.0},
    'AverageScenario':      {'WHEAT': 2.5, 'CORN': 3.0, 'SUGAR_BEETS': 20.0},
    'AboveAverageScenario': {'WHEAT': 3.0, 'CORN': 3.6, 'SUGAR_BEETS': 24.0},
}

_BASENAMES = ['BelowAverageScenario', 'AverageScenario', 'AboveAverageScenario']


def _scenario_data(scenario_name, crops_multiplier=1, seedoffset=0):
    """Return the random data for one scenario as a plain Python dict.

    Deterministic given (scenario_name, seedoffset). Uses a local
    np.random.RandomState so the function is thread-safe and can be
    called from multiple threads in average_scenario_creator.

    Byte-for-byte identical to the previous pattern that seeded a
    module-level RandomState with (scennum + seedoffset) and then
    consumed .rand() draws in the same order.
    """
    scennum = sputils.extract_num(scenario_name)
    basenum = scennum % 3
    groupnum = scennum // 3
    basename = _BASENAMES[basenum]
    rng = np.random.RandomState(scennum + seedoffset)

    yields = {}
    for i in range(crops_multiplier):
        for crop in ['WHEAT', 'CORN', 'SUGAR_BEETS']:
            jitter = rng.rand() if groupnum != 0 else 0.0
            yields[crop + str(i)] = _BASE_YIELD[basename][crop] + jitter
    return {"Yield": yields}


def _build_model(scenario_name, data, *, use_integer=False, sense=pyo.minimize,
                 crops_multiplier=1, probability):
    """Build the Pyomo model for the (scalable) farmer example from a
    data dict. Shared by scenario_creator and average_scenario_creator."""
    if sense not in (pyo.minimize, pyo.maximize):
        raise ValueError("Model sense not recognized")

    model = pyo.ConcreteModel(scenario_name)

    def crops_init(m):
        retval = []
        for i in range(crops_multiplier):
            retval.append("WHEAT" + str(i))
            retval.append("CORN" + str(i))
            retval.append("SUGAR_BEETS" + str(i))
        return retval

    model.CROPS = pyo.Set(initialize=crops_init)

    model.TOTAL_ACREAGE = 500.0 * crops_multiplier

    def _scale_up_data(indict):
        outdict = {}
        for i in range(crops_multiplier):
            for crop in ['WHEAT', 'CORN', 'SUGAR_BEETS']:
                outdict[crop + str(i)] = indict[crop]
        return outdict

    model.PriceQuota = _scale_up_data(
        {'WHEAT': 100000.0, 'CORN': 100000.0, 'SUGAR_BEETS': 6000.0})
    model.SubQuotaSellingPrice = _scale_up_data(
        {'WHEAT': 170.0, 'CORN': 150.0, 'SUGAR_BEETS': 36.0})
    model.SuperQuotaSellingPrice = _scale_up_data(
        {'WHEAT': 0.0, 'CORN': 0.0, 'SUGAR_BEETS': 10.0})
    model.CattleFeedRequirement = _scale_up_data(
        {'WHEAT': 200.0, 'CORN': 240.0, 'SUGAR_BEETS': 0.0})
    model.PurchasePrice = _scale_up_data(
        {'WHEAT': 238.0, 'CORN': 210.0, 'SUGAR_BEETS': 100000.0})
    model.PlantingCostPerAcre = _scale_up_data(
        {'WHEAT': 150.0, 'CORN': 230.0, 'SUGAR_BEETS': 260.0})

    # Stochastic Data
    model.Yield = pyo.Param(model.CROPS,
                            within=pyo.NonNegativeReals,
                            initialize=data["Yield"],
                            mutable=True)

    # Variables
    if use_integer:
        model.DevotedAcreage = pyo.Var(model.CROPS,
                                       within=pyo.NonNegativeIntegers,
                                       bounds=(0.0, model.TOTAL_ACREAGE))
    else:
        model.DevotedAcreage = pyo.Var(model.CROPS,
                                       bounds=(0.0, model.TOTAL_ACREAGE))

    model.QuantitySubQuotaSold = pyo.Var(model.CROPS, bounds=(0.0, None))
    model.QuantitySuperQuotaSold = pyo.Var(model.CROPS, bounds=(0.0, None))
    model.QuantityPurchased = pyo.Var(model.CROPS, bounds=(0.0, None))

    # Constraints
    def ConstrainTotalAcreage_rule(model):
        return pyo.sum_product(model.DevotedAcreage) <= model.TOTAL_ACREAGE
    model.ConstrainTotalAcreage = pyo.Constraint(rule=ConstrainTotalAcreage_rule)

    def EnforceCattleFeedRequirement_rule(model, i):
        return model.CattleFeedRequirement[i] <= (
            model.Yield[i] * model.DevotedAcreage[i]
            + model.QuantityPurchased[i]
            - model.QuantitySubQuotaSold[i]
            - model.QuantitySuperQuotaSold[i])
    model.EnforceCattleFeedRequirement = pyo.Constraint(
        model.CROPS, rule=EnforceCattleFeedRequirement_rule)

    def LimitAmountSold_rule(model, i):
        return (model.QuantitySubQuotaSold[i]
                + model.QuantitySuperQuotaSold[i]
                - (model.Yield[i] * model.DevotedAcreage[i])) <= 0.0
    model.LimitAmountSold = pyo.Constraint(
        model.CROPS, rule=LimitAmountSold_rule)

    def EnforceQuotas_rule(model, i):
        return (0.0, model.QuantitySubQuotaSold[i], model.PriceQuota[i])
    model.EnforceQuotas = pyo.Constraint(model.CROPS, rule=EnforceQuotas_rule)

    # Stage-specific cost computations
    def ComputeFirstStageCost_rule(model):
        return pyo.sum_product(model.PlantingCostPerAcre, model.DevotedAcreage)
    model.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

    def ComputeSecondStageCost_rule(model):
        expr = pyo.sum_product(model.PurchasePrice, model.QuantityPurchased)
        expr -= pyo.sum_product(model.SubQuotaSellingPrice,
                                model.QuantitySubQuotaSold)
        expr -= pyo.sum_product(model.SuperQuotaSellingPrice,
                                model.QuantitySuperQuotaSold)
        return expr
    model.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

    def total_cost_rule(model):
        if sense == pyo.minimize:
            return model.FirstStageCost + model.SecondStageCost
        return -model.FirstStageCost - model.SecondStageCost
    model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule,
                                               sense=sense)

    varlist = [model.DevotedAcreage]
    sputils.attach_root_node(model, model.FirstStageCost, varlist)
    model._mpisppy_probability = probability
    return model


def scenario_creator(
    scenario_name, use_integer=False, sense=pyo.minimize, crops_multiplier=1,
    num_scens=None, seedoffset=0
):
    """ Create a scenario for the (scalable) farmer example.

    Args:
        scenario_name (str):
            Name of the scenario to construct.
        use_integer (bool, optional):
            If True, restricts variables to be integer. Default is False.
        sense (int, optional):
            Model sense (minimization or maximization). Must be either
            pyo.minimize or pyo.maximize. Default is pyo.minimize.
        crops_multiplier (int, optional):
            Factor to control scaling. There will be three times this many
            crops. Default is 1.
        num_scens (int, optional):
            Number of scenarios. We use it to compute _mpisppy_probability.
            Default is None.
        seedoffset (int): used by confidence interval code to create replicates
    """
    data = _scenario_data(scenario_name,
                          crops_multiplier=crops_multiplier,
                          seedoffset=seedoffset)
    probability = (1.0 / num_scens) if num_scens is not None else "uniform"
    return _build_model(scenario_name, data,
                        use_integer=use_integer, sense=sense,
                        crops_multiplier=crops_multiplier,
                        probability=probability)


def average_scenario_creator(
    scenario_name, use_integer=False, sense=pyo.minimize, crops_multiplier=1,
    num_scens=None, seedoffset=0
):
    """Build the average scenario used by --*-try-jensens-first.

    This is a deterministic single-scenario model whose random data is
    the sample mean of the data for every scenario in the run (i.e.
    every name in scenario_names_creator(num_scens)). Its
    _mpisppy_probability is 1.0.

    Every rank constructs an identical model independently. The loop over
    scenarios is serial; for large num_scens it could be multi-threaded,
    e.g.:

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as ex:
            datas = list(ex.map(
                lambda s: _scenario_data(s, crops_multiplier, seedoffset),
                snames))

    This is safe because _scenario_data uses a local RandomState. That
    parallelization is TBD.

    IMPORTANT: Jensen's lower-bound interpretation of this model is only
    valid when the recourse value function is convex in the random
    parameters. See doc/src/jensens.rst for the precondition.
    """
    if num_scens is None:
        raise ValueError("average_scenario_creator requires num_scens")
    snames = scenario_names_creator(num_scens)
    datas = [_scenario_data(s,
                            crops_multiplier=crops_multiplier,
                            seedoffset=seedoffset)
             for s in snames]
    avg_yield = {
        k: sum(d["Yield"][k] for d in datas) / len(datas)
        for k in datas[0]["Yield"]
    }
    return _build_model(scenario_name, {"Yield": avg_yield},
                        use_integer=use_integer, sense=sense,
                        crops_multiplier=crops_multiplier,
                        probability=1.0)


# begin helper functions
# =========
def scenario_names_creator(num_scens, start=None):
    # (only for Amalgamator): return the full list of num_scens scenario names
    # if start!=None, the list starts with the 'start' labeled scenario
    if start is None:
        start = 0
    return [f"scen{i}" for i in range(start, start + num_scens)]


# =========
def inparser_adder(cfg):
    # add options unique to farmer
    cfg.num_scens_required()
    cfg.add_to_config("crops_multiplier",
                      description="number of crops will be three times this (default 1)",
                      domain=int,
                      default=1)

    cfg.add_to_config("farmer_with_integers",
                      description="make the version that has integers (default False)",
                      domain=bool,
                      default=False)


# =========
def kw_creator(cfg):
    # (for Amalgamator): linked to the scenario_creator and inparser_adder
    kwargs = {"use_integer": cfg.get('farmer_with_integers', False),
              "crops_multiplier": cfg.get('crops_multiplier', 1),
              "num_scens": cfg.get('num_scens', None),
              }
    return kwargs


def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    """ Create a scenario within a sample tree. Mainly for multi-stage and simple for two-stage.
        (this function supports zhat and confidence interval code)
    Args:
        sname (string): scenario name to be created
        stage (int >=1 ): for stages > 1, fix data based on sname in earlier stages
        sample_branching_factors (list of ints): branching factors for the sample tree
        seed (int): To allow random sampling (for some problems, it might be scenario offset)
        given_scenario (Pyomo concrete model): if not None, use this to get data for ealier stages
        scenario_creator_kwargs (dict): keyword args for the standard scenario creator funcion
    Returns:
        scenario (Pyomo concrete model): A scenario for sname with data in stages < stage determined
                                         by the arguments
    """
    # Since this is a two-stage problem, we don't have to do much.
    sca = scenario_creator_kwargs.copy()
    sca["seedoffset"] = seed
    sca["num_scens"] = sample_branching_factors[0]  # two-stage problem
    return scenario_creator(sname, **sca)


# ============================
def scenario_denouement(rank, scenario_name, scenario):
    sname = scenario_name
    s = scenario
    if sname == 'scen0':
        print("Arbitrary sanity checks:")
        print("SUGAR_BEETS0 for scenario", sname, "is",
              pyo.value(s.DevotedAcreage["SUGAR_BEETS0"]))
        print("FirstStageCost for scenario", sname, "is", pyo.value(s.FirstStageCost))
