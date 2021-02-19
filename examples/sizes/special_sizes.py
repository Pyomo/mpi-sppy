# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# special sizes to support exotic options
import os
import models.ReferenceModel as ref
import mpisppy.utils.sputils as sputils

def scenario_creator(scenario_name, scenario_count=None):
    if scenario_count not in (3, 10):
        raise ValueError(
            "scenario_count passed to scenario counter must equal either 3 or 10"
        )

    sizes_dir = os.path.dirname(__file__)
    datadir = os.sep.join((sizes_dir, f"SIZES{scenario_count}"))
    try:
        fname = datadir + os.sep + scenario_name + ".dat"
    except:
        print("FAIL: datadir=", datadir, " scenario_name=", scenario_name)

    model = ref.model.create_instance(fname)
    assert not hasattr(model, "_PySP_scenario_name")
    model._PySP_scenario_name = scenario_name  # in case it is needed
    assert not hasattr(model, "_PySP_scenario_count")
    model._PySP_scenario_count = scenario_count  # in case it is needed

    # now attach the one and only tree node
    varlist = [model.NumProducedFirstStage, model.NumUnitsCutFirstStage]
    sputils.attach_root_node(model, model.FirstStageCost, varlist)

    return model


def scenario_denouement(rank, scenario_name, scenario):
    pass


def _rho_setter(scen):
    """ rho values for the scenario.
    Args:
        scen (pyo.ConcreteModel): the scenario
    Returns:
        a list of (id(vardata), rho)
    """
    retlist = []
    RF = 0.001  # a factor for rho, if you like
    cutrho = scen.UnitReductionCost * RF

    for i in scen.ProductSizes:
        idv = id(scen.NumProducedFirstStage[i])
        rho = scen.UnitProductionCosts[i] * RF
        retlist.append((idv, rho))

        for j in scen.ProductSizes:
            if j <= i:
                idv = id(scen.NumUnitsCutFirstStage[i, j])
                retlist.append((idv, cutrho))

    return retlist

def _variable_probability(scen):
    """ per variable probabilities
    Args:
        scen (pyo.ConcreteModel): the scenario
    Returns:
        a list of (id(vardata), probability)
    NOTE: this is a silly function just for testing
    """
    retlist = []

    # 10 has to be produced
    idv = id(scen.NumProducedFirstStage[10])
    scencnt = scen._PySP_scenario_count
    if scen._PySP_scenario_name == "Scenario1":
        retlist.append((idv, 0))
    else:
        retlist.append((idv, 1/(scencnt-1)))
        
    return retlist


def id_fix_list_fct(s):
    """ specify tuples used by the fixer.

    Args:
        s (ConcreteModel): the sizes instance.
    Returns:
         i0, ik (tuples): one for iter 0 and other for general iterations.
             Var id,  threshold, nb, lb, ub
             The threshold is on the square root of the xbar squared differnce
             nb, lb an bu an "no bound", "upper" and "lower" and give the numver
                 of iterations or None for ik and for i0 anything other than None
                 or None. In both cases, None indicates don't fix.
    """
    import mpisppy.extensions.fixer as fixer

    iter0tuples = []
    iterktuples = []
    for i in s.ProductSizes:
        iter0tuples.append(
            fixer.Fixer_tuple(s.NumProducedFirstStage[i], th=0.01, nb=None, lb=0, ub=0)
        )
        iterktuples.append(
            fixer.Fixer_tuple(s.NumProducedFirstStage[i], th=0.2, nb=3, lb=1, ub=2)
        )
        for j in s.ProductSizes:
            if j <= i:
                iter0tuples.append(
                    fixer.Fixer_tuple(
                        s.NumUnitsCutFirstStage[i, j], th=0.5, nb=None, lb=0, ub=0
                    )
                )
                iterktuples.append(
                    fixer.Fixer_tuple(
                        s.NumUnitsCutFirstStage[i, j], th=0.2, nb=3, lb=1, ub=2
                    )
                )

    return iter0tuples, iterktuples
