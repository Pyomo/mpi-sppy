# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import os
import mpisppy.examples.sizes.models.ReferenceModel as ref
import mpisppy.examples.sizes.sizes
import mpisppy.utils.sputils as sputils

def scenario_creator(scenario_name, node_names=None, cb_data=None):
    """ The callback needs to create an instance and then attach
        the PySP nodes to it in a list _PySPnode_list ordered by stages.
        Optionally attach _PHrho. 
        Use cb_data for the scenario count (3 or 10)
    """
    if cb_data not in [3, 10]:
        raise RuntimeError(
            "cb_data passed to scenario counter " "must equal either 3 or 10"
        )

    sizes_dir = os.path.dirname(mpisppy.examples.sizes.sizes.__file__)
    datadir = os.sep.join((sizes_dir, f"SIZES{cb_data}"))
    try:
        fname = datadir + os.sep + scenario_name + ".dat"
    except:
        print("FAIL: datadir=", datadir, " scenario_name=", scenario_name)

    model = ref.model.create_instance(fname)

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
