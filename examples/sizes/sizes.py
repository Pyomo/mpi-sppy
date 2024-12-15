###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import os
import models.ReferenceModel as ref
import mpisppy.utils.sputils as sputils

def scenario_creator(scenario_name, scenario_count=None):
    if scenario_count not in (3, 10):
        raise RuntimeError(
            "scenario_count passed to scenario counter must equal either 3 or 10"
        )

    sizes_dir = os.path.dirname(__file__)
    datadir = os.sep.join((sizes_dir, f"SIZES{scenario_count}"))
    try:
        fname = datadir + os.sep + scenario_name + ".dat"
    except Exception:
        print("FAIL: datadir=", datadir, " scenario_name=", scenario_name)

    model = ref.model.create_instance(fname)

    # now attach the one and only tree node
    varlist = [model.NumProducedFirstStage, model.NumUnitsCutFirstStage]
    sputils.attach_root_node(model, model.FirstStageCost, varlist)

    return model


def scenario_denouement(rank, scenario_name, scenario):
    pass

########## helper functions ########

#=========
def scenario_names_creator(num_scens,start=None):
    # if start!=None, the list starts with the 'start' labeled scenario
    # note that the scenarios for the sizes problem are one-based
    if (start is None) :
        start=1
    return [f"Scenario{i}" for i in range(start, start+num_scens)]


#=========
def inparser_adder(cfg):
    # add options unique to sizes
    cfg.num_scens_required()
    cfg.mip_options()


#=========
def kw_creator(cfg):
    # (for Amalgamator): linked to the scenario_creator and inparser_adder
    if cfg.num_scens not in (3, 10):
        raise RuntimeError(f"num_scen must the 3 or 10; was {cfg.num_scen}")    
    kwargs = {"scenario_count": cfg.num_scens}
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

######## end helper functions #########

########## a customized rho setter #############
# If you are using sizes.py as a starting point for your model,
#  you should be aware that you don't need a _rho_setter function.
# This demonstrates how to use a customized rho setter; consider instead
#  a gradient based rho setter.
# note that _rho_setter is a reserved name....

def _rho_setter(scen, **kwargs):
    """ rho values for the scenario.
    Args:
        scen (pyo.ConcreteModel): the scenario
    Returns:
        a list of (id(vardata), rho)
    Note:
        This rho_setter will not work with proper bundles.
    """
    retlist = []
    if not hasattr(scen, "UnitReductionCost"):
        print("WARNING: _rho_setter not used (probably because of proper bundles)")
        return retlist        
    RF = 0.001  # a factor for rho, if you like

    if "RF" in kwargs and isinstance(kwargs["RF"], float):
        RF = kwargs["RF"]
        
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
