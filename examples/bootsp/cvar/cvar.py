###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# A CVaR example (as in the Lam & Qian paper) for the bootstrap
# confidence-interval code. The scenario data are draws from a standard normal
# (empirical path) or from a statdist distribution fitted to the sample data
# (smoothed path), so importing this example needs the statdist library.

import pyomo.environ as pyo
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import numpy as np
# importing Sampler pulls in the statdist package; the smoothed path fits a
# distribution (upstream) and samples it here, so cvar itself needs only Sampler
from mpisppy.confidence_intervals.bootsp.statdist.sampler import Sampler

# Use this random stream:
sstream = np.random.RandomState(1)


def make_model(xi, num_scens, alpha=0.1):

    # Create the concrete model object
    model = pyo.ConcreteModel("Lam_CVaR")

    model.nu = pyo.Var(within=pyo.NonNegativeReals)  # second stage (xi - x)+ in L&Q
    model.eta = pyo.Var(within=pyo.Reals)  # first stage (x in Lam and Qian)

    model.Obj1 = pyo.Expression(expr=model.eta + (model.nu/alpha))

    model.obj = pyo.Objective(expr=model.Obj1)

    def excess_rule(m):
        return m.nu >= xi - m.eta
    model.excess_constraint = pyo.Constraint(rule=excess_rule)

    # Create the list of nodes associated with the scenario (for two stage,
    # there is only one node associated with the scenario--leaf nodes are
    # ignored).
    model._mpisppy_node_list = [
        scenario_tree.ScenarioNode(
            name="ROOT",
            cond_prob=1.0,
            stage=1,
            cost_expression=model.Obj1,
            nonant_list=[model.eta],
            scen_model=model,
        )
    ]

    # Add the probability of the scenario
    if num_scens is not None:
        model._mpisppy_probability = 1/num_scens
    else:
        model._mpisppy_probability = "uniform"
    return model


def data_sampler(record_num, cfg):
    # return a single point from a sample
    # Note: we are syncronizing using the seed
    sstream.seed(record_num + cfg.seed_offset)
    xi = sstream.normal(0, 1)
    return xi


def scenario_creator(scenario_name, cfg):
    """ Create a CVaR scenario.

    Args:
        scenario_name (str):
            Name of the scenario to construct.
        cfg (Config): the control parameters
    """
    # scenario_name has the form <str><int> e.g. scen12, foobar7
    # The digits are scraped off the right of scenario_name using regex.
    scennum = sputils.extract_num(scenario_name)
    sstream.seed(scennum + cfg.seed_offset)  # allows for resampling easily

    if getattr(cfg, "use_fitted", False):
        # sampler works with a list
        sampler = Sampler([cfg.fitted_distribution], sstream)
        xi = sampler.sample_one()[0]
    else:
        xi = sstream.normal(0, 1)
    num_scens = cfg.get('num_scens', None)
    return make_model(xi, num_scens, alpha=0.1)


#=========
def scenario_names_creator(num_scens, start=None):
    # (only for Amalgamator): return the full list of num_scens scenario names
    # if start!=None, the list starts with the 'start' labeled scenario
    if (start is None):
        start = 0
    return [f"scen{i}" for i in range(start, start+num_scens)]


#=========
def inparser_adder(cfg):
    # add options unique to the model
    pass


#=========
def kw_creator(cfg):
    # linked to the scenario_creator and inparser_adder
    kwargs = {"cfg": cfg}
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
    sca["seed_offset"] = seed
    sca["num_scens"] = sample_branching_factors[0]  # two-stage problem
    return scenario_creator(sname, **sca)


#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass


#============================
def xhat_generator(scenario_names, solver_name=None, solver_options=None, cfg=None):
    """ Solve the extensive form over the given scenarios and return xhat.

    This is the fixed-name generator the bootstrap code calls when no xhat file
    is supplied (see boot_utils.compute_xhat). It builds the EF directly from
    this module's scenario_creator so the example is self-contained.

    Args:
        scenario_names (list of str): scenarios to build the EF from
        solver_name (str): solver to use
        solver_options (dict, optional): options passed to the solver
        cfg (Config): control parameters
    Returns:
        xhat (dict): the first-stage nonants keyed by tree node (e.g. ROOT)
    """
    ef = sputils.create_EF(
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs={"cfg": cfg},
    )
    solver = pyo.SolverFactory(solver_name)
    if solver_options is not None:
        for k, v in solver_options.items():
            solver.options[k] = v
    if 'persistent' in solver_name:
        solver.set_instance(ef, symbolic_solver_labels=True)
        solver.solve(tee=False)
    else:
        solver.solve(ef, tee=False, symbolic_solver_labels=True)
    return sputils.nonant_cache_from_ef(ef)
