###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# A small two-stage Schultz example for the bootstrap confidence-interval code.
# The scenario data is fully deterministic (a function of the scenario number),
# so the extensive-form optimum and the bootstrap draws are reproducible and
# solver-independent. This "unique" variant has a unique optimal solution.

import pyomo.environ as pyo
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import numpy as np

# Use this random stream:
sstream = np.random.RandomState()


def scenario_creator(scenario_name, num_scens=None, seedoffset=0):
    """ Create the little Schultz example

    Args:
        scenario_name (str):
            Name of the scenario to construct.
        num_scens (int, optional):
            Number of scenarios. We use it to compute _mpisppy_probability.
            Default is None (which yields a uniform probability).
        seedoffset (int): used by confidence interval code
    """
    # scenario_name has the form <str><int> e.g. scen12, foobar7
    # The digits are scraped off the right of scenario_name using regex; the
    # (deterministic) scenario data is a function of that number modulo 121.
    scennum = sputils.extract_num(scenario_name)
    scennum = scennum % 121

    sstream.seed(scennum + seedoffset)  # allows for resampling easily
    ri1 = scennum // 11 + 5
    ri2 = scennum % 11 + 5
    xi = [ri1, ri2]

    fsc = (-1.5, -4)
    ssc = (-16, -19, -23, -28)
    T = [[2, 3, 4, 5], [6, 1, 3, 2]]

    # Create the concrete model object
    model = pyo.ConcreteModel(scenario_name)

    xrange = range(2)
    yrange = range(4)
    model.x = pyo.Var(xrange, within=pyo.NonNegativeIntegers, bounds=(0, 5))
    model.y = pyo.Var(yrange, within=pyo.Binary)

    model.Obj1 = pyo.Expression(expr=sum(model.x[i] * fsc[i] for i in xrange))
    model.Obj2 = pyo.Expression(expr=sum(model.y[i] * ssc[i] for i in yrange))

    model.obj = pyo.Objective(expr=model.Obj1 + model.Obj2, sense=pyo.minimize)

    def upper_rule(m, i):
        return sum(T[i][j] * m.y[j] for j in yrange) <= xi[i] - m.x[i]
    model.constraint = pyo.Constraint(xrange, rule=upper_rule)

    # Create the list of nodes associated with the scenario (for two stage,
    # there is only one node associated with the scenario--leaf nodes are
    # ignored).
    model._mpisppy_node_list = [
        scenario_tree.ScenarioNode(
            name="ROOT",
            cond_prob=1.0,
            stage=1,
            cost_expression=model.Obj1,
            nonant_list=[model.x],
            scen_model=model,
        )
    ]

    # Add the probability of the scenario
    if num_scens is not None:
        model._mpisppy_probability = 1 / num_scens
    else:
        model._mpisppy_probability = "uniform"
    return model


#=========
def scenario_names_creator(num_scens, start=None):
    # (only for Amalgamator): return the full list of num_scens scenario names
    # if start!=None, the list starts with the 'start' labeled scenario
    if (start is None):
        start = 0
    return [f"scen{i}" for i in range(start, start + num_scens)]


#=========
def inparser_adder(cfg):
    # add options unique to this model
    pass


#=========
def kw_creator(cfg):
    # linked to the scenario_creator and inparser_adder
    kwargs = {"num_scens": cfg.get('num_scens', None)}
    return kwargs


#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass


#============================
def xhat_generator(scenario_names, solver_name=None, solver_options=None):
    """ Solve the extensive form over the given scenarios and return xhat.

    This is the fixed-name generator the bootstrap code calls when no
    xhat file is supplied (see boot_utils.compute_xhat).

    Args:
        scenario_names (list of str): scenarios to build the EF from
        solver_name (str): solver to use
        solver_options (dict, optional): options passed to the solver
    Returns:
        xhat (dict): the first-stage nonants keyed by tree node (e.g. ROOT)
    """
    num_scens = len(scenario_names)
    ef = sputils.create_EF(
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs={"num_scens": num_scens},
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


if __name__ == "__main__":
    # This is command line callable just to support ad hoc testing by developers
    m = scenario_creator("scen0")
    opt = pyo.SolverFactory('cplex')
    results = opt.solve(m)
    pyo.assert_optimal_termination(results)
    m.pprint()
