###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# A data-file version of the little Schultz bootstrap example.
#
# This is the same two-stage model as examples/bootsp/schultz, but instead of
# computing the right-hand-side data arithmetically from the scenario number,
# each scenario reads one observation (a row) from a dataset file
# (schultz_data.csv). This is the "data-based" workflow the bootstrap methods
# are designed for: the confidence-interval code draws scenario indices from
# the dataset, so max_count is the number of rows in the file, xhat is found
# from candidate_sample_size of them, and the bootstrap resamples the rest.

import os
import numpy as np
import pyomo.environ as pyo
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils

# Use this random stream:
sstream = np.random.RandomState()

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FILE = "schultz_data.csv"
_DATA_CACHE = {}


def _resolve_data_file(data_file):
    # accept an absolute path, a path relative to the current directory, or a
    # bare name that lives next to this module
    if os.path.isabs(data_file):
        return data_file
    if os.path.exists(data_file):
        return os.path.abspath(data_file)
    return os.path.join(_MODULE_DIR, data_file)


def load_data(data_file=DEFAULT_DATA_FILE):
    """ Load (and cache) the dataset as an (n, 2) array of (xi1, xi2) rows. """
    path = _resolve_data_file(data_file)
    if path not in _DATA_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"schultz_data could not find the dataset file: {path}\n"
                "Generate it with schultz_data_generator.py or pass --data-file.")
        _DATA_CACHE[path] = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return _DATA_CACHE[path]


def scenario_creator(scenario_name, num_scens=None, seedoffset=0,
                     data_file=DEFAULT_DATA_FILE):
    """ Create the little Schultz example for one observation in the dataset.

    Args:
        scenario_name (str):
            Name of the scenario; its trailing digits index a dataset row.
        num_scens (int, optional):
            Number of scenarios (for _mpisppy_probability); None -> uniform.
        seedoffset (int): accepted for interface compatibility (unused here).
        data_file (str): the dataset file (default schultz_data.csv).
    """
    data = load_data(data_file)
    scennum = sputils.extract_num(scenario_name)
    row = scennum % len(data)
    xi = [data[row][0], data[row][1]]

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

    # For two stage, there is only one node associated with the scenario.
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
    cfg.add_to_config("data_file",
                      description="csv dataset of (xi1, xi2) observations "
                                  f"(default {DEFAULT_DATA_FILE})",
                      domain=str,
                      default=DEFAULT_DATA_FILE)


#=========
def kw_creator(cfg):
    # linked to the scenario_creator and inparser_adder
    kwargs = {"num_scens": cfg.get('num_scens', None),
              "data_file": cfg.get('data_file', DEFAULT_DATA_FILE)}
    return kwargs


#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass


#============================
def xhat_generator(scenario_names, solver_name=None, solver_options=None,
                   data_file=DEFAULT_DATA_FILE):
    """ Solve the extensive form over the given scenarios and return xhat.

    Args:
        scenario_names (list of str): scenarios (dataset rows) to build the EF
        solver_name (str): solver to use
        solver_options (dict, optional): options passed to the solver
        data_file (str): the dataset file
    Returns:
        xhat (dict): the first-stage nonants keyed by tree node (e.g. ROOT)
    """
    num_scens = len(scenario_names)
    ef = sputils.create_EF(
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs={"num_scens": num_scens, "data_file": data_file},
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
    # ad hoc developer check
    m = scenario_creator("scen0")
    opt = pyo.SolverFactory('cplex')
    results = opt.solve(m)
    pyo.assert_optimal_termination(results)
    m.pprint()
