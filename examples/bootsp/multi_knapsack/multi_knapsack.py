###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# A multi-product knapsack example (Vaagen & Wallace, IJPE 2007; the model
# version is from chapter 6 of the King/Wallace book) for the bootstrap
# confidence-interval code. Deterministic data come from a json file named by
# --deterministic-data-json; the random demands are drawn from statdist
# univariate-normal distributions (empirical path) or from distributions fitted
# to the sample data (smoothed path), so importing this example needs statdist.

import os
import json
import numpy as np
import pyomo.environ as pyo
import mpisppy.scenario_tree as scenario_tree  # noqa: F401  (kept for parity/attach_root_node users)
import mpisppy.utils.sputils as sputils
import mpisppy.confidence_intervals.bootsp.statdist as statdist
from mpisppy.confidence_intervals.bootsp.statdist.sampler import Sampler

# Use this random stream:
sstream = np.random.RandomState(1)


def _read_detdata(cfg):
    # deterministic data; resolve the file relative to this module if it is not
    # found relative to the current working directory
    json_fname = cfg.deterministic_data_json
    if not os.path.isabs(json_fname) and not os.path.exists(json_fname):
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, json_fname)
        if os.path.exists(candidate):
            json_fname = candidate
    try:
        with open(json_fname, "r") as read_file:
            detdata = json.load(read_file)
    except Exception:
        print(f"Could not read the json file: {json_fname}")
        raise
    return detdata


def _detdata_for(cfg):
    # the smoothed driver stashes the parsed data on cfg.detdata; otherwise read
    # it from the file (this makes the empirical path work without that stash)
    if "detdata" in cfg and cfg.detdata is not None:
        return cfg.detdata
    return _read_detdata(cfg)


def _get_distr_dict(cfg, detdata):
    if not getattr(cfg, "use_fitted", False):
        unorm = statdist.distribution_factory('univariate-normal')
        varset = pyo.RangeSet(detdata["num_prods"])
        distr_dict = {}
        for i in varset:
            distr_dict[i] = {
                "high": unorm(var=(detdata["stdev_d"]["high"])**2, mean=detdata["mean_d"]["high"]),
                "low": unorm(var=(detdata["stdev_d"]["low"])**2, mean=detdata["mean_d"]["low"])
            }
    else:
        distr_dict = cfg.fitted_distribution
    return distr_dict


def data_sampler(record_num, cfg):
    detdata = _detdata_for(cfg)

    distr_dict = _get_distr_dict(cfg, detdata)
    sstream.seed(record_num+cfg.seed_offset)

    # this part of the code is the same as in the scenario creator
    data = {}
    varset = pyo.RangeSet(detdata["num_prods"])
    if getattr(cfg, "use_fitted", False):
        for i in varset:
            sampler = Sampler([distr_dict[i]], sstream)
            data[i] = max(0, int(sampler.sample_one()[0]))
    else:
        for i in varset:
            state = 'high' if sstream.uniform() < 0.5 else 'low'
            sampler = Sampler([distr_dict[i][state]], sstream)
            data[i] = max(0, int(sampler.sample_one()[0]))
    return data


def scenario_creator(scenario_name, cfg=None, seed_offset=None, num_scens=None):
    """ Create a multi-knapsack scenario.

    Args:
        scenario_name (str):
            Name of the scenario to construct.
        cfg (Config): the control parameters
        seed_offset (int): used by confidence interval code
    Returns:
        model (ConcreteModel): the Pyomo model
    """
    # scenario_name has the form <str><int> e.g. scen12, foobar7
    # The digits are scraped off the right of scenario_name using regex.
    scennum = sputils.extract_num(scenario_name)

    seed_offset = cfg.get("seed_offset", 0) if seed_offset is None else seed_offset
    sstream.seed(scennum+seed_offset)  # allows for resampling easily
    num_scens = cfg.get('num_scens', None)

    # Create the concrete model object
    model = pyo.ConcreteModel(f"multi-knapsack {scenario_name}")

    detdata = _detdata_for(cfg)
    v = detdata["v"]
    c = detdata["c"]
    g = detdata["g"]
    alpha = detdata["alpha"]  # a dict of lists

    # use the same variable names as in chapter 6 of the King/Wallace book
    # item numbers start at 1
    model.I = pyo.RangeSet(detdata["num_prods"])

    model.x = pyo.Var(model.I, within=pyo.NonNegativeReals, initialize=0)
    model.y = pyo.Var(model.I, within=pyo.NonNegativeReals, initialize=0)
    model.z = pyo.Var(model.I, model.I, within=pyo.NonNegativeReals, initialize=0)
    model.zt = pyo.Var(model.I, within=pyo.NonNegativeReals, initialize=0)
    model.w = pyo.Var(model.I, within=pyo.NonNegativeReals, initialize=0)

    d = data_sampler(scennum, cfg)

    # note: the json indexes are strings

    def d_rule(m, i):
        return m.y[i] + sum(m.z[j, i] for j in model.I if j != i) <= d[i]
    model.d_constraint = pyo.Constraint(model.I, rule=d_rule)

    def z_rule(m, i, j):
        # note that alpha is a dict of lists
        if i == j:
            return pyo.Constraint.Skip
        else:
            return m.z[i, j] <= alpha[str(i)][j-1] * (d[j]-m.y[j])
    model.z_constraint = pyo.Constraint(model.I, model.I, rule=z_rule)

    def zt_rule(m, i):
        return m.zt[i] == sum(m.z[i, j] for j in model.I if j != i)
    model.zt_constraint = pyo.Constraint(model.I, rule=zt_rule)

    def w_rule(m, i):
        return m.w[i] == m.x[i] - (m.y[i]+m.zt[i])
    model.w_constraint = pyo.Constraint(model.I, rule=w_rule)

    m = model  # typing aid
    model.Obj1 = pyo.Expression(expr=-sum(v[str(i)]*(m.y[i]+m.zt[i])
                                          + g[str(i)]*m.w[i]
                                          - c[str(i)]*m.x[i] for i in m.I))

    model.obj = pyo.Objective(expr=model.Obj1, sense=pyo.minimize)

    # Create the list of nodes associated with the scenario (for two stage,
    # there is only one node associated with the scenario--leaf nodes are
    # ignored).
    varlist = [model.x]
    sputils.attach_root_node(model, model.Obj1, varlist)

    # Add the probability of the scenario
    if num_scens is not None:
        model._mpisppy_probability = 1/num_scens
    else:
        model._mpisppy_probability = "uniform"
    return model


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
    cfg.add_to_config("deterministic_data_json",
                      description="file name for json file with determinstic data",
                      domain=str,
                      default=None)


#=========
def kw_creator(cfg):
    # linked to the scenario_creator and inparser_adder
    kwargs = {"cfg": cfg}
    return kwargs


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
        cfg (Config): control parameters (includes deterministic_data_json)
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
