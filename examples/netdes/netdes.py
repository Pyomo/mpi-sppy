###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
''' Implementation of a simple network design problem. See README for more info

    Created: 9 November 2019 by DTM

    Scenario indices are ZERO based
'''

import os
import mpisppy.utils.sputils as sputils
import pyomo.environ as pyo
import numpy as np

from parse import parse


# ===========================================================================
# Underscore helpers (best-practice pattern -- see doc/src/jensens.rst):
#
#   _scenario_data : pure-Python data dict for one scenario, no Pyomo
#   _build_model   : build the Pyomo model from a data dict
#
# scenario_creator and average_scenario_creator both go through these
# helpers so the model-build code lives in exactly one place.
# ===========================================================================

def _scenario_data(scenario_ix, path):
    """Return the full data dict for one scenario as plain Python.

    Thin wrapper over parse(...) so scenario_creator and
    average_scenario_creator share the same data-loading entry point.
    """
    return parse(path, scenario_ix=scenario_ix)


def _build_model(data, probability):
    """Build the network-design Pyomo model from a data dict.

    Shared by scenario_creator and average_scenario_creator. The dict
    must have keys N, A, el, c, d, u, b (the first four are
    deterministic; d, u, b are the second-stage stochastic data).
    """
    num_nodes = data['N']
    adj = data['A']    # Adjacency matrix
    edges = data['el'] # Edge list
    c = data['c']      # First-stage  cost matrix (per edge)
    d = data['d']      # Second-stage cost matrix (per edge)
    u = data['u']      # Capacity of each arc
    b = data['b']      # Demand of each node

    model = pyo.ConcreteModel()
    model.x = pyo.Var(edges, domain=pyo.Binary)           # First stage vars
    model.y = pyo.Var(edges, domain=pyo.NonNegativeReals) # Second stage vars

    model.edges = edges
    model._mpisppy_probability = probability

    ''' Objective '''
    model.FirstStageCost  = pyo.quicksum(c[e] * model.x[e] for e in edges)
    model.SecondStageCost = pyo.quicksum(d[e] * model.y[e] for e in edges)
    obj_expr = model.FirstStageCost + model.SecondStageCost
    model.MinCost = pyo.Objective(expr=obj_expr, sense=pyo.minimize)

    ''' Variable upper bound constraints on each edge '''
    model.vubs = pyo.ConstraintList()
    for e in edges:
        expr = model.y[e] - u[e] * model.x[e]
        model.vubs.add(expr <= 0)

    ''' Flow balance constraints for each node '''
    model.bals = pyo.ConstraintList()
    for i in range(num_nodes):
        in_nbs  = np.where(adj[:,i] > 0)[0]
        out_nbs = np.where(adj[i,:] > 0)[0]
        lhs = pyo.quicksum(model.y[i,j] for j in out_nbs) - \
              pyo.quicksum(model.y[j,i] for j in in_nbs)
        model.bals.add(lhs == b[i])

    return model


def scenario_creator(scenario_name, path=None):
    if path is None:
        raise RuntimeError('Must provide the name of the .dat file '
                           'containing the instance data via the '
                           'path argument to scenario_creator')

    scenario_ix = _get_scenario_ix(scenario_name)
    data = _scenario_data(scenario_ix, path)
    model = _build_model(data, probability=data['p'])

    # now attach the one and only scenario tree node
    sputils.attach_root_node(model, model.FirstStageCost, [model.x[:,:], ])

    return model


def average_scenario_creator(scenario_name, path=None):
    """Build the average scenario for ``--*-try-jensens-first``.

    Constructs a single deterministic model whose second-stage data
    (d, u, b) is the probability-weighted mean of the per-scenario
    data shipped in the instance file. The first-stage data (c) and
    network topology (N, A, el) are deterministic and copied through.
    ``_mpisppy_probability`` is 1.0.

    Every rank constructs an identical model independently. The
    instance file is parsed once.

    Jensen's-bound validity (see doc/src/jensens.rst):

      * netdes has continuous recourse (``model.y`` is
        ``NonNegativeReals``), so the integer-recourse guard on the
        outer-bound path will pass.
      * Randomness lives in the second-stage objective coefficients
        ``d``, the per-arc capacity ``u`` (which appears in
        ``y[e] - u[e]*x[e] <= 0``, i.e. as an effective RHS once x
        is fixed), and the node demand RHS ``b``. RHS-only randomness
        gives a recourse value that is convex in those parameters,
        so the Jensen's outer bound is valid for randomness in
        ``u`` and ``b``. Cost-coefficient randomness (``d``) makes
        the recourse value concave in ``d``; users who require a
        provably valid Jensen's *outer* bound should be aware of
        this and decide whether the bound is meaningful for their
        instance.
      * The xhat (inner-bound) path is unaffected by the convexity
        question -- it uses the average scenario only as a source of
        a candidate first-stage x, which is then honestly evaluated
        across all real scenarios.
    """
    if path is None:
        raise RuntimeError('Must provide the name of the .dat file '
                           'containing the instance data via the '
                           'path argument to average_scenario_creator')

    full = parse(path, scenario_ix=None)
    K = full['K']
    p = full['p']
    # Probability-weighted mean across all scenarios. Netdes ships
    # non-uniform probabilities in general, so this is NOT a simple
    # arithmetic mean.
    avg_d = sum(p[k] * full['d'][k] for k in range(K))
    avg_u = sum(p[k] * full['u'][k] for k in range(K))
    avg_b = sum(p[k] * full['b'][k] for k in range(K))

    avg_data = {
        'N': full['N'],
        'A': full['A'],
        'el': full['el'],
        'c': full['c'],
        'd': avg_d,
        'u': avg_u,
        'b': avg_b,
    }
    model = _build_model(avg_data, probability=1.0)
    sputils.attach_root_node(model, model.FirstStageCost, [model.x[:,:], ])
    return model




def scenario_denouement(rank, scenario_name, scenario):
    pass

def _get_scenario_ix(sname):
    ''' Get the scenario index from the given scenario name by strpiping all
        digits off of the right of the scenario name, until a non-digit is
        encountered.
    '''
    i = len(sname) - 1
    while (i > 0 and sname[i-1].isdigit()):
        i -= 1
    return int(sname[i:])

########## helper functions ########

#=========
def scenario_names_creator(num_scens,start=None):
    # if start!=None, the list starts with the 'start' labeled scenario
    if (start is None) :
        start=0
    return [f"Scenario{i}" for i in range(start,start+num_scens)]


#=========
def inparser_adder(cfg):
    # add options unique to sizes
    # we don't want num_scens from the command line
    cfg.mip_options()
    cfg.add_to_config("instance_name",
                        description="netdes instance name (e.g., network-10-20-L-01)",
                        domain=str,
                        default=None)                
    cfg.add_to_config("netdes_data_path",
                        description="path to detdes data (e.g., ./data)",
                        domain=str,
                        default=None)                


#=========
def kw_creator(cfg):
    # linked to the scenario_creator and inparser_adder
    # side-effect is dealing with num_scens
    inst = cfg.instance_name
    ns = int(inst.split("-")[-3])
    if hasattr(cfg, "num_scens"):
        if cfg.num_scens != ns:
            raise RuntimeError(f"Argument num-scens={cfg.num_scens} does not match the number "
                               "implied by instance name={ns} "
                               "\n(--num-scens is not needed for netdes)")
    else:
        cfg.add_and_assign("num_scens","number of scenarios", int, None, ns)
    path = os.path.join(cfg.netdes_data_path, f"{inst}.dat")
    kwargs = {"path": path}
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


if __name__=='__main__':
    print('netdes.py has no main')
