# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' Implementation of a simple network design problem. See README for more info

    Created: 9 November 2019 by DTM

    Scenario indices are ZERO based
'''

import mpisppy.utils.sputils as sputils
import pyomo.environ as pyo
import numpy as np

from mpisppy.examples.netdes.parse import parse


def scenario_creator(scenario_name, node_names=None, cb_data=None):
    if (cb_data is None):
        raise RuntimeError('Must provide the name of the .dat file '
                           'containing the instance data via the '
                           'cb_data argument to scenario_creator')
    
    scenario_ix = _get_scenario_ix(scenario_name)
    model = build_scenario_model(cb_data, scenario_ix)

    # now attach the one and only scenario tree node
    sputils.attach_root_node(model, model.FirstStageCost, [model.x])
    
    return model


def build_scenario_model(fname, scenario_ix):
    data = parse(fname, scenario_ix=scenario_ix)
    num_nodes = data['N']
    adj = data['A']    # Adjacency matrix
    edges = data['el'] # Edge list
    c = data['c']      # First-stage  cost matrix (per edge)
    d = data['d']      # Second-stage cost matrix (per edge)
    u = data['u']      # Capacity of each arc
    b = data['b']      # Demand of each node
    p = data['p']      # Probability of scenario

    model = pyo.ConcreteModel()
    model.x = pyo.Var(edges, domain=pyo.Binary)           # First stage vars
    model.y = pyo.Var(edges, domain=pyo.NonNegativeReals) # Second stage vars

    model.edges = edges
    model.PySP_prob = p

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

if __name__=='__main__':
    print('netdes.py has no main')
