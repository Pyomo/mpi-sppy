# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' Solution to the Lagrangian relaxation of a hybrid solar-battery storage
    system from the paper:

    B. Singh and B. Knueven. Lagrangian relaxation based heuristics for a
    chance-constrained optimization model of a hybrid solar-battery storage
    system. Submitted, 2019.

    This code solves the Lagrangian relaxation (4) from the paper, not the
    original chance-constrained problem (2) (because PH cannot be directly
    applied to chance-constrained two-stage programs).
'''

import numpy as np
import pyomo.environ as pyo
import mpisppy.scenario_tree as stree

def scenario_creator(scenario_name, node_names=None, cb_data=None):
    if (cb_data is None):
        raise RuntimeError('Must provide a cb_data dict to scenario creator. '
                'At a minimum, this dictionary must contain a key "lam" '
                'specifying the value of the dual multiplier lambda to use, '
                'and a key "solar_filename" specifying where the solar data '
                'is stored.')
    if ('solar_filename' not in cb_data):
        raise RuntimeError('Please provide a cb_data dict that contains '
                           '"solar_filename", with a valid path')
    if ('lam' not in cb_data):
        raise RuntimeError('Please provide a cb_data dict that contains '
                           '"lam", a value of the dual multiplier lambda.')

    data = getData(cb_data['solar_filename'])
    num_scenarios = data['solar'].shape[0]
    scenario_index = extract_scenario_index(scenario_name)
    if (scenario_index < 0) or (scenario_index >= num_scenarios):
        raise RuntimeError('Provided scenario index is invalid (must lie in '
                           '{0,1,...' + str(num_scenarios-1) + '} inclusive)')
    if ('use_LP' in cb_data):
        use_LP = cb_data['use_LP']
    else:
        use_LP = False

    model = pyo.ConcreteModel()

    T   = range(data['T'])
    Tm1 = range(data['T']-1)

    model.y = pyo.Var(T, within=pyo.NonNegativeReals)
    model.p = pyo.Var(T, bounds=(0, data['cMax']))
    model.q = pyo.Var(T, bounds=(0, data['dMax']))
    model.x = pyo.Var(T, bounds=(data['eMin'], data['eMax']))
    if (use_LP):
        model.z = pyo.Var([0], within=pyo.UnitInterval)
    else:
        model.z = pyo.Var([0], within=pyo.Binary)

    ''' "Flow balance" constraints '''
    def flow_balance_constraint_rule(model, t):
        return model.x[t+1]==model.x[t] + \
            data['eff'] * model.p[t] - (1/data['eff']) * model.q[t]
    model.flow_constr = pyo.Constraint(Tm1, rule=flow_balance_constraint_rule)
    
    ''' Big-M constraints '''
    def big_M_constraint_rule(model, t):
        return model.y[t] - model.q[t] + model.p[t] \
            <= data['solar'][scenario_index,t] + \
            data['M'][scenario_index,t] * model.z[0] # Why indexed??
    model.big_m_constr = pyo.Constraint(T, rule=big_M_constraint_rule)

    ''' Objective function (must be minimization or PH crashes) '''
    model.obj = pyo.Objective(expr=-pyo.dot_product(data['rev'], model.y)
        + data['char'] * pyo.quicksum(model.p)
        + data['disc'] * pyo.quicksum(model.q) + cb_data['lam'] * model.z[0],
        sense=pyo.minimize)

    fscr = lambda model: pyo.dot_product(data['rev'], model.y)
    model.first_stage_cost = pyo.Expression(rule=fscr)

    model._PySPnode_list = [
        stree.ScenarioNode(name='ROOT', cond_prob=1., stage=1,
            cost_expression=model.first_stage_cost, scen_name_list=None, 
            nonant_list=[model.y], scen_model=model)
    ]

    return model

def scenario_denouement(rank, scenario_name, scenario):
    pass

def getData(solar_filename):
    ''' Define problem parameters based on the Singh-Knueven paper '''
    data = {
        'T'    : 24,       # Number of time periods (hours) |T|
        'N'    : 50,       # Number of scenarios |\Omega|
        'eff'  : 0.9,      # Battery efficiency eta
        'eMax' : 960,      # Maximum energy that can be stored in the battery
        'eMin' : 192,      # Minimum energy that can be stored in the battery
        'rev'  : np.array( # Marginal revenue earned at each hour R_t
               [0.0189, 0.0172, 0.0155, 0.0148, 0.0146, 0.0151, 0.0173, 0.0219,
                0.0227, 0.0226, 0.0235, 0.0242, 0.0250, 0.0261, 0.0285, 0.0353, 
                0.0531, 0.0671, 0.0438, 0.0333, 0.0287, 0.0268, 0.0240, 0.0211]),
        'char' : 0.0256,   # Cost of    charging the battery
        'disc' : 0.0256,   # Cost of discharging the battery
        'cMax' : 480,      # Maximum    charging rate of battery
        'dMax' : 480,      # Maximum discharging rate of battery
        'eps'  : 0.05,     # Chance constraint violation probability
        'x0'   : 0.5 * 960,# Initial level of battery charge
    }
    data['prob'] = 1/data['N'] * np.ones(data['N']) # Scenario probabilities
    data['solar'] = np.loadtxt(solar_filename, delimiter=',')
    data['M'] = getBigM(data)
    return data

def getBigM(data):
    ''' Compute big-M values as given in Corollary 1 
    
        This method assumes that all scenarios are equally likely
    '''
    base = np.min([data['dMax'], data['eff'] * (data['eMax'] - data['eMin'])])
    M    = base * np.ones((data['N'], data['T'])) - data['solar']
    ell  = np.int64(np.floor(data['N'] * data['eps']) + 1)
    M   += np.sort(data['solar'], axis=0)[-ell,:]
    return M

def extract_scenario_index(scenario_name):
    ix = -1
    while (ix >= -len(scenario_name) and scenario_name[ix] in '0123456789'):
        ix -= 1
    if (ix == -1):
        raise RuntimeError('Provided scenario name must end with a number')
    return int(scenario_name[ix+1:])

if __name__=='__main__':
    print('No main for battery.py')
