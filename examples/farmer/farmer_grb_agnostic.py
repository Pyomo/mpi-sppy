# In this example, AMPL is the guest language
# *** This is a special example where this file serves

import gurobipy as gp
from gurobipy import GRB
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import farmer
import numpy as np

farmerstream = np.random.RandomState()

# debugging
from mpisppy import MPI
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

def scenario_creator(scenario_name, use_integer=False, sense=pyo.minimize, crops_multiplier=1, num_scens=None, seedoffset=0):
    """ Create a scenario for the (scalable) farmer example
    
    Args:
        scenario_name (str):
            Name of the scenario to construct.
        use_integer (bool, optional):
            If True, restricts variables to be integer. Default is False.
        sense (int, optional):
            gurobipy sense (minimization or maximization). Must be either
            pyo.minimize or pyo.maximize. Default is pyo.minimize.
        crops_multiplier (int, optional):
            Factor to control scaling. There will be three times this many
            crops. Default is 1.
        num_scens (int, optional):
            Number of scenarios. We use it to compute _mpisppy_probability. 
            Default is None.
        seedoffset (int): used by confidence interval code

    NOTE: 
    """
    assert crops_multiplier == 1,   "for gurobipy, just getting started with 3 crops"
    gurobipy = gp.Model()

    gurobipy.load('two_stage_farmer_model.lp')
    
    # scenario specific data applied
    scennum = sputils.extract_num(scenario_name)
    assert scennum < 3, "three scenarios hardwired for now"
    # y = gurobipy.getParam('Random_Yield')
    if scennum == 0:    # below
        gurobipy.setParam('Random_Yield', {"wheat": 2.0, "corn": 2.4, "beets": 16.0})
    elif scennum == 2:  # above
        gurobipy.setParam('Random_Yield', {"wheat": 3.0, "corn": 3.6, "beets": 24.0}) 

    areaVarDatas = [var for var in gurobipy.getVars() if var.varName.startswith('area')]

    # In general, be sure to process variables in the same order has the guest does (so indexes match)
    gd = {
        "scenario": gurobipy,
        "nonants": {("ROOT",i): v[1] for i,v in enumerate(areaVarDatas)},
        "nonant_fixedness": {("ROOT",i): v[1].astatus()=="fixed" for i,v in enumerate(areaVarDatas)},
        "nonant_start": {("ROOT",i): v[1].value() for i,v in enumerate(areaVarDatas)},
        "nonant_names": {("ROOT",i): ("area", v[0]) for i,v in enumerate(areaVarDatas)},
        "probability": "uniform",
        "sense": pyo.minimize,
        "BFs": None
    }

    return gd

