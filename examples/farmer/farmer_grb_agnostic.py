# In this example, AMPL is the guest language
# *** This is a special example where this file serves

import gurobipy as gp
from gurobipy import GRB
# since we are working with Gurobi directly we can just use the model directly
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

    gurobipy = gp.read('two_stage_farmer_model.lp')
    
    # scenario specific data applied
    scennum = sputils.extract_num(scenario_name)
    assert scennum < 3, "three scenarios hardwired for now"
    # y = gurobipy.getParam('Random_Yield')
    if scennum == 0:    # below
        gurobipy.setParam('Random_Yield', {"wheat": 2.0, "corn": 2.4, "beets": 16.0})
    elif scennum == 2:  # above
        gurobipy.setParam('Random_Yield', {"wheat": 3.0, "corn": 3.6, "beets": 24.0}) 

    areaVarDatas = [var for var in gurobipy.getVars() if var.varName.startswith('area')]

    gurobipy.update() # not sure if this is needed
    # In general, be sure to process variables in the same order has the guest does (so indexes match)
    gd = {
        "scenario": gurobipy,
        "nonants": {("ROOT",i): v[1] for i,v in enumerate(areaVarDatas)},
        "nonant_fixedness": {("ROOT",i): v.VType == gp.GRB.BINARY for i,v in enumerate(areaVarDatas)},
        "nonant_start": {("ROOT",i): v.X for i,v in enumerate(areaVarDatas)},
        "nonant_names": {("ROOT",i): (v.VarName, i) for i,v in enumerate(areaVarDatas)},
        "probability": "uniform",
        "sense": sense,
        "BFs": None
    }

    return gd

#==========
def scenario_names_creator(num_scens,start=None):
    return farmer.scenario_names_creator(num_scens,start)

#==========
def inparser_adder(cfg):
    return farmer.inparser_adder(cfg)

#==========
def kw_creator(cfg):
    # creates keywords for scenario creator
    return farmer.kw_creator(cfg)

# This is not needed for PH
def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    return farmer.sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                                           given_scenario, **scenario_creator_kwargs)

def scenario_denouement(rank, scenario_name, scenario):
    pass
    # (the fct in farmer won't work because the Var names don't match)
    #farmer.scenario_denouement(rank, scenario_name, scenario)

##################################################################################################
# begin callouts
# NOTE: the callouts all take the Ag object as their first argument, mainly to see cfg if needed
# the function names correspond to function names in mpisppy

def attach_Ws_and_prox(Ag, sname, scenario):
    # this is gurobipy farmer specific, so we know there is not a W already 
    # Attach W's and rho to the guest scenario
    gs = scenario._agnostic_dict["scenario"]
    gd = scenario._agnostic_dict

    crops = ['wheat', 'corn', 'beets']

    # Mutable params for guest scenario 
    gs.__dict__['W_on'] = 0
    gs.__dict__['prox_on'] = 0
    gs.__dict__['W'] = {crop: 0 for crop in crops}
    gs.__dict__['rho'] = {crop: 0 for crop in crops}

    # addting to _agnostic_dict for easy access
    gd['W_on'] = gs.__dict__['W_on']
    gd['prox_on'] = gs.__dict__['prox_on']
    gd['W'] = gs.__dict__['W']
    gd['rho'] = gs.__dict__['rho']

def _disable_prox(Ag, scenario):
    gs = scenario._agnostic_dict["scenario"]    # guest scenario handle
    gs['W_on'] = 0
    # gs.setParam('W_on', 0)

def _disable_prox(Ag, scenario):
    gs = scenario._agnostic_dict["scenario"]    # guest scenario handle
    gs['prox_on'] = 0
    # gs.setParam('prox_on', 0)

def _reenable_prox(Ag, scenario):
    gs = scenario._agnostic_dict["scenario"]    # guest scenario handle
    gs['prox_on'] = 1
    # gs.setParam('prox_on', 1)

def _reenable_W(Ag, scenario):
    gs = scenario._agnostic_dict["scenario"]    # guest scenario handle
    gs['W_on'] = 1
    # gs.setParam('W_on', 1)


def attach_PH_to_objective(Ag, sname, scenario, add_duals, add_prox):
    # Deal with prox linearization and approximation later,
    # i.e., just do the quadratic version

    # The host has xbars and computes without involving the guest language
    gd = scenario._agnostic_dict
    gs = gd["scenario"]  # guest scenario handle

    crops = ['wheat', 'corn', 'beets']
    gs.__dict__['xbars'] = {crop: 0 for crop in crops}
 
    # Dual term (weights W)
    try:
        profitobj = gs.get_objective("minus_profit")
    except:
        print("oh noes! we can't find the objective function")
        print("doing export to export.")
        raise

    obj_expr = original_obj.getValue()
    
    # Add dual terms to the objective function
    if add_duals:
        for crop in crops:
            obj_expr += gs.__dict__['W'][crop] * gs.getVarByName(f'area[{crop}]')

    # Add proximal terms to the objective function
    if add_prox:
        for crop in crops:
            area = gs.getVarByName(f'area[{crop}]')
            xbar = gs.__dict__['xbars'][crop]
            rho = gs.__dict__['rho'][crop]
            obj_expr += (rho / 2.0) * (area * area - 2.0 * xbar * area + xbar * xbar)

    # Set the new objective function
    gs.setObjective(obj_expr, gp.GRB.MINIMIZE)
    gs.update()

    # Store parameters for Progressive Hedging in the _agnostic_dict
    gd["PH"] = {
        "W": gs.__dict__['W'],
        "xbars": gs.__dict__['xbars'],
        "rho": gs.__dict__['rho'],
        "obj": gs.getObjective()
    }

def solve_one(Ag, s, solve_keyword_args, gripe, tee):
    _copy_Ws_xbars_rho_from_host(s)

    gd = s._agnostic_dict
    gs = gd["scenario"]

    """
    Debugging but need to change for gurobipy

    #### start debugging
    if global_rank == 0:
    print(f" in _solve_one W = {gs.__dict__.get('W')}, {global_rank =}")
    print(f" in _solve_one xbars = {gs.__dict__.get('xbars')}, {global_rank =}")
    print(f" in _solve_one rho = {gs.__dict__.get('rho')}, {global_rank =}")`

    #### stop debugging
    """

    solver_name = s._solver_plugin.name
    gs.set_option("solver", solver_name)
    if 'persitent' in solver_name:
        raise RuntimeError("Persistent solvers are not currently supported in the farmer agnostic example.")
    gs.set_option("presolve", 0)

    solver_exception = None

    try:
        gs.optimize()
    except gp.GurobiError as e:
        solver_exception = e

    if gs.status != gp.GRB.OPTIMAL:
        s._mpisppy_data.scenario_feasible = False
        if gripe:
            print (f"Solve failed for scenario {s.name} on rank {global_rank}")
            print(f"{gs.solve_result =}")
            
    if solver_exception is not None:
        raise solver_exception

    s._mpisppy_data.scenario_feasible = True
    objval = gs.objVal

    # If statemtn is useless but might need it later on
    if gd["sense"] == pyo.minimize:
        s._mpisppy_data.outer_bound = objval
    else:
        s._mpisppy_data.outer_bound = objval

    # copy the nonant x values from gs to s so mpisppy can use them in s
    # in general, we need more checks (see the pyomo agnostic guest example)
    for ndn_i, gxvar in gd["nonants"].items():
        try:
            gxvar_val = gxvar.x
        except AttributeError:

            raise RuntimeError(
                            f"Non-anticipative variable {gxvar.varName} on scenario {s.name} "
                            "had no value. This usually means this variable "
                            "did not appear in any (active) components, and hence "
                            "was not communicated to the subproblem solver. ")
        if gxvar.varName not in gs.getVars():
            raise RuntimeError(
                f"Non-anticipative variable {gxvar.varName} on scenario {s.name} "
                "was presolved out. This usually means this variable "
                "did not appear in any (active) components, and hence "
                "was not communicated to the subproblem solver. ")

        s._mpisppy_data.nonant_indices[ndn_i]._value = gxvar_val

    s._mpisppy_data._obj_from_agnostic = objval

# local helper
def _copy_Ws_xbars_rho_from_host(s):
    # print(f"   debug copy_Ws {s.name =}, {global_rank =}")
    gd = s._agnostic_dict
    gs = gd["scenario"]  # guest scenario handle

    # We can't use a simple list because of indexes, we have to use a dict
    # NOTE that we know that W is indexed by crops for this problem
    #  and the nonant_names are tuple with the index in the 1 slot
    # AMPL params are tuples (index, value), which are immutable
    if hasattr(s._mpisppy_model, "W"):
        Wict = {gd["nonant_names"][ndn_i][1]: v.X for ndn_i, v in s._mpisppy_model.W.items()}
        rho_dict = {gd["nonant_names"][ndn_i][1]: v.X for ndn_i, v in s._mpisppy_model.rho.items()}
        xbars_dict = {gd["nonant_names"][ndn_i][1]: v.X for ndn_i, v in s._mpisppy_model.xbars.items()}


        gs.__dict__['W'].update(Wdict)
        gs.__dict__['rho'].update(rho_dict)
        gs.__dict__['xbars'].update(xbars_dict)
        
        gs.update()
    else:
        pass  # presumably an xhatter; we should check, I suppose


"""
In farmer_ampl_agnostic.py these helpers are created but never used

def _copy_nonants_from_host(s):
    # values and fixedness; 
    gd = s._agnostic_dict
    for ndn_i, gxvar in gd["nonants"].items():
        hostVar = s._mpisppy_data.nonant_indices[ndn_i]
        guestVar = gd["nonants"][ndn_i]
        if guestVar.astatus() == "fixed":
            guestVar.unfix()
        if hostVar.is_fixed():
            guestVar.fix(hostVar._value)
        else:
            guestVar.set_value(hostVar._value)


def _restore_nonants(Ag, s):
    # the host has already restored
    _copy_nonants_from_host(s)

    
def _restore_original_fixedness(Ag, s):
    # The host has restored already
    #  Note that this also takes values from the host, which should be OK
    _copy_nonants_from_host(s)


def _fix_nonants(Ag, s):
    # the host has already fixed
    _copy_nonants_from_host(s)


def _fix_root_nonants(Ag, s):
    # the host has already fixed
    _copy_nonants_from_host(s)
"""
