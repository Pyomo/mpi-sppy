###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# In this example, Gurobipy is the guest language
import gurobipy as gp
from gurobipy import GRB
import pyomo.environ as pyo
import examples.farmer.farmer as farmer
import example.farmer.farmer_gurobipy_mod as farmer_gurobipy_mod
import numpy as np

from mpisppy import MPI  # for debugging
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

farmerstream = np.random.RandomState()


def scenario_creator(scenario_name, use_integer=False, sense=GRB.MINIMIZE, crops_multiplier=1, num_scens=None, seedoffset=0):
    """ Create a scenario for the (scalable) farmer example
    
    Args:
        scenario_name (str):
            Name of the scenario to construct.
        use_integer (bool, optional):
            If True, restricts variables to be integer. Default is False.
        sense (int, optional):
            gurobipy sense (minimization or maximization). Must be either
            GRB.MINIMIZE or GRB.MAXIMIZE. Default is GRB.MINIMIZE.
        crops_multiplier (int, optional):
            Factor to control scaling. There will be three times this many
            crops. Default is 1.
        num_scens (int, optional):
            Number of scenarios. We use it to compute _mpisppy_probability. 
            Default is None.
        seedoffset (int): used by confidence interval code

    NOTE: 
    Most other people wont have the model file already written
    """
    gd = farmer_gurobipy_mod.scenario_creator(scenario_name, use_integer, sense, crops_multiplier)

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
    # Gurobipy is special so we are gonna need to maintain the coeffs ourselves
    gs = scenario._agnostic_dict["scenario"]  # guest scenario handle
    gd = scenario._agnostic_dict
    obj = gs.getObjective()

    obj_func_terms = [obj.getVar(i) for i in range(obj.size())]
    nonant_coeffs = {}
    nonants_not_in_obj = {}

    # Check to see if the nonants are in the objective function
    for i, nonant in gd["nonants"].items():
        found_nonant_in_obj = False
        for obj_func_term in obj_func_terms:
            if obj_func_term.sameAs(nonant):
                nonant_coeffs[nonant] = nonant.Obj
                found_nonant_in_obj = True
                break
        if not found_nonant_in_obj:
            print('We dont have a coeff for this nonant which is bad so we will default to 0')
            nonant_coeffs[nonant] = 0
            nonants_not_in_obj[i] = nonant

    # Update/attach nonants' coeffs to the dictionary 
    gd["nonant_coeffs"] = nonant_coeffs 
    gd["nonants_not_in_obj"] = nonants_not_in_obj


def _disable_prox(Ag, scenario):
    pass
    # raise RuntimeError("Did not expect _disable_prox")

def _disable_W(Ag, scenario):
    pass
    # raise RuntimeError("Did not expect _disable_W")

def _reenable_prox(Ag, scenario):
    pass
    # raise RuntimeError("Did not expect _reenable_prox")

def _reenable_W(Ag, scenario):
    pass
    # raise RuntimeError("Did not expect _reenable_W")

def attach_PH_to_objective(Ag, sname, scenario, add_duals, add_prox):
    gs = scenario._agnostic_dict["scenario"]  # guest scenario handle
    gd = scenario._agnostic_dict
    nonant_coeffs = gd["nonant_coeffs"]

    '''
    At this point we can assume that all that all the nonants are already in the objective function from attach_Ws_and_prox  
    but for the nonants that are not in the objective function we can just attach it to the objective funciton with thier respective coefficients
    '''

    # Adding all the nonants into the obj function
    obj = gs.getObjective()
    for nonant_not_in_obj in gd["nonants_not_in_obj"]:
        obj += (nonant_coeffs[nonant_not_in_obj] * nonant_not_in_obj)

    # At this point all nonants are in the obj
    nonant_sqs = {}
    # Need to create a new var for x^2, and then add a constraint to the var with x val, so that we can set coeff value later on
    for i, nonant in gd["nonants"].items():
        # Create a constaint that sets x * x = xsq 
        nonant_sq = gs.addVar(vtype=GRB.CONTINUOUS, obj=nonant.Obj**2, name=f"{nonant.VarName}sq")
        gs.addConstr(nonant * nonant == nonant_sq, f'{nonant.VarName}sqconstr')
        # Put the x^2 in the objective function
        obj += nonant_sq
        nonant_sqs[i] = nonant_sq
    
    # Update model and gd
    gs.update()
    gd["nonant_sqs"] = nonant_sqs

    _copy_Ws_xbars_rho_from_host(scenario)

def solve_one(Ag, s, solve_keyword_args, gripe, tee, need_solution=True):
    _copy_Ws_xbars_rho_from_host(s)
    gd = s._agnostic_dict
    gs = gd["scenario"]  # guest scenario handle

    solver_exception = None
    # Assuming gs is a Gurobi model, we can start solving
    try:
        gs.optimize()
    except gp.GurobiError as e:
        s._mpisppy_data.scenario_feasible = False
        solver_exception = e
        if gripe:
            print(f"Solve failed with error {str(e)} for scenario {s.name}")
        return

    # TBD: what about not optimal and need_solution?
    if solver_exception is not None and need_solution:
        raise solver_exception

    if gs.status != gp.GRB.Status.OPTIMAL:
        s._mpisppy_data.scenario_feasible = False
        if gripe:
            print(f"Solve failed for scenario {s.name}")
        return

    s._mpisppy_data.scenario_feasible = True
    
    # Objective value extraction
    objval = gs.getObjective().getValue()
    
    if gd["sense"] == gp.GRB.MINIMIZE:
        s._mpisppy_data.outer_bound = objval
    else:
        s._mpisppy_data.outer_bound = objval
    
    # Copy the non-anticipative variable values from guest to host scenario
    for ndn_i, gxvar in gd["nonants"].items():
        grb_var = gs.getVarByName(gxvar.VarName)
        if grb_var is None:
            raise RuntimeError(
                f"Non-anticipative variable {gxvar.varname} on scenario {s.name} "
                "was not found in the Gurobi model."
            )
        s._mpisppy_data.nonant_indices[ndn_i]._value = grb_var.X
    
    # Store the objective function value in the host scenario
    s._mpisppy_data._obj_from_agnostic = objval

    # Additional checks and operations for bundling if needed (depending on the problem)
    # ...

def _copy_Ws_xbars_rho_from_host(scenario):
    # Calculates the coefficients of the new expanded objective function
    # Regardless need to calculate coefficients for x^2 and x
    gd = scenario._agnostic_dict
    gs = scenario._agnostic_dict["scenario"]    # guest handle

    # Decide if we are using PH or xhatter
    if hasattr(scenario._mpisppy_model, "W"):
        # Get our Ws, rhos, and xbars
        Wdict = {gd["nonant_names"][ndn_i]:\
                 pyo.value(v) for ndn_i, v in scenario._mpisppy_model.W.items()}
        rhodict = {gd["nonant_names"][ndn_i]:\
                   pyo.value(v) for ndn_i, v in scenario._mpisppy_model.rho.items()}
        xbarsdict = {gd["nonant_names"][ndn_i]:\
                   pyo.value(v) for ndn_i, v in scenario._mpisppy_model.xbars.items()}

        # Get data from host model
        nonants_coeffs = gd["nonants_coeffs"] 
        host_model = scenario._mpisppy_model
        W_on = host_model.W_on.value
        prox_on = host_model.prox_on.value
        # Update x coeff and x^2 coeff
        for i, nonant in gd["nonants"].items():     # (Root, 1) : Nonant
            new_coeff_val_xvar = nonants_coeffs[i] + W_on * (Wdict[nonant.VarName] * nonants_coeffs[i]) - prox_on * (rhodict[nonant.VarName] * xbarsdict[nonant.VarName]) 
            new_coeff_val_xsq = prox_on * rhodict[nonant.VarName]/2.0
            nonant.Obj = new_coeff_val_xvar
            gd["nonant_sqs"][i].Obj = new_coeff_val_xsq

        gs.update()
    else:
        pass # presumably an xhatter; we should check, I suppose

def _copy_nonants_from_host(s):
    gd = s._agnostic_dict
    for ndn_i, gxvar in gd["nonants"].items():
        hostVar = s._mpisppy_data.nonant_indices[ndn_i]
        guestVar = gd["nonants"][ndn_i]
        if guestVar.lb == guestVar.ub:
            guestVar.lb = 0 
            guestVar.ub = float("inf") 
        if hostVar.is_fixed():
            guestVar.lb = hostVar._value
            guestVar.ub = hostVar._value
        else:
            guestVar.Start = hostVar._value

def _restore_nonants(Ag, s=None):
    _copy_nonants_from_host(s)

def _restore_original_fixedness(Ag, scenario):
    _copy_nonants_from_host(scenario)

def _fix_nonants(Ag, s=None):
    _copy_nonants_from_host(s)

def _fix_root_nonants(Ag, scenario):
    _copy_nonants_from_host(scenario)
