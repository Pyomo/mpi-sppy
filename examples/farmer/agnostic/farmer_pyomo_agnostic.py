###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# <special for agnostic debugging DLW Aug 2023>
# In this example, Pyomo is the guest language just for
# testing and documentation purposed.
"""
For other guest languages, the corresponding module is
still written in Python, it just needs to interact
with the guest language
"""

import pyomo.environ as pyo
from pyomo.opt import SolutionStatus, TerminationCondition
import farmer   # the native farmer (makes a few things easy)

# for debuggig
from mpisppy import MPI
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

def scenario_creator(
    scenario_name, use_integer=False, sense=pyo.minimize, crops_multiplier=1,
        num_scens=None, seedoffset=0
):
    """ Create a scenario for the (scalable) farmer example, but
   but treat Pyomo as a guest language.
    
    Args:
        scenario_name (str):
            Name of the scenario to construct.
        use_integer (bool, optional):
            If True, restricts variables to be integer. Default is False.
        sense (int, optional):
            Model sense (minimization or maximization). Must be either
            pyo.minimize or pyo.maximize. Default is pyo.minimize.
        crops_multiplier (int, optional):
            Factor to control scaling. There will be three times this many
            crops. Default is 1.
        num_scens (int, optional):
            Number of scenarios. We use it to compute _mpisppy_probability. 
            Default is None.
        seedoffset (int): used by confidence interval code
    """
    s = farmer.scenario_creator(scenario_name, use_integer, sense, crops_multiplier,
        num_scens, seedoffset)
    # In general, be sure to process variables in the same order has the guest does (so indexes match)
    gd = {
        "scenario": s,
        "nonants": {("ROOT",i): v for i,v in enumerate(s.DevotedAcreage.values())},
        "nonant_fixedness": {("ROOT",i): v.is_fixed() for i,v in enumerate(s.DevotedAcreage.values())},
        "nonant_start": {("ROOT",i): v._value for i,v in enumerate(s.DevotedAcreage.values())},
        "nonant_names": {("ROOT",i): v.name for i, v in enumerate(s.DevotedAcreage.values())},
        "probability": "uniform",
        "sense": pyo.minimize,
        "BFs": None
        }
    return gd
    
#=========
def scenario_names_creator(num_scens,start=None):
    return farmer.scenario_names_creator(num_scens,start)


#=========
def inparser_adder(cfg):
    farmer.inparser_adder(cfg)

    
#=========
def kw_creator(cfg):
    # creates keywords for scenario creator
    return farmer.kw_creator(cfg)

# This is not needed for PH
def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    return farmer.sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                                           given_scenario, **scenario_creator_kwargs)

#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass
    # (the fct in farmer won't work because the Var names don't match)
    #farmer.scenario_denouement(rank, scenario_name, scenario)



##################################################################################################
# begin callouts
# NOTE: the callouts all take the Ag object as their first argument, mainly to see cfg if needed
# the function names correspond to function names in mpisppy

def attach_Ws_and_prox(Ag, sname, scenario):
    # this is farmer specific, so we know there is not a W already, e.g.
    # Attach W's and prox to the guest scenario.
    gs = scenario._agnostic_dict["scenario"]  # guest scenario handle
    nonant_idx = list(scenario._agnostic_dict["nonants"].keys())
    gs.W = pyo.Param(nonant_idx, initialize=0.0, mutable=True)
    gs.W_on = pyo.Param(initialize=0, mutable=True, within=pyo.Binary)
    gs.prox_on = pyo.Param(initialize=0, mutable=True, within=pyo.Binary)
    gs.rho = pyo.Param(nonant_idx, mutable=True, default=Ag.cfg.default_rho)


def _disable_prox(Ag, scenario):
    scenario._agnostic_dict["scenario"].prox_on._value = 0

    
def _disable_W(Ag, scenario):
    scenario._agnostic_dict["scenario"].W_on._value = 0
    
    
def _reenable_prox(Ag, scenario):
    scenario._agnostic_dict["scenario"].prox_on._value = 1

    
def _reenable_W(Ag, scenario):
    scenario._agnostic_dict["scenario"].W_on._value = 1
    
    
def attach_PH_to_objective(Ag, sname, scenario, add_duals, add_prox):
    # Deal with prox linearization and approximation later,
    # i.e., just do the quadratic version

    ### The host has xbars and computes without involving the guest language
    ### xbars = scenario._mpisppy_model.xbars
    ### but instead, we are going to make guest xbars like other guests
    

    gd = scenario._agnostic_dict
    gs = gd["scenario"]  # guest scenario handle
    nonant_idx = list(gd["nonants"].keys())    
    objfct = gs.Total_Cost_Objective  # we know this is farmer...
    ph_term = 0
    gs.xbars = pyo.Param(nonant_idx, mutable=True)
    # Dual term (weights W)
    if add_duals:
        gs.WExpr = pyo.Expression(expr= sum(gs.W[ndn_i] * xvar for ndn_i,xvar in gd["nonants"].items()))
        ph_term += gs.W_on * gs.WExpr
        
        # Prox term (quadratic)
        if (add_prox):
            prox_expr = 0.
            for ndn_i, xvar in gd["nonants"].items():
                # expand (x - xbar)**2 to (x**2 - 2*xbar*x + xbar**2)
                # x**2 is the only qradratic term, which might be
                # dealt with differently depending on user-set options
                if xvar.is_binary():
                    xvarsqrd = xvar
                else:
                    xvarsqrd = xvar**2
                prox_expr += (gs.rho[ndn_i] / 2.0) * \
                    (xvarsqrd - 2.0 * gs.xbars[ndn_i] * xvar + gs.xbars[ndn_i]**2)
            gs.ProxExpr = pyo.Expression(expr=prox_expr)
            ph_term += gs.prox_on * gs.ProxExpr
                    
            if gd["sense"] == pyo.minimize:
                objfct.expr += ph_term
            elif gd["sense"] == pyo.maximize:
                objfct.expr -= ph_term
            else:
                raise RuntimeError(f"Unknown sense {gd['sense'] =}")
            

def solve_one(Ag, s, solve_keyword_args, gripe, tee=False, need_solution=True):
    # This needs to attach stuff to s (see solve_one in spopt.py)
    # Solve the guest language version, then copy values to the host scenario

    # This function needs to  W on the guest right before the solve

    # We need to operate on the guest scenario, not s; however, attach things to s (the host scenario)
    # and copy to s. If you are working on a new guest, you should not have to edit the s side of things

    # To acommdate the solve_one call from xhat_eval.py, we need to attach the obj fct value to s

    _copy_Ws_xbars_rho_from_host(s)
    gd = s._agnostic_dict
    gs = gd["scenario"]  # guest scenario handle

    print(f" in _solve_one  {global_rank =}")
    if global_rank == 0:
        print(f"{gs.W.pprint() =}")
        print(f"{gs.xbars.pprint() =}")
    solver_name = s._solver_plugin.name
    solver = pyo.SolverFactory(solver_name)
    if 'persistent' in solver_name:
        raise RuntimeError("Persistent solvers are not currently supported in the farmer agnostic example.")
        ###solver.set_instance(ef, symbolic_solver_labels=True)
        ###solver.solve(tee=True)
    else:
        solver_exception = None
        try:
            results = solver.solve(gs, tee=tee, symbolic_solver_labels=True,load_solutions=False)
        except Exception as e:
            results = None
            solver_exception = e

    if (results is None) or (len(results.solution) == 0) or \
            (results.solution(0).status == SolutionStatus.infeasible) or \
            (results.solver.termination_condition == TerminationCondition.infeasible) or \
            (results.solver.termination_condition == TerminationCondition.infeasibleOrUnbounded) or \
            (results.solver.termination_condition == TerminationCondition.unbounded):

        s._mpisppy_data.scenario_feasible = False

        if gripe:
            print (f"Solve failed for scenario {s.name} on rank {global_rank}")
            if results is not None:
                print ("status=", results.solver.status)
                print ("TerminationCondition=",
                       results.solver.termination_condition)

        if solver_exception is not None and need_solution:
            raise solver_exception

    else:
        s._mpisppy_data.scenario_feasible = True
        if gd["sense"] == pyo.minimize:
            s._mpisppy_data.outer_bound = results.Problem[0].Lower_bound
        else:
            s._mpisppy_data.outer_bound = results.Problem[0].Upper_bound
        gs.solutions.load_from(results)
        # copy the nonant x values from gs to s so mpisppy can use them in s
        for ndn_i, gxvar in gd["nonants"].items():
            # courtesy check for staleness on the guest side before the copy
            if not gxvar.fixed and gxvar.stale:
                try:
                    float(pyo.value(gxvar))
                except:  # noqa
                    raise RuntimeError(
                        f"Non-anticipative variable {gxvar.name} on scenario {s.name} "
                        "reported as stale. This usually means this variable "
                        "did not appear in any (active) components, and hence "
                        "was not communicated to the subproblem solver. ")
                
            s._mpisppy_data.nonant_indices[ndn_i]._value = gxvar._value

        # the next line ignore bundling
        s._mpisppy_data._obj_from_agnostic = pyo.value(gs.Total_Cost_Objective)

    # TBD: deal with other aspects of bundling (see solve_one in spopt.py)


# local helper
def _copy_Ws_xbars_rho_from_host(s):
    # This is an important function because it allows us to capture whatever the host did
    # print(f"   {s.name =}, {global_rank =}")
    gd = s._agnostic_dict
    gs = gd["scenario"]  # guest scenario handle
    for ndn_i, gxvar in gd["nonants"].items():
        assert hasattr(s, "_mpisppy_model"),\
            f"what the heck!! no _mpisppy_model {s.name =} {global_rank =}"
        if hasattr(s._mpisppy_model, "W"):
            gs.W[ndn_i] = pyo.value(s._mpisppy_model.W[ndn_i])
            gs.rho[ndn_i] = pyo.value(s._mpisppy_model.rho[ndn_i])
            gs.xbars[ndn_i] = pyo.value(s._mpisppy_model.xbars[ndn_i])
        else:
            # presumably an xhatter
            pass


# local helper
def _copy_nonants_from_host(s):
    # values and fixedness; 
    gd = s._agnostic_dict
    for ndn_i, gxvar in gd["nonants"].items():
        hostVar = s._mpisppy_data.nonant_indices[ndn_i]
        guestVar = gd["nonants"][ndn_i]
        if guestVar.is_fixed():
            guestVar.fixed = False
        if hostVar.is_fixed():
            guestVar.fix(hostVar._value)
        else:
            guestVar._value = hostVar._value


def _restore_nonants(Ag, s):
    # the host has already restored
    _copy_nonants_from_host(s)

    
def _restore_original_fixedness(Ag, s):
    _copy_nonants_from_host(s)


def _fix_nonants(Ag, s):
    _copy_nonants_from_host(s)


def _fix_root_nonants(Ag, s):
    _copy_nonants_from_host(s)

    
    
