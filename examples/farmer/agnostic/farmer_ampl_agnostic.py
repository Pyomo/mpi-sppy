###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# <special for agnostic debugging DLW Aug 2023>
# In this example, AMPL is the guest language.
# ***This is a special example where this file serves
# the rolls of ampl_guest.py and the python model file.***

"""
This file tries to show many ways to do things in AMPLpy,
but not necessarily the best ways in all cases.
"""

from amplpy import AMPL
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import numpy as np

from mpisppy import MPI  # for debugging
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

# If you need random numbers, use this random stream:
farmerstream = np.random.RandomState()

def scenario_creator(
    scenario_name, use_integer=False, sense=pyo.minimize, crops_multiplier=1,
        num_scens=None, seedoffset=0
):
    """ Create a scenario for the (scalable) farmer example
    
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

    NOTE: for ampl, the names will be tuples name, index
    """

    assert crops_multiplier == 1, "for AMPL, just getting started with 3 crops"

    ampl = AMPL()

    ampl.read("farmer.mod")

    # scenario specific data applied
    scennum = sputils.extract_num(scenario_name)
    assert scennum < 3, "three scenarios hardwired for now"
    y = ampl.get_parameter("RandomYield")
    if scennum == 0:  # below
        y.set_values({"wheat": 2.0, "corn": 2.4, "beets": 16.0})
    elif scennum == 2: # above
        y.set_values({"wheat": 3.0, "corn": 3.6, "beets": 24.0})

    areaVarDatas = list(ampl.get_variable("area").instances())

    # In general, be sure to process variables in the same order has the guest does (so indexes match)
    gd = {
        "scenario": ampl,
        "nonants": {("ROOT",i): v[1] for i,v in enumerate(areaVarDatas)},
        "nonant_fixedness": {("ROOT",i): v[1].astatus()=="fixed" for i,v in enumerate(areaVarDatas)},
        "nonant_start": {("ROOT",i): v[1].value() for i,v in enumerate(areaVarDatas)},
        "nonant_names": {("ROOT",i): ("area", v[0]) for i, v in enumerate(areaVarDatas)},
        "probability": "uniform",
        "sense": pyo.minimize,
        "BFs": None
    }

    return gd
    
#=========
def scenario_names_creator(num_scens,start=None):
    if (start is None) :
        start=0
    return [f"scen{i}" for i in range(start,start+num_scens)]


#=========
def inparser_adder(cfg):
    cfg.num_scens_required()
    cfg.add_to_config("crops_multiplier",
                      description="number of crops will be three times this (default 1)",
                      domain=int,
                      default=1)
    
    cfg.add_to_config("farmer_with_integers",
                      description="make the version that has integers (default False)",
                      domain=bool,
                      default=False)

    
#=========
def kw_creator(cfg):
    # creates keywords for scenario creator
    kwargs = {"use_integer": cfg.get('farmer_with_integers', False),
              "crops_multiplier": cfg.get('crops_multiplier', 1),
              "num_scens" : cfg.get('num_scens', None),
              }
    return kwargs


# This is not needed for PH
def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    # Since this is a two-stage problem, we don't have to do much.
    sca = scenario_creator_kwargs.copy()
    sca["seedoffset"] = seed
    sca["num_scens"] = sample_branching_factors[0]  # two-stage problem
    return scenario_creator(sname, **sca)


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
    # this is AMPL farmer specific, so we know there is not a W already, e.g.
    # Attach W's and rho to the guest scenario (mutable params).
    gs = scenario._agnostic_dict["scenario"]  # guest scenario handle
    # (there must be some way to create and assign *mutable* params in on call to AMPL)
    gs.eval("param W_on;")
    gs.eval("let W_on := 0;")
    gs.eval("param prox_on;")
    gs.eval("let prox_on := 0;")
    # we are trusting the order to match the nonant indexes
    gs.eval("param W{Crops};")
    # Note: we should probably use set_values instead of let
    gs.eval("let {c in Crops}  W[c] := 0;")
    # start with rho at zero, but update before solve
    gs.eval("param rho{Crops};")
    gs.eval("let {c in Crops}  rho[c] := 0;")

    
def _disable_prox(Ag, scenario):
    gs = scenario._agnostic_dict["scenario"]  # guest scenario handle
    gs.get_parameter("prox_on").set(0)

    
def _disable_W(Ag, scenario):
    gs = scenario._agnostic_dict["scenario"]  # guest scenario handle
    gs.get_parameter("W_on").set(0)

    
def _reenable_prox(Ag, scenario):
    gs = scenario._agnostic_dict["scenario"]  # guest scenario handle
    gs.get_parameter("prox_on").set(1)

    
def _reenable_W(Ag, scenario):
    gs = scenario._agnostic_dict["scenario"]  # guest scenario handle
    gs.get_parameter("W_on").set(1)
    
    
def attach_PH_to_objective(Ag, sname, scenario, add_duals, add_prox):
    # Deal with prox linearization and approximation later,
    # i.e., just do the quadratic version

    # The host has xbars and computes without involving the guest language
    gd = scenario._agnostic_dict
    gs = gd["scenario"]  # guest scenario handle
    gs.eval("param xbars{Crops};")

    # Dual term (weights W)
    try:
        profitobj = gs.get_objective("minus_profit")
    except:
        print("big troubles!!; we can't find the objective function")
        print("doing export to export.mod")
        gs.export_model("export.mod")
        raise
        
    objstr = str(profitobj)
    phobjstr = ""
    if add_duals:
        phobjstr += " + W_on * sum{c in Crops} (W[c] * area[c])"
        
    # Prox term (quadratic)
    if add_prox:
        """
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
                (xvarsqrd - 2.0 * xbars[ndn_i] * xvar + xbars[ndn_i]**2)
        """
        phobjstr += " + prox_on * sum{c in Crops} ((rho[c]/2.0) * (area[c] * area[c] "+\
                    " - 2.0 * xbars[c] * area[c] + xbars[c]^2))"
    objstr = objstr[:-1] + "+ (" + phobjstr + ");"
    objstr = objstr.replace("minimize minus_profit", "minimize phobj")
    profitobj.drop()
    gs.eval(objstr)
    gs.eval("delete minus_profit;")
    currentobj = gs.get_current_objective()
    # see _copy_Ws_...  see also the gams version
    WParamDatas = list(gs.get_parameter("W").instances())
    xbarsParamDatas = list(gs.get_parameter("xbars").instances())
    rhoParamDatas = list(gs.get_parameter("rho").instances())
    gd["PH"] = {
        "W": {("ROOT",i): v for i,v in enumerate(WParamDatas)},
        "xbars": {("ROOT",i): v for i,v in enumerate(xbarsParamDatas)},
        "rho": {("ROOT",i): v for i,v in enumerate(rhoParamDatas)},
        "obj": currentobj,
    }


def solve_one(Ag, s, solve_keyword_args, gripe, tee, need_solution=True):
    # This needs to attach stuff to s (see solve_one in spopt.py)
    # Solve the guest language version, then copy values to the host scenario

    # This function needs to  W on the guest right before the solve

    # We need to operate on the guest scenario, not s; however, attach things to s (the host scenario)
    # and copy to s. If you are working on a new guest, you should not have to edit the s side of things

    # To acommdate the solve_one call from xhat_eval.py, we need to attach the obj fct value to s

    # time.sleep(np.random.uniform()/10)
    
    _copy_Ws_xbars_rho_from_host(s)
    gd = s._agnostic_dict
    gs = gd["scenario"]  # guest scenario handle

    #### start debugging
    if False:  # True:  # global_rank == 0:
        try:
            WParamDatas = list(gs.get_parameter("W").instances())
            print(f" ^^^ in _solve_one {WParamDatas =} {global_rank =}")
        except:  # noqa
            print(f"    ^^^^ no W for xhat {global_rank=}")
        #prox_on = gs.get_parameter("prox_on").value()
        #print(f" ^^^ in _solve_one {prox_on =} {global_rank =}")
        #W_on = gs.get_parameter("W_on").value()
        #print(f" ^^^ in _solve_one {W_on =} {global_rank =}")
        #xbarsParamDatas = list(gs.get_parameter("xbars").instances())
        #print(f" in _solve_one {xbarsParamDatas =} {global_rank =}")
        #rhoParamDatas = list(gs.get_parameter("rho").instances())
        #print(f" in _solve_one {rhoParamDatas =} {global_rank =}")
    #### stop debugging
    
    solver_name = s._solver_plugin.name
    gs.set_option("solver", solver_name)    
    if 'persistent' in solver_name:
        raise RuntimeError("Persistent solvers are not currently supported in the farmer agnostic example.")
    gs.set_option("presolve", 0)

    solver_exception = None
    try:
        gs.solve()
    except Exception as e:
        solver_exception = e

    # debug
    #fname = f"{s.name}_{global_rank}"
    #print(f"debug export to {fname}")
    #gs.export_model(f"{fname}.mod")
    #gs.export_data(f"{fname}.dat")
    
    if gs.solve_result != "solved":
        s._mpisppy_data.scenario_feasible = False
        if gripe:
            print (f"Solve failed for scenario {s.name} on rank {global_rank}")
            print(f"{gs.solve_result =}")
            
    if solver_exception is not None and need_solution:
        raise solver_exception


    s._mpisppy_data.scenario_feasible = True
    # For AMPL mips, we need to use the gap option to compute bounds
    # https://amplmp.readthedocs.io/rst/features-guide.html
    objobj = gs.get_current_objective()  # different for xhatters
    objval = objobj.value()
    if gd["sense"] == pyo.minimize:
        s._mpisppy_data.outer_bound = objval
    else:
        s._mpisppy_data.inner_bound = objval

    # copy the nonant x values from gs to s so mpisppy can use them in s
    # in general, we need more checks (see the pyomo agnostic guest example)
    for ndn_i, gxvar in gd["nonants"].items():
        try:   # not sure this is needed
            float(gxvar.value())
        except:  # noqa
            raise RuntimeError(
                f"Non-anticipative variable {gxvar.name} on scenario {s.name} "
                "had no value. This usually means this variable "
                "did not appear in any (active) components, and hence "
                "was not communicated to the subproblem solver. ")
        if gxvar.astatus() == "pre":
            raise RuntimeError(
                f"Non-anticipative variable {gxvar.name} on scenario {s.name} "
                "was presolved out. This usually means this variable "
                "did not appear in any (active) components, and hence "
                "was not communicated to the subproblem solver. ")

        s._mpisppy_data.nonant_indices[ndn_i]._value = gxvar.value()

    # the next line ignores bundling
    s._mpisppy_data._obj_from_agnostic = objval

    # TBD: deal with other aspects of bundling (see solve_one in spopt.py)


# local helper called right before the solve
def _copy_Ws_xbars_rho_from_host(s):
    # print(f"   debug copy_Ws {s.name =}, {global_rank =}")
    gd = s._agnostic_dict
    gs = gd["scenario"]  # guest scenario handle

    # We can't use a simple list because of indexes, we have to use a dict
    # NOTE that we know that W is indexed by crops for this problem
    #  and the nonant_names are tuple with the index in the 1 slot
    # AMPL params are tuples (index, value), which are immutable
    if hasattr(s._mpisppy_model, "W"):
        Wdict = {gd["nonant_names"][ndn_i][1]:\
                 pyo.value(v) for ndn_i, v in s._mpisppy_model.W.items()}
        gs.get_parameter("W").set_values(Wdict)
        rhodict = {gd["nonant_names"][ndn_i][1]:\
                   pyo.value(v) for ndn_i, v in s._mpisppy_model.rho.items()}
        gs.get_parameter("rho").set_values(rhodict)
        xbarsdict = {gd["nonant_names"][ndn_i][1]:\
                   pyo.value(v) for ndn_i, v in s._mpisppy_model.xbars.items()}
        gs.get_parameter("xbars").set_values(xbarsdict)
        # debug
        fname = f"{s.name}_{global_rank}"
        print(f"debug export to {fname}")
        gs.export_model(f"{fname}.mod")
        gs.export_data(f"{fname}.dat")
        
    else:
        pass  # presumably an xhatter; we should check, I suppose
        

# local helper
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

    
    
