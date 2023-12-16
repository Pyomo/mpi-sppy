# <special for agnostic debugging DLW Aug 2023>
# In this example, GAMS is the guest language.
# NOTE: unlike everywhere else, we are using xbar instead of xbars (no biggy)

"""
This file tries to show many ways to do things in gams,
but not necessarily the best ways in any case.
"""

import os
import time
import gams
import gamspy_base

this_dir = os.path.dirname(os.path.abspath(__file__))
gamspy_base_dir = gamspy_base.__path__[0]

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import farmer
import numpy as np

# If you need random numbers, use this random stream:
farmerstream = np.random.RandomState()


# for debugging
from mpisppy import MPI
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

def scenario_creator(
    scenario_name, use_integer=False, sense=pyo.minimize, crops_multiplier=1,
        num_scens=None, seedoffset=0
):
    """ Create a scenario for the (scalable) farmer example.
    
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

    assert crops_multiplier == 1, "just getting started with 3 crops"

    ws = gams.GamsWorkspace(working_directory=this_dir, system_directory=gamspy_base_dir)

    job = ws.add_job_from_file("GAMS/farmer_augmented.gms")
    job.run()

    cp = ws.add_checkpoint()
    mi = cp.add_modelinstance()

    job.run(checkpoint=cp)

    crop = mi.sync_db.add_set("crop", 1, "crop type")

    y = mi.sync_db.add_parameter_dc("yield", [crop,], "tons per acre")
    
    ph_W = mi.sync_db.add_parameter_dc("ph_W", [crop,], "ph weight")
    xbar = mi.sync_db.add_parameter_dc("xbar", [crop,], "ph average")
    rho = mi.sync_db.add_parameter_dc("rho", [crop,], "ph rho")

    W_on = mi.sync_db.add_parameter("W_on", 0, "activate w term")
    prox_on = mi.sync_db.add_parameter("prox_on", 0, "activate prox term")

    mi.instantiate("simple min negprofit using nlp",
        [
            gams.GamsModifier(y),
            gams.GamsModifier(ph_W),
            gams.GamsModifier(xbar),
            gams.GamsModifier(rho),
            gams.GamsModifier(W_on),
            gams.GamsModifier(prox_on),
        ],
    )

    # initialize W, rho, xbar, W_on, prox_on
    crops = [ "wheat", "corn", "sugarbeets" ]
    for c in crops:
        ph_W.add_record(c).value = 0
        xbar.add_record(c).value = 0
        rho.add_record(c).value = 0
    W_on.add_record().value = 0
    prox_on.add_record().value = 0

    # scenario specific data applied
    scennum = sputils.extract_num(scenario_name)
    assert scennum < 3, "three scenarios hardwired for now"
    if scennum == 0:  # below
        y.add_record("wheat").value = 2.0
        y.add_record("corn").value = 2.4
        y.add_record("sugarbeets").value = 16.0
    elif scennum == 1: # average
        y.add_record("wheat").value = 2.5
        y.add_record("corn").value = 3.0
        y.add_record("sugarbeets").value = 20.0
    elif scennum == 2: # above
        y.add_record("wheat").value = 3.0
        y.add_record("corn").value = 3.6
        y.add_record("sugarbeets").value = 24.0

    mi.solve()
    areaVarDatas = list( mi.sync_db["x"] ) 

    # In general, be sure to process variables in the same order has the guest does (so indexes match)
    gd = {
        "scenario": mi,
        "nonants": {("ROOT",i): v for i,v in enumerate(areaVarDatas)},
        "nonant_fixedness": {("ROOT",i): v.get_lower() == v.get_upper() for i,v in enumerate(areaVarDatas)},
        "nonant_start": {("ROOT",i): v.get_level() for i,v in enumerate(areaVarDatas)},
        "nonant_names": {("ROOT",i): ("x", v.key(0)) for i, v in enumerate(areaVarDatas)},
        "probability": "uniform",
        "sense": pyo.minimize,
        "BFs": None,
        "ph" : {
            "ph_W" : {("ROOT",i): p for i,p in enumerate(ph_W)},
            "xbar" : {("ROOT",i): p for i,p in enumerate(xbar)},
            "rho" : {("ROOT",i): p for i,p in enumerate(rho)},
            "W_on" : W_on.first_record(),
            "prox_on" : prox_on.first_record(),
            "obj" : mi.sync_db["negprofit"].find_record(),
            "nonant_lbs" : {("ROOT",i): v.get_lower() for i,v in enumerate(areaVarDatas)},
            "nonant_ubs" : {("ROOT",i): v.get_upper() for i,v in enumerate(areaVarDatas)},
        },
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
    # TODO: the current version has this hardcoded in the GAMS model
    # (W, rho, and xbar all get values right before the solve)
    pass


def _disable_prox(Ag, scenario):
    scenario._agnostic_dict["ph"]["prox_on"].set_value(0)

    
def _disable_W(Ag, scenario):
    scenario._agnostic_dict["ph"]["W_on"].set_value(0)

    
def _reenable_prox(Ag, scenario):
    scenario._agnostic_dict["ph"]["prox_on"].set_value(1)

    
def _reenable_W(Ag, scenario):
    scenario._agnostic_dict["ph"]["W_on"].set_value(1)
    
    
def attach_PH_to_objective(Ag, sname, scenario, add_duals, add_prox):
    # TODO: hard coded in GAMS model
    pass


def solve_one(Ag, s, solve_keyword_args, gripe, tee):
    # s is the host scenario
    # This needs to attach stuff to s (see solve_one in spopt.py)
    # Solve the guest language version, then copy values to the host scenario

    # This function needs to put W on the guest right before the solve

    # We need to operate on the guest scenario, not s; however, attach things to s (the host scenario)
    # and copy to s. If you are working on a new guest, you should not have to edit the s side of things

    # To acommdate the solve_one call from xhat_eval.py, we need to attach the obj fct value to s

    _copy_Ws_xbar_rho_from_host(s)
    gd = s._agnostic_dict
    gs = gd["scenario"]  # guest scenario handle

    solver_name = s._solver_plugin.name   # not used?

    solver_exception = None
    try:
        gs.solve()
    except Exception as e:
        results = None
        solver_exception = e
    print(f"debug {gs.model_status =}")
    time.sleep(1)  # just hoping this helps...
    
    solve_ok = (1, 2, 7, 8, 15, 16, 17)

    if gs.model_status not in solve_ok:
        s._mpisppy_data.scenario_feasible = False
        if gripe:
            print (f"Solve failed for scenario {s.name} on rank {global_rank}")
            print(f"{gs.model_status =}")
            
    if solver_exception is not None:
        raise solver_exception

    s._mpisppy_data.scenario_feasible = True

    ## TODO: how to get lower bound??
    objval = gd["ph"]["obj"].get_level()  # use this?
    ###phobjval = gs.get_objective("phobj").value()   # use this???
    s._mpisppy_data.outer_bound = objval

    # copy the nonant x values from gs to s so mpisppy can use them in s
    # in general, we need more checks (see the pyomo agnostic guest example)
    for ndn_i, gxvar in gd["nonants"].items():
        try:   # not sure this is needed
            float(gxvar.get_level())
        except:
            raise RuntimeError(
                f"Non-anticipative variable {gxvar.name} on scenario {s.name} "
                "had no value. This usually means this variable "
                "did not appear in any (active) components, and hence "
                "was not communicated to the subproblem solver. ")
        if False: # needed?
            raise RuntimeError(
                f"Non-anticipative variable {gxvar.name} on scenario {s.name} "
                "was presolved out. This usually means this variable "
                "did not appear in any (active) components, and hence "
                "was not communicated to the subproblem solver. ")

        s._mpisppy_data.nonant_indices[ndn_i]._value = gxvar.get_level()
        if global_rank == 0:  # debugging
            print(f"solve_one: {s.name =}, {ndn_i =}, {gxvar.get_level() =}")

    print(f"   {objval =}")

    # the next line ignores bundling
    s._mpisppy_data._obj_from_agnostic = objval

    # TBD: deal with other aspects of bundling (see solve_one in spopt.py)


# local helper
def _copy_Ws_xbar_rho_from_host(s):
    # special for farmer
    # print(f"   debug copy_Ws {s.name =}, {global_rank =}")
    gd = s._agnostic_dict
    # could/should use set values
    for ndn_i, gxvar in gd["nonants"].items():
        if hasattr(s._mpisppy_model, "W"):
            gd["ph"]["ph_W"][ndn_i].set_value(s._mpisppy_model.W[ndn_i].value)
            gd["ph"]["rho"][ndn_i].set_value(s._mpisppy_model.rho[ndn_i].value)
            gd["ph"]["xbar"][ndn_i].set_value(s._mpisppy_model.xbars[ndn_i].value)
        else:
            # presumably an xhatter; we should check, I suppose
            pass


# local helper
def _copy_nonants_from_host(s):
    # values and fixedness; 
    gd = s._agnostic_dict
    for ndn_i, gxvar in gd["nonants"].items():
        hostVar = s._mpisppy_data.nonant_indices[ndn_i]
        guestVar = gd["nonants"][ndn_i]
        guestVar.set_level(hostVar._value)
        if hostVar.is_fixed():
            guestVar.set_lower(hostVar._value)
            guestVar.set_upper(hostVar._value)
        else:
            guestVar.set_lower(gd["ph"]["nonant_lbs"][ndn_i])
            guestVar.set_upper(gd["ph"]["nonant_ubs"][ndn_i])


def _restore_nonants(Ag, s):
    # the host has already restored
    _copy_nonants_from_host(s)

    
def _restore_original_fixedness(Ag, s):
    _copy_nonants_from_host(s)


def _fix_nonants(Ag, s):
    _copy_nonants_from_host(s)


def _fix_root_nonants(Ag, s):
    _copy_nonants_from_host(s)
