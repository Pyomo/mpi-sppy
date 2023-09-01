# <special for agnostic debugging DLW Aug 2023>
# In this example, AMPL is the guest language.

"""
Notes about generalization:
  - need a list of vars and indexes for nonants
"""

from amplpy import AMPL
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
    """ Create a scenario for the (scalable) farmer example, but
   but pretend that Pyomo is a guest language.
    
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

    ampl.read("farmer_test.ampl")

    # scenario specific data applied
    scennum = sputils.extract_num(scenario_name)
    assert scennum < 3, "three scenarios hardwired for now"
    if scennum == 0:
        xxxxx
    elif scenumm == 2:
        xxxxx

    areaVarDatas = list(ampl.get_variable("area").instances())

    # In general, be sure to process variables in the same order has the guest does (so indexes match)
    gd = {
        "scenario": ampl,
        "nonants": {("ROOT",i): v[1] for i,v in enumerate(areaVarDatas)},
        "nonant_fixedness": {("ROOT",i): v[1].astatus()=="fixed" for i,v in enumerate(areaVarDatas)},
        "nonant_start": {("ROOT",i): v[1].value() for i,v in enumerate(areaVarDatas)},
        "nonant_names": {("ROOT",i): ("area", v[0]) for i, v in enumerate(areaVarDatas)},
        "probability": "uniform",
        "sense": pyo.maximize,
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
    # Attach W's and rho to the guest scenario (mutable params).
    gs = scenario._agnostic_dict["scenario"]  # guest scenario handle
    gd = scenario._agnostic_dict
    # (there must be some way to create and assign *mutable* params in on call to AMPL)
    gs.eval("param W_on;")
    gs.eval("let W_on := 0;")
    gs.eval("param prox_on;")
    gs.eval("let prox_on := 0;")
    # we are trusing the order to match the nonant indexes
    gs.eval("param W{Crops};")
    # should use set_values instead of let
    gs.eval("let {c in Crops}  W[c] := 0;")
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
    gs.eval("param xbars{Crops} := 0;")

    # Dual term (weights W)
    objstr = str(gs.get_objective("profit"))
    phobjstr = ""
    if add_duals:
        phobjstr += " + W_on * sum{c in Crops} (W[c] * area[c])"
        print(phobjstr)
        
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
        phobjstr += " + prox_on * sum{c in Crops} (rho[c] * area[c] * area[c] "+\
                    " - 2.0 * xbars[c] - xbars[c] * xbars[c]^2)"

    objstr = objstr[:-1] + phobjstr + ";"
    objstr = objstr.replace("maximize profit", "maximize phobj")
    gs.eval(objstr)
    gs.export_model("export.mod")


def solve_one(Ag, s, solve_keyword_args, gripe, tee):
    # This needs to attach stuff to s (see solve_one in spopt.py)
    # Solve the guest language version, then copy values to the host scenario

    # This function needs to  W on the guest right before the solve

    # We need to operate on the guest scenario, not s; however, attach things to s (the host scenario)
    # and copy to s. If you are working on a new guest, you should not have to edit the s side of things

    # To acommdate the solve_one call from xhat_eval.py, we need to attach the obj fct value to s

    print("DO NOT LET AMPL PRESOLVE!!!!")
    
    _copy_Ws_from_host(s)
    gd = s._agnostic_dict
    gs = gd["scenario"]  # guest scenario handle

    solver_name = s._solver_plugin.name
    gs.set_option("solver", solver_name)    
    if 'persistent' in solver_name:
        raise RuntimeError("Persistent solvers are not currently supported in the farmer agnostic example.")

    solver_exception = None
    try:
        gs.solve()
    except Exception as e:
        results = None
        solver_exception = e

    if gs.solve_result != "solved":
        s._mpisppy_data.scenario_feasible = False

    if gripe:
        print (f"Solve failed for scenario {s.name} on rank {global_rank}")
        print(f"{gs.solve_result =}")
            
    if solver_exception is not None:
        raise solver_exception


    s._mpisppy_data.scenario_feasible = True
    # For AMPL mips, we need to use the gap option to compute bounds
    # https://amplmp.readthedocs.io/rst/features-guide.html
    # xxxxx TBD: does this work??? (what objective is active???)
    objval = gs.get_objective("profit").value()
    if gd["sense"] == pyo.minimize:
        s._mpisppy_data.outer_bound = objval
    else:
        s._mpisppy_data.outer_bound = objval

    # copy the nonant x values from gs to s so mpisppy can use them in s
    # in general, we need more checks (see the pyomo agnostic guest example)
    for ndn_i, gxvar in gd["nonants"].items():
        try:
            float(gxvar.value())
        except:
            raise RuntimeError(
                f"Non-anticipative variable {gxvar.name} on scenario {s.name} "
                "had not value. This usually means this variable "
                "did not appear in any (active) components, and hence "
                "was not communicated to the subproblem solver. ")

        s._mpisppy_data.nonant_indices[ndn_i]._value = gxvar.value()

    # the next line ignore bundling
    s._mpisppy_data._obj_from_agnostic = objval

    # TBD: deal with other aspects of bundling (see solve_one in spopt.py)


# local helper
def _copy_Ws_from_host(s):
    # special for farmer
    # print(f"   debug copy_Ws {s.name =}, {global_rank =}")
    gd = s._agnostic_dict
    gs = gd["scenario"]  # guest scenario handle
    # could/should use set values
    parm = gs.get_parameter("W")
    for ndn_i, gxvar in gd["nonants"].items():
        if hasattr(s._mpisppy_model, "W"):
            c = gd["nonant_names"][ndn_i][1]
            print(f"{c =}")
            parm.set(c, s._mpisppy_model.W[ndn_i].value)
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
    _copy_nonants_from_host(s)


def _fix_nonants(Ag, s):
    _copy_nonants_from_host(s)


def _fix_root_nonants(Ag, s):
    _copy_nonants_from_host(s)

    
    
