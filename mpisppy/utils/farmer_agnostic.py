# COPY !!!!!!!!!!!!!!!!!!!!! this is a copy.... but has probably been edited !!!! think!!!! meld with the original!!!!

# <special for agnostic debugging DLW Aug 2023>
# In this example, Pyomo is the guest language just for
# testing and documentation purposed.

import pyomo.environ as pyo
import farmer   # the native farmer

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
    """
    s = farmer.scenario_creator(scenario_name, use_integer, sense, crops_multiplier,
        num_scens, seedoffset)
    gd = {
        "scenario": s,
        "nonants": {("ROOT",i): v for i,v in enumerate(s.DevotedAcreage.values())},
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
    return farmer.kw_creator(cfg)

def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    return farmer.sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                                           given_scenario, **scenario_creator_kwargs)

#============================
def scenario_denouement(rank, scenario_name, scenario):
    farmer.scenario_denouement(rank, scenario_name, scenario)



##################################################################################################
# begin callouts
# NOTE: the callouts all take the Ag object as their first argument, mainly to see cfg if needed
# the function names correspond to function names in mpisppy

def attach_Ws_and_prox(Ag, sname, scenario):
    # this is farmer specific, so we know there is not a W already, e.g.
    print("guest Ws and prox")
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
    
    
def prox_disabled(Ag):
    return scenario._agnostic_dict["scenario"].prox_on._value == 0


def W_disabled(Ag):
    return scenario._agnostic_dict["scenario"].W_on._value == 0

    
def attach_PH_to_objective(Ag, sname, scenario, add_duals, add_prox):
    # Deal with prox linearization and approximation later,
    # i.e., just do the quadratic version

    # The host has xbars and computes without involving the guest language
    xbars = scenario._mpisppy_model.xbars

    gd = scenario._agnostic_dict
    gs = gd["scenario"]  # guest scenario handle
    nonant_idx = list(gd["nonants"].keys())    
    objfct = gs.Total_Cost_Objective  # we know this is farmer...
    ph_term = 0
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
                    (xvarsqrd - 2.0 * xbars[ndn_i] * xvar + xbars[ndn_i]**2)
            gs.ProxExpr = pyo.Expression(expr=prox_expr)
            ph_term += gs.prox_on * gs.ProxExpr
                    
            if gd["sense"] == pyo.minimize:
                objfct.expr += ph_term
            elif gd["sense"] == pyo.maximize:
                objfct.expr -= ph_term
            else:
                raise RuntimeError(f"Unknown sense {gd['sense'] =}")
            

def solve_one(Ag, s, solve_kw_args):
    # This needs to attach stuff to s (see solve_one in spopt.py)
    # xxxxxx how do you deal with the solver plugin? What about staleness?
    # Solve the guest language version, then copy values to the host scenario
    fdsalkjlakj
