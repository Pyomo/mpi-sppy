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
