# In this example, AMPL is the guest language.
# This is the python model file for AMPL farmer.
# It will work with farmer.mod and slight deviations.

from amplpy import AMPL
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import numpy as np

# If you need random numbers, use this random stream:
steelstream = np.random.RandomState()

# for debugging
from mpisppy import MPI
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

# the first two args are in every scenario_creator for an AMPL model
ampl = AMPL()

def scenario_creator(scenario_name, ampl_file_name, cfg=None):
    """"
    NOTE: for ampl, the names will be tuples name, index
    
    Returns:
        ampl_model (AMPL object): the AMPL model
        prob (float or "uniform"): the scenario probability
        nonant_var_data_list (list of AMPL variables): the nonants
        obj_fct (AMPL Objective function): the objective function
        tbd finish doc string
        need to add the args after completed
    """
 

    ampl = AMPL()
    ampl.read(ampl_file_name)
    ampl.read_data(cfg.ampl_data_file)
    scennum = sputils.extract_num(scenario_name)     
    seedoffset = cfg.seed
    #steelstream.seed(scennum+seedoffset)
    np.random.seed(scennum + seedoffset)
    
    # RANDOMIZE THE DATA******
    products = ampl.get_set("PROD")
    ppu = ampl.get_parameter("profit")
    if np.random.uniform()<cfg.steel_bandprob:
        ppu["bands"]/=2

    i = 0
    for product in products:
        np.random.seed(scennum + seedoffset + i*2**16)
        new_value = ppu[product] * max(0,np.random.normal(1,cfg.steel_cv))
        ppu[product] = new_value
        i += 1
    
    # the ampl vairable name is Make (which is too bad, but we will live with it)
    MakeVarDatas = list(ampl.get_variable("Make").instances())
    try:
        obj_fct = ampl.get_objective("minus_profit")
    except:
        print("big troubles!!; we can't find the objective function")
        print("doing export to _export.mod")
        gs.export_model("_export.mod")
        raise
    return ampl, "uniform", MakeVarDatas, obj_fct
    

def inparser_adder(cfg):
        cfg.num_scens_required()
    
        cfg.add_to_config("steel_cv",
            description="coefficient of variation for random profit(default 0.1)",
            domain=float,
            default=0.1) 
        cfg.add_to_config("steel_bandprob",
            description="probability of bands profit being at half price(default=0.5)",
            domain=float,
            default=0.5)
        cfg.add_to_config(name="ampl_data_file",
            description="The .d file needed if the language is AMPL",
            domain=str,
            default=None,
            argparse=True)
        
def kw_creator(cfg):
    kwargs = {"cfg": cfg}
    return kwargs

def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    return farmer.sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                                           given_scenario, **scenario_creator_kwargs)
def scenario_names_creator(num_scens,start=None):
    # (only for Amalgamator): return the full list of num_scens scenario names
    # if start!=None, the list starts with the 'start' labeled scenario
    if (start is None) :
        start=0
    return [f"scen{i}" for i in range(start,start+num_scens)]
def scenario_denouement(rank, scenario_name, scenario):
    pass