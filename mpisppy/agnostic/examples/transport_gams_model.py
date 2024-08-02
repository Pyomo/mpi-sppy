LINEARIZED = True

import os
import gams
import gamspy_base

this_dir = os.path.dirname(os.path.abspath(__file__))
gamspy_base_dir = gamspy_base.__path__[0]

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils

# for debugging
from mpisppy import MPI
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

from mpisppy.agnostic import gams_guest
import numpy as np

def nonants_name_pairs_creator():
    return [("i,j", "x")]


def stoch_param_name_pairs_creator():
    return [("j", "b")]


def scenario_creator(scenario_name, new_file_name, nonants_name_pairs, cfg=None):
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
    assert new_file_name is not None
    stoch_param_name_pairs = stoch_param_name_pairs_creator()


    ws = gams.GamsWorkspace(working_directory=this_dir, system_directory=gamspy_base_dir)
    
    ### Calling this function is required regardless of the model
    # This function creates a model instance not instantiated yet, and gathers in glist all the parameters and variables that need to be modifiable
    mi, job, set_element_names_dict, stoch_sets_sync_dict, glist, all_ph_parameters_dicts, xlo_dict, xup_dict, x_out_dict = gams_guest.pre_instantiation_for_PH(ws, new_file_name, nonants_name_pairs, stoch_param_name_pairs)

    opt = ws.add_options()
    opt.all_model_types = cfg.solver_name
    if LINEARIZED:
        mi.instantiate("transport using lp minimizing objective_ph", glist, opt)
    else:
        mi.instantiate("transport using qcp minimizing objective_ph", glist, opt)

    ### Calling this function is required regardless of the model
    # This functions initializes, by adding records (and values), all the parameters that appear due to PH
    gams_guest.adding_record_for_PH(nonants_name_pairs, set_element_names_dict, cfg, all_ph_parameters_dicts, xlo_dict, xup_dict, x_out_dict)

    scennum = sputils.extract_num(scenario_name)

    count = 0
    b = mi.sync_db.get_parameter("b")
    #j = stoch_sets_sync_dict["b"]
    j = job.out_db.get_set("j")
    for market in j:
        np.random.seed(scennum * j.num_records + count)
        b.add_record(market).value = np.random.normal(1,cfg.cv) * job.out_db.get_parameter("b").find_record(market).value
        count += 1

    return mi, nonants_name_pairs, set_element_names_dict


#=========
def scenario_names_creator(num_scens,start=None):
    if (start is None) :
        start=0
    return [f"scen{i}" for i in range(start,start+num_scens)]


#=========
def inparser_adder(cfg):
    # add options unique to transport
    cfg.num_scens_required()
    cfg.add_to_config("cv",
                      description="covariance of the demand at the markets",
                      domain=int,
                      default=0.2)

    
#=========
def kw_creator(cfg):
    # creates keywords for scenario creator
    kwargs = {}
    kwargs["cfg"] = cfg
    return kwargs


#============================
def scenario_denouement(rank, scenario_name, scenario):
    # doesn't seem to be called
    pass
