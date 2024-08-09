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
    """Mustn't take any argument. Is called in agnostic cylinders

    Returns:
        list of pairs (str, str): for each non-anticipative variable, the name of the support set must be given with the name of the parameter.
        If the set is a cartesian set, there should be no paranthesis when given
    """
    return [("i,j", "x")]


def stoch_param_name_pairs_creator():
    """
    Returns:
        list of pairs (str, str): for each stochastic parameter, the name of the support set must be given with the name of the variable.
        If the set is a cartesian set, there should be no paranthesis when given
    """
    return [("j", "b")]


def scenario_creator(scenario_name, new_file_name, nonants_name_pairs, cfg=None):
    """ Create a scenario for the (scalable) farmer example.
    
    Args:
        scenario_name (str):
            Name of the scenario to construct.
        new_file_name (str):
            the gms file in which is created the gams model with the ph_objective
        nonants_name_pairs (list of (str,str)): list of (non_ant_support_set_name, non_ant_variable_name)
        cfg: pyomo config
    """
    assert new_file_name is not None
    stoch_param_name_pairs = stoch_param_name_pairs_creator()


    ws = gams.GamsWorkspace(working_directory=this_dir, system_directory=gamspy_base_dir)
    
    ### Calling this function is required regardless of the model
    # This function creates a model instance not instantiated yet, and gathers in glist all the parameters and variables that need to be modifiable
    mi, job, glist, all_ph_parameters_dicts, xlo_dict, xup_dict, x_out_dict = gams_guest.pre_instantiation_for_PH(ws, new_file_name, nonants_name_pairs, stoch_param_name_pairs)

    opt = ws.add_options()
    opt.all_model_types = cfg.solver_name
    if LINEARIZED:
        mi.instantiate("transport using lp minimizing objective_ph", glist, opt)
    else:
        mi.instantiate("transport using qcp minimizing objective_ph", glist, opt)

    ### Calling this function is required regardless of the model
    # This functions initializes, by adding records (and values), all the parameters that appear due to PH
    nonant_set_sync_dict = gams_guest.adding_record_for_PH(nonants_name_pairs, cfg, all_ph_parameters_dicts, xlo_dict, xup_dict, x_out_dict, job)

    scennum = sputils.extract_num(scenario_name)

    count = 0
    b = mi.sync_db.get_parameter("b")
    j = job.out_db.get_set("j")
    for market in j:
        # In order to be able to easily verify whether the EF matches with what is obtained with PH, we don't generate "random" demands
        """np.random.seed(scennum * j.number_records + count)
        b.add_record(market.keys[0]).value = np.random.normal(1,cfg.cv) * job.out_db.get_parameter("b").find_record(market.keys[0]).value
        count += 1"""
        b.add_record(market.keys[0]).value = (1+2*(scennum-1)/10) * job.out_db.get_parameter("b").find_record(market.keys[0]).value

    return mi, nonants_name_pairs, nonant_set_sync_dict


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
                      domain=float,
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
