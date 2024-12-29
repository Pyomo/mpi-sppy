###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import os
import gamspy_base
import mpisppy.utils.sputils as sputils
import mpisppy.agnostic.farmer4agnostic as farmer

from mpisppy import MPI # for debugging
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

LINEARIZED = True
this_dir = os.path.dirname(os.path.abspath(__file__))
gamspy_base_dir = gamspy_base.__path__[0]


def nonants_name_pairs_creator():
    """Mustn't take any argument. Is called in agnostic cylinders

    Returns:
        list of pairs (str, str): for each non-anticipative variable, the name of the support set must be given with the name of the variable.
        If the set is a cartesian set, there should be no paranthesis when given
    """
    return [("crop", "x")]


def stoch_param_name_pairs_creator():
    """
    Returns:
        list of pairs (str, str): for each stochastic parameter, the name of the support set must be given with the name of the parameter.
        If the set is a cartesian set, there should be no paranthesis when given
    """
    return [("crop", "yield")]


def scenario_creator(scenario_name, mi, job, cfg=None):
    """ Create a scenario for the (scalable) farmer example.
    
    Args:
        scenario_name (str):
            Name of the scenario to construct.
        mi (gams model instance): the base model
        job (gams job) : not used for farmer
        cfg: pyomo config
    """
    ### This part is model specific, we define the values of the stochastic parameters depending on scenario_name
    scennum = sputils.extract_num(scenario_name)
    assert scennum < 3, "three scenarios hardwired for now"
    y = mi.sync_db.get_parameter("yield")

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

    return mi

        
#=========
def scenario_names_creator(num_scens,start=None):
    return farmer.scenario_names_creator(num_scens,start)


#=========
def inparser_adder(cfg):
    farmer.inparser_adder(cfg)

    
#=========
def kw_creator(cfg):
    # creates keywords for scenario creator
    #kwargs = farmer.kw_creator(cfg)
    kwargs = {}
    kwargs["cfg"] = cfg
    #kwargs["nonants_name_pairs"] = nonants_name_pairs_creator()
    return kwargs


#============================
def scenario_denouement(rank, scenario_name, scenario):
    # doesn't seem to be called
    if global_rank == 1:
        x_dict = {}
        for x_record in scenario._agnostic_dict["scenario"].sync_db.get_variable('x'):
            x_dict[x_record.get_keys()[0]] = x_record.get_level()
        print(f"In {scenario_name}: {x_dict}")
    pass
    # (the fct in farmer won't work because the Var names don't match)
    #farmer.scenario_denouement(rank, scenario_name, scenario)
