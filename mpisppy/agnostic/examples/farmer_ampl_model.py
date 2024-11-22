###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# In this example, AMPL is the guest language.
# This is the python model file for AMPL farmer.
# It will work with farmer.mod and slight deviations.

from amplpy import AMPL
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import mpisppy.agnostic.examples.farmer as farmer
import numpy as np
from mpisppy import MPI  # for debugging
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

# If you need random numbers, use this random stream:
farmerstream = np.random.RandomState()


# the first two args are in every scenario_creator for an AMPL model
def scenario_creator(scenario_name, ampl_file_name,
        use_integer=False, sense=pyo.minimize, crops_multiplier=1,
        num_scens=None, seedoffset=0
):
    """ Create a scenario for the (scalable) farmer example
    
    Args:
        scenario_name (str):
            Name of the scenario to construct.
        ampl_file_name (str):
            The name of the ampl model file (with AMPL in it)
            (This adds flexibility that maybe we don't need; it could be hardwired)
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
    
    Returns:
        ampl_model (AMPL object): the AMPL model
        prob (float or "uniform"): the scenario probability
        nonant_var_data_list (list of AMPL variables): the nonants
        obj_fct (AMPL Objective function): the objective function
    """

    assert crops_multiplier == 1, "for AMPL, just getting started with 3 crops"

    ampl = AMPL()

    ampl.read(ampl_file_name)

    # scenario specific data applied
    scennum = sputils.extract_num(scenario_name)
    assert scennum < 3, "three scenarios hardwired for now"
    y = ampl.get_parameter("RandomYield")
    if scennum == 0:  # below
        y.set_values({"wheat": 2.0, "corn": 2.4, "beets": 16.0})
    elif scennum == 2: # above
        y.set_values({"wheat": 3.0, "corn": 3.6, "beets": 24.0})

    areaVarDatas = list(ampl.get_variable("area").instances())

    try:
        obj_fct = ampl.get_objective("minus_profit")
    except:
        print("big troubles!!; we can't find the objective function")
        raise
    return ampl, "uniform", areaVarDatas, obj_fct
    
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
