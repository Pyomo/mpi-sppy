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
from mpisppy import MPI  # for debugging
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

LINEARIZED = True
this_dir = os.path.dirname(os.path.abspath(__file__))
gamspy_base_dir = gamspy_base.__path__[0]


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


def scenario_creator(scenario_name, mi, job, cfg=None):
    """ Create a scenario for the (scalable) farmer example.
    
    Args:
        scenario_name (str):
            Name of the scenario to construct.
        mi (gams model instance): the base model
        job (gams job)
        cfg: pyomo config
    """
    scennum = sputils.extract_num(scenario_name)

    #count = 0
    b = mi.sync_db.get_parameter("b")
    j = job.out_db.get_set("j")
    for market in j:
        # In order to be able to easily verify whether the EF matches with what is obtained with PH, we don't generate "random" demands
        """np.random.seed(scennum * j.number_records + count)
        b.add_record(market.keys[0]).value = np.random.normal(1,cfg.cv) * job.out_db.get_parameter("b").find_record(market.keys[0]).value
        count += 1"""
        b.add_record(market.keys[0]).value = (1+2*(scennum-1)/10) * job.out_db.get_parameter("b").find_record(market.keys[0]).value

    return mi


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
