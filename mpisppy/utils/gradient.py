# Copyright 2023 by U. Naepels and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Code to compute gradient cost and rhos from the gradient. It also provides a corresponding rho setter.
# To test: /examples/farmer/farmer_rho_demo.py

import sys
import os
import inspect
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolutionStatus, TerminationCondition
import logging
import numpy as np
import math
import importlib
import csv
import inspect
import typing
import copy
import time

import mpisppy.log
from mpisppy import MPI
import mpisppy.utils.sputils as sputils
import mpisppy.spopt
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.utils.wxbarwriter import WXBarWriter
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.confidence_intervals.ciutils as ciutils
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import mpisppy.utils.wxbarutils as wxbarutils
import mpisppy.utils.rho_utils as rho_utils
import mpisppy.utils.find_rho as find_rho
import mpisppy.phbase as phbase


# Could also pass, e.g., sys.stdout instead of a filename
"""mpisppy.log.setup_logger("mpisppy.utils.find_grad",
                         "findgrad.log",
                         level=logging.CRITICAL)                         
logger = logging.getLogger("mpisppy.utils.find_grad")"""

############################################################################
class Find_Grad():
    """Interface to compute and write gradient cost
    
    Args:
       ph_object (PHBase): ph object
       cfg (Config): config object

    Attributes:
       c (dict): gradient cost

    """

    def __init__(self,
                 ph_object,
                 cfg):
        self.ph_object = ph_object
        self.cfg = cfg
        self.c = dict()

    #======================================================================

    def compute_grad(self, sname, scenario):
        """ Computes gradient cost for a given scenario
        
        Args:
           sname (str): the scenario name
           scenario (Pyomo Concrete Model): scenario
        
        Returns:
           grad_cost (dict): a dictionnary {nonant indice: gradient cost}

        """
        nlp = PyomoNLP(scenario)
        nlp_vars = nlp.get_pyomo_variables()
        grad = nlp.evaluate_grad_objective()
        grad_cost = {ndn_i: -grad[ndn_i[1]]
                     for ndn_i, var in scenario._mpisppy_data.nonant_indices.items()}
        return grad_cost
        

    def find_grad_cost(self):
        """ Computes gradient cost for all scenarios.
        
        ASSUMES:
           The cfg object should contain an xhat path corresponding to the xhat file.

        """
        if self.cfg.grad_cost_file == '': pass
        else:
            assert self.cfg.xhatpath != '', "to compute gradient cost, you have to give an xhat path using --xhatpath"
            
            self.ph_object.disable_W_and_prox()
            xhatfile = self.cfg.xhatpath
            xhat = ciutils.read_xhat(xhatfile)
            xhat_one = xhat["ROOT"]
            self.ph_object._save_nonants()
            self.ph_object._fix_nonants(xhat)
            self.ph_object.solve_loop()
            for (sname, scenario) in self.ph_object.local_scenarios.items():
                for node in scenario._mpisppy_node_list:
                    for v in node.nonant_vardata_list:
                        v.unfix()

            grad_cost ={sname: self.compute_grad(sname, scenario) 
                        for sname, scenario in self.ph_object.local_scenarios.items()} 
            local_costs = {(sname, var.name): grad_cost[sname][node.name, ix]
                           for (sname, scenario) in self.ph_object.local_scenarios.items()
                           for node in scenario._mpisppy_node_list
                           for (ix, var) in enumerate(node.nonant_vardata_list)}
            comm = self.ph_object.comms['ROOT']
            costs = comm.gather(local_costs, root=0)
            rank = self.ph_object.cylinder_rank
            if (self.ph_object.cylinder_rank == 0):
                self.c = {key: val 
                          for cost in costs
                          for key, val in cost.items()}
            comm.Barrier()
            self.ph_object._restore_nonants()
            self.ph_object.reenable_W_and_prox()


    def write_grad_cost(self):
        """ Writes gradient cost for all scenarios.

        ASSUMES: 
           The cfg object should contain an xhat path corresponding to the xhat file.

        """
        self.find_grad_cost()
        comm = self.ph_object.comms['ROOT']
        if (self.ph_object.cylinder_rank == 0):
            with open(self.cfg.grad_cost_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(['#grad cost values'])
                for (key, val) in self.c.items():
                    sname, vname = key[0], key[1]
                    writer.writerow([sname, vname, str(val)])
        comm.Barrier()


#====================================================================================

    def find_grad_rho(self):
        """Writes gradient cost for all variables.

        ASSUMES:
           The cfg object should contain a grad_cost_file.

        """
        assert self.cfg.grad_cost_file != '', "to compute rho you have to give the name of a csv file (using --grad-cost-file) where grad cost will be written"
        if (not os.path.exists(self.cfg.grad_cost_file)):
            raise RuntimeError('Could not find file {fn}'.format(fn=self.cfg.grad_cost_file))
        self.cfg.whatpath = self.cfg.grad_cost_file
        return find_rho.Find_Rho(self.ph_object, self.cfg).compute_rho()

    def write_grad_rho(self):
         """Writes gradient rho for all variables.

        ASSUMES:
           The cfg object should contain a grad_cost_file.

        """
         if self.cfg.grad_rho_file == '':
             pass
         else:
             rho_data = self.find_grad_rho()
             if self.ph_object.cylinder_rank == 0:
                 with open(self.cfg.grad_rho_file, 'a', newline='') as file:
                     writer = csv.writer(file)
                     writer.writerow(['#grad rho values'])
                     for (vname, rho) in rho_data.items():
                         writer.writerow([vname, rho_data[vname]])


###################################################################################


def _parser_setup():
    """ Set up config object and return it, but don't parse 

    Returns:
       cfg (Config): config object

    Notes:
       parsers for the non-model-specific arguments; but the model_module_name will be pulled off first

    """

    cfg = config.Config()
    cfg.add_branching_factors()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    
    cfg.gradient_args()

    return cfg


def grad_cost_and_rho(mname, original_cfg):
    """ Creates a ph object from cfg and using the module 'mname' functions. Then computes the corresponding grad cost and rho.

    Args:
       mname (str): module name
       original_cfg (Config object): config object

    """
    if  (original_cfg.grad_rho_file == '') and (original_cfg.grad_cost_file == ''): return

    try:
        model_module = importlib.import_module(mname)
    except:
        raise RuntimeError(f"Could not import module: {mname}")
    cfg = copy.deepcopy(original_cfg)
    cfg.max_iterations = 0 #we only need x0 here

    #create ph_object via vanilla           
    scenario_creator = model_module.scenario_creator
    scenario_denouement = model_module.scenario_denouement
    scen_names_creator_args = inspect.getfullargspec(model_module.scenario_names_creator).args #partition requires to do that
    if scen_names_creator_args[0] == 'cfg':
        all_scenario_names = model_module.scenario_names_creator(cfg)
    else :
        all_scenario_names = model_module.scenario_names_creator(cfg.num_scens)
    scenario_creator_kwargs = model_module.kw_creator(cfg)
    variable_probability = None
    if hasattr(model_module, '_variable_probability'):
        variable_probability = model_module._variable_probability
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)
    hub_dict = vanilla.ph_hub(*beans,
                              scenario_creator_kwargs=scenario_creator_kwargs,
                              ph_extensions=WXBarWriter,
                              variable_probability=variable_probability)
    list_of_spoke_dict = list()
    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin() #TODO: steal only what's needed in  WheelSpinner
    if wheel.strata_rank == 0:  # don't do this for bound ranks
        ph_object = wheel.spcomm.opt
    
    #============================================================================== 
    # Compute grad cost and rhos
    Find_Grad(ph_object, cfg).write_grad_cost()
    Find_Grad(ph_object, cfg).write_grad_rho()


if __name__ == "__main__":
    print("call gradient.grad_cost_and_rho(modulename, cfg) and use --xhatpath --grad-cost-file --grad-rho-file to compute and write gradient cost and rho") 
    
    
