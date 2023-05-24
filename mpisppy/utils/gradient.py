# Copyright 2023 by U. Naepels and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Code to compute gradient cost and rhos from the gradient. It also provides a corresponding rho setter.
# To test: /examples/farmer/farmer_demo.py

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

    """


    def __init__(self,
                 ph_object,
                 cfg):
        self.ph_object = ph_object
        self.cfg = cfg

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
        grad_cost = {}
        for ndn_i, var in scenario._mpisppy_data.nonant_indices.items():
            grad_cost[ndn_i] = -grad[ndn_i[1]]
        return grad_cost
        

    def scenario_loop(self):
        """ Computes gradient cost for all scenarios and write them in a file.
        
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

            grad_cost ={}
            for sname, scenario in self.ph_object.local_scenarios.items():
                grad_cost[sname] = self.compute_grad(sname, scenario)

            local_costs = {(sname, var.name): grad_cost[sname][node.name, ix]
                           for (sname, scenario) in self.ph_object.local_scenarios.items()
                           for node in scenario._mpisppy_node_list
                           for (ix, var) in enumerate(node.nonant_vardata_list)}
            comm = self.ph_object.comms['ROOT']
            costs = comm.gather(local_costs, root=0)
            rank = self.ph_object.cylinder_rank
            if (self.ph_object.cylinder_rank == 0):
                with open(self.cfg.grad_cost_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(['#grad cost values'])
                    for cost in costs:
                        for (key, val) in cost.items():
                            sname, vname = key[0], key[1]
                            writer.writerow([sname, vname, str(val)])
            
            comm.Barrier()
            self.ph_object._restore_nonants()
            self.ph_object.reenable_W_and_prox()

###################################################################################

class Find_Rhos():
    """ Interface to compute rhos for a given ph object and write them in a file
    
    Args:
       ph_object (PHBase): ph object
       cfg (Config): config object

    Attributes:
       c (dict): a dictionnary {(scenario name, nonant indice): c value} 
       corresponding to the cost vector in the PH algorithm

    """

    def __init__(self, ph_object, cfg):
        self.ph_object = ph_object
        self.cfg = cfg
        self.c = dict()

        if cfg.grad_rho_file == '': pass
        else:
            assert self.cfg.grad_cost_file != '', "to compute rhos you have to give the name of a csv file (using --grad-cost-file) where grad cost will be written"
            if (not os.path.exists(self.cfg.grad_cost_file)):
                raise RuntimeError('Could not find file {fn}'.format(fn=self.cfg.grad_cost_file))
            with open(self.cfg.grad_cost_file, 'r') as f:
                for line in f:
                    if (line.startswith('#')):
                        continue
                    line  = line.split(',')
                    cval = float(line[2][:-2])
                    self.c[(line[0], line[1])] = cval


    def _global_denom(self):
        """ Computes the rho formula denominator in the WW heuristic.

        Returns:
            g_denom (array): np array containing the denominator for each variable

        """
        phbase._Compute_Xbar(self.ph_object)
        sname, scenario = list(self.ph_object.local_scenarios.items())[0]
        for node in scenario._mpisppy_node_list:
            assert node.name == "ROOT", "compute rho only works for two stage for now"
        g_denom = np.zeros(3, dtype='d')
        nlen0 = scenario._mpisppy_data.nlens["ROOT"]
        xbar_array = np.array([scenario._mpisppy_model.xbars[("ROOT",j)]._value for j in range(nlen0)])
        denom = 0
        for k,s in self.ph_object.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            for node in s._mpisppy_node_list:
                ndn = node.name
                nlen = nlens[ndn]
                nonants_array = np.fromiter((v._value for v in node.nonant_vardata_list),
                                        dtype='d', count=nlen)
                if not s._mpisppy_data.has_variable_probability:
                    denom += s._mpisppy_data.prob_coeff[ndn] * abs(nonants_array - xbar_array)
                else:
                    # rarely-used overwrite in the event of variable probability (not efficient for multi-stage) 
                    prob_array = np.fromiter((s._mpisppy_data.prob_coeff[ndn_i[0]][ndn_i[1]]
                                              for ndn_i in s._mpisppy_data.nonant_indices if ndn_i[0] == ndn),
                                             dtype='d', count=nlen)
                    # Note: Intermediate scen_contribution to get proper overloading
                    scen_contribution = prob_array * abs(nonants_array - xbar_array)
                    denom += scen_contribution
        self.ph_object.comms["ROOT"].Allreduce([denom, MPI.DOUBLE],
                                               [g_denom, MPI.DOUBLE],
                                               op=MPI.SUM)

        if self.ph_object.cylinder_rank == 0:
            #g_denom = np.maximum(np.ones(len(g_denom)), g_denom)
            g_denom = np.maximum(np.ones(len(g_denom))*1e-8, g_denom)
            return g_denom


    def compute_rhos(self):
        """ Computes rhos for each scenario and each variable using the WW heuristic.
        
        Returns:
           arranged_rho (dict): dict {variable name: list of rhos for this variable}

        """
        all_vnames, all_snames = [], []
        for (sname, vname) in self.c.keys():
            if sname not in all_snames: all_snames.append(sname)
            if vname not in all_vnames:all_vnames.append(vname)
        k0, s0 = list(self.ph_object.local_scenarios.items())[0]
        vname_to_idx = {var.name : ndn_i[1] for ndn_i, var in s0._mpisppy_data.nonant_indices.items()}

        denom = self._global_denom()
        if self.ph_object.cylinder_rank == 0:
            cost = dict()
            for k in all_snames:
                cost[k] = np.array([self.c[k, vname] for vname in all_vnames])
            rho = dict()
            for k in all_snames:
                rho[k] = np.abs(np.divide(cost[k], denom))

            arranged_rho = dict()
            for vname, idx in vname_to_idx.items():
                arranged_rho[vname] = [rho_list[idx] for _, rho_list in rho.items()]
            return arranged_rho
                

    def _pstat(self, rho_list):
        """ Computes a scenario independant rho from a list of rhos. It's the mean for now.

        Returns:
           rho (float): the rho value for a given list

        """
        return np.mean(rho_list)


    def rhos(self):
        """ Write the computed rhos in the file --grad-rho-file.

        """
        if self.cfg.grad_rho_file == '': pass
        else:
            rhos = dict()
            rho_data = self.compute_rhos()
            if self.ph_object.cylinder_rank == 0:
                for (vname, rho_list) in rho_data.items():
                    rhos[vname] = self._pstat(rho_list)

                with open(self.cfg.grad_rho_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['#Rho values'])
                    for (vname, rho) in rhos.items():
                        writer.writerow([vname, rhos[vname]])

    #======================================================================   

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
    Find_Grad(ph_object, cfg).scenario_loop()
    Find_Rhos(ph_object, cfg).rhos()


if __name__ == "__main__":
    print("call find_grad.grad_cost_and_rho(modulename, cfg) and use --xhatpath --grad-cost-file --grad-rho-file to compute and write gradient cost and rho") 
    
    
