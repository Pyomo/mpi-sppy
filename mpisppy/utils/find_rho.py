# Copyright 2023 by U. Naepels and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Code to compute rhos from Ws. It also provides a corresponding rho setter.
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
"""mpisppy.log.setup_logger("mpisppy.utils.find_rho",
                         "findrho.log",
                         level=logging.CRITICAL)                         
logger = logging.getLogger("mpisppy.utils.find_rho")"""

############################################################################


class Find_Rho():
    """ Interface to compute rhos from Ws for a given ph object and write them in a file
    
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

        if cfg.rho_file == '' and cfg.grad_rho_file == '': 
            pass
        else:
            assert self.cfg.whatpath != '', "to compute rhos you have to give the name of a What csv file (using --whatpath)"
            if (not os.path.exists(self.cfg.whatpath)):
                raise RuntimeError('Could not find file {fn}'.format(fn=self.cfg.whatpath))
            with open(self.cfg.whatpath, 'r') as f:
                for line in f:
                    if (line.startswith('#')):
                        continue
                    line  = line.split(',')
                    cval = float(line[2][:-2])
                    self.c[(line[0], line[1])] = cval


    def _w_denom(self, s, node):
        """ Computes the denominator for w-based rho. This denominator is scenario dependant. 

        Args:
           s (Pyomo Concrete Model): scenario
           node: only ROOT for now

        Returns:
           w_denom (numpy array): denominator

        """
        assert node.name == "ROOT", "compute rho only works for two stage for now"
        nlen = s._mpisppy_data.nlens[node.name]
        xbar_array = np.array([s._mpisppy_model.xbars[(node.name,j)]._value for j in range(nlen)])
        nonants_array = np.fromiter((v._value for v in node.nonant_vardata_list),
                                    dtype='d', count=nlen)
        w_denom = np.abs(nonants_array - xbar_array)
        return w_denom


    def _prox_denom(self, s, node):
        """ Computes the denominator corresponding to the proximal term. This denominator is scenario dependant.

        Args:
           s (Pyomo Concrete Model): scenario
           node: only ROOT for now

        Returns:
           w_denom (numpy array): denominator

        """
        assert node.name == "ROOT", "compute rho only works for two stage for now"
        nlen = s._mpisppy_data.nlens[node.name]
        xbar_array = np.array([s._mpisppy_model.xbars[(node.name,j)]._value for j in range(nlen)])
        nonants_array = np.fromiter((v._value for v in node.nonant_vardata_list),
                                    dtype='d', count=nlen)
        prox_denom = 2 * np.square(nonants_array - xbar_array)
        return prox_denom


    def _grad_denom(self):
        """Computes the scenario independant denominator in the WW heuristic.

        Returns:
           g_denom (numpy array): denominator

        """
        phbase._Compute_Xbar(self.ph_object)
        sname, scenario = list(self.ph_object.local_scenarios.items())[0]
        for node in scenario._mpisppy_node_list:
            assert node.name == "ROOT", "compute rho only works for two stage for now"
        nlen0 = scenario._mpisppy_data.nlens["ROOT"]
        g_denom = np.zeros(nlen0, dtype='d')
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
            g_denom = np.maximum(np.ones(len(g_denom))/self.cfg.rho_relative_bound, g_denom)
            return g_denom
        

    def _order_stat(self, rho_list):
        """ Computes a scenario independant rho from a list of rhos.

        Args:
           rho_list (list): list of rhos

        Returns: 
           rho (float): rho value
        """
        alpha = self.cfg.order_stat
        assert alpha != -1.0, "you need to set the order statistic parameter for rho using --order-stat"
        assert (alpha >= 0 and alpha <= 1), "0 is the min, 0.5 the average, 1 the max"
        rho_mean, rho_min, rho_max = np.mean(rho_list), np.min(rho_list), np.max(rho_list)
        if alpha == 0.5:
            return rho_mean
        if alpha < 0.5:
            return (rho_min + alpha * 2 * (rho_mean - rho_min))
        if alpha > 0.5:
            return (2 * rho_mean - rho_max) + alpha * 2 * (rho_max - rho_mean)

    def compute_rho(self, indep_denom = False):
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
        cost = {k : np.array([self.c[k, vname] 
                for vname in all_vnames])
                for k in all_snames}
        if indep_denom:
            grad_denom = self._grad_denom()
            denom = {k: grad_denom for k in all_snames}
        else:
            loc_denom = {k: np.max((self._w_denom(s, node), self._prox_denom(s, node)))
                           for k, s in self.ph_object.local_scenarios.items()
                           for node in s._mpisppy_node_list}
            global_denom = self.ph_object.comms['ROOT'].gather(loc_denom, root=0)
            denom = dict()
            if self.ph_object.cylinder_rank == 0:
                for loc_denom in global_denom:
                    denom.update(loc_denom)
        if self.ph_object.cylinder_rank == 0:
            rho = dict()
            for k in all_snames:
                rho[k] = np.abs(np.divide(cost[k], denom[k]))
            arranged_rho = {vname: [rho_list[idx] for _, rho_list in rho.items()]
                            for vname, idx in vname_to_idx.items()}
            rho = {vname: self._order_stat(rho_list) for (vname, rho_list) in arranged_rho.items()}
            return rho


    def write_rho(self):
        """ Write the computed rhos in the file --rho-file.
        """
        if self.cfg.rho_file == '': pass
        else:
            rho_data = self.compute_rho()
            if self.ph_object.cylinder_rank == 0:
                with open(self.cfg.rho_file, 'w') as file:
                    writer = csv.writer(file)
                    writer.writerow(['#Rho values'])
                    for (vname, rho) in rho_data.items():
                        writer.writerow([vname, rho])


class Set_Rho():
    """ Interface to set the computed rhos in PH.

    Args:
       cfg (Config): config object

    """
    def __init__(self, cfg):
        self.cfg = cfg

    def rho_setter(self, scenario):
        """ rho setter to be used in the PH algorithm
        
        Args:
        scenario (Pyomo Concrete Model): scenario
        cfg (Config object): config object

        Returns:
        rho_list (list): list of (id(variable), rho)

        """
        assert self.cfg != None, "you have to give the rho_setter a cfg"
        assert self.cfg.rho_path != '', "use --rho-path to give the path of your rhos file"
        rhofile = self.cfg.rho_path
        rho_list = list()
        with open(rhofile) as infile:
            reader = csv.reader(infile)
            for row in reader:
                if (row[0].startswith('#')):
                    continue
                else:
                    fullname = row[0]
                    vo = scenario.find_component(fullname)
                    if vo is not None:
                        rho_list.append((id(vo), float(row[1])))
                    else:
                        raise RuntimeError(f"rho values from {filename} found Var {fullname} "
                                           f"that is not found in the scenario given (name={s._name})")
        return rho_list


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
    cfg.rho_args()

    return cfg


def get_rho_from_W(mname, original_cfg):
    """ Creates a ph object from cfg and using the module functions. Then computes rhos from the Ws.

    Args:
       mname (str): module name
       original_cfg (Config object): config object

    """
    if  (original_cfg.rho_file == ''): return

    try:
        model_module = importlib.import_module(mname)
    except:
        raise RuntimeError(f"Could not import module: {mname}")
    cfg = copy.deepcopy(original_cfg)
    cfg.max_iterations = 0 #we only need x0 here

    #create ph_object via vanilla           
    scenario_creator = model_module.scenario_creator
    scenario_denouement = model_module.scenario_denouement
    scen_names_creator_args = inspect.getargspec(model_module.scenario_names_creator).args #partition requires to do that
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
    # Compute rhos                                                                                                              
    Find_Rho(ph_object, cfg).rhos()


if __name__ == "__main__":
    print("call find_rho.get_rho_from_W(modulename, cfg) and use --whatpath --rho-file to compute and write rhos") 
    
    
