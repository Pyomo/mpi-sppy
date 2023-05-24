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

        if cfg.rho_file == '': pass
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


    def compute_rhos(self):
        """ Computes rhos for each scenario and each variable using the WW heuristic.
        
        Returns:
           rho (dict): dict {scenario name: list of rhos for this scenario}

        """
        all_vnames, all_snames = [], []
        cost = dict()
        rho = dict()
        for (sname, vname) in self.c.keys():
            if sname not in all_snames: all_snames.append(sname)
            if vname not in all_vnames:all_vnames.append(vname)
        for k in all_snames:
            cost[k] = np.array([self.c[k, vname] for vname in all_vnames])
        phbase._Compute_Xbar(self.ph_object)
        for k, s in self.ph_object.local_scenarios.items():
            for node in s._mpisppy_node_list:
                nlen = s._mpisppy_data.nlens[node.name]
                xbar_array = np.array([s._mpisppy_model.xbars[(node.name,j)]._value for j in range(nlen)])
                nonants_array = np.fromiter((v._value for v in node.nonant_vardata_list),
                                            dtype='d', count=nlen)
                rho[k] = np.abs(np.divide(cost[k], (nonants_array - xbar_array)))
        return rho

    def _pstat(self, rho_list):
        """ Computes a scenario independant rho from a list of rhos. It's the mean for now.

        Returns:
           rho (float): the rho value for a given list

        """
        return np.mean(rho_list)


    def rhos(self):
        """ Write the computed rhos in a file named using --rho-file.

        """
        if self.cfg.rho_file == '': pass
        else:
            k0, s0 = list(self.ph_object.local_scenarios.items())[0]
            vname_to_idx = {var.name : ndn_i[1] for ndn_i, var in s0._mpisppy_data.nonant_indices.items()}
            rhos = dict()
            global_rhos = dict()
            arranged_rhos = dict()
            local_rhos = self.compute_rhos()
            g_rhos = self.ph_object.comms['ROOT'].gather(local_rhos, root=0)
            if self.ph_object.cylinder_rank == 0:
                for l_rhos in g_rhos: 
                    global_rhos.update(l_rhos)
                for vname, idx in vname_to_idx.items():
                    arranged_rhos[vname] = [rho_list[idx] for _, rho_list in global_rhos.items()]
                for (vname, rho_list) in arranged_rhos.items():
                    rhos[vname] = self._pstat(rho_list)

                with open(self.cfg.rho_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['#Rho values'])
                    writer.writerow
                    for (vname, rho) in rhos.items():
                        writer.writerow([vname, rhos[vname]])


class Set_Rhos():
    """ Interface to set the computed rhos in PH.

    Args:
       cfg (Config): config object

    """
    def __init__(self, cfg):
        if not cfg.rho_setter: pass
        else:
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


def get_rhos_from_Ws(mname, original_cfg):
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
    Find_Rhos(ph_object, cfg).rhos()


if __name__ == "__main__":
    print("call find_rho.get_rhos_from_Ws(modulename, cfg) and use --whatpath --rho-file to compute and write rhos") 
    
    
