###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Code to compute rhos from Ws. It also provides a corresponding rho setter.
# To test: /examples/farmer/farmer_demo.py

import os
import inspect
import numpy as np
import importlib
import csv
import copy
from sortedcollections import OrderedSet

from mpisppy import MPI
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.utils.wxbarwriter import WXBarWriter
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.phbase as phbase



# Could also pass, e.g., sys.stdout instead of a filename
"""mpisppy.log.setup_logger("mpisppy.utils.find_rho",
                         "findrho.log",
                         level=logging.CRITICAL)
logger = logging.getLogger("mpisppy.utils.find_rho")"""

############################################################################


class Find_Rho():
    """ Interface to compute rhos from Ws for a given ph object and write them in a file
    DLW July 2024 ? Is it from W's or from costs?

    Args:
       ph_object (PHBase): ph object
       cfg (Config): config object

    Attributes:
       c (dict): a dictionary {(scenario name, nonant indice): c value}
       corresponding to the cost vector in the PH algorithm

    """

    def __init__(self, ph_object, cfg):
        self.ph_object = ph_object
        self.cfg = cfg
        self.c = dict()

        if cfg.get("grad_cost_file_in", ifmissing='')  == '': 
            raise RuntimeError("Find_Rho constructor called without grad_cost_file_in")
        else:
            if (not os.path.exists(self.cfg.grad_cost_file_in)):
                raise RuntimeError(f'Could not find file {self.cfg.grad_cost_file_in}')
            with open(self.cfg.grad_cost_file_in, 'r') as f:
                for line in f:
                    if (line.startswith('#')):
                        continue
                    line  = line.split(',')
                    sname = line[0]
                    vname = ','.join(line[1:-1])
                    cval  = float(line[-1])
                    self.c[(sname, vname)] = cval


    def _w_denom(self, s, node):
        """ Computes the denominator for w-based rho. This denominator is scenario dependant.

        Args:
           s (Pyomo Concrete Model): scenario
           node: only ROOT for now

        Returns:
           w_denom (numpy array): denominator

        """
        assert node.name == "ROOT", "gradient-based compute rho only works for two stage for now"
        nlen = s._mpisppy_data.nlens[node.name]
        xbar_array = np.array([s._mpisppy_model.xbars[(node.name,j)]._value for j in range(nlen)])
        nonants_array = np.fromiter((v._value for v in node.nonant_vardata_list),
                                    dtype='d', count=nlen)
        w_denom = np.abs(nonants_array - xbar_array)
        denom_max = np.max(w_denom)
        for i in range(len(w_denom)):
            if w_denom[i] <= self.ph_object.E1_tolerance:
                w_denom[i] = max(denom_max, self.ph_object.E1_tolerance)
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
        # Need a comment indicating why this isn't already computed/available
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
                probs = s._mpisppy_data.prob_coeff[ndn] * np.ones(nlen)
                # using 1 as a default value because the variable values are converged if the diff is 0, so ...
                denom += probs * np.maximum(np.abs(nonants_array - xbar_array), 1.0)
                print("{denom=}")
        self.ph_object.comms["ROOT"].Allreduce([denom, MPI.DOUBLE],
                                               [g_denom, MPI.DOUBLE],
                                               op=MPI.SUM)
        if self.ph_object.cylinder_rank == 0:
            g_denom = np.maximum(np.ones(len(g_denom))/self.cfg.grad_rho_relative_bound, g_denom)
            return g_denom


    def compute_rho(self, indep_denom=False):
        """ Computes rhos for each scenario and each variable using the WW heuristic
        and first order condition.

        Returns:
           rhos (dict): dict {variable name: list of rhos for this variable}
        """
        local_snames = self.ph_object.local_scenario_names
        vnames = OrderedSet(vname for (sname, vname) in self.c.keys())
        k0, s0 = list(self.ph_object.local_scenarios.items())[0]
        # TBD: drop vname_to_idx and use the map already provided
        vname_to_idx = {var.name : ndn_i[1] for ndn_i, var in s0._mpisppy_data.nonant_indices.items()}
        cost = {k : np.array([self.c[k, vname]
                for vname in vnames])
                for k in local_snames}
        if indep_denom:
            grad_denom = self._grad_denom()
            loc_denom = {k: grad_denom for k in local_snames}
        else:
            # two-stage only for now
            loc_denom = {k: self._w_denom(s, s._mpisppy_node_list[0])
                           for k, s in self.ph_object.local_scenarios.items()}
        prob_list = [s._mpisppy_data.prob_coeff["ROOT"]
                     for s in self.ph_object.local_scenarios.values()]
        w = dict()
        for k, scenario in self.ph_object.local_scenarios.items():
            w[k] = np.array([scenario._mpisppy_model.W[ndn_i]._value
                             for ndn_i in scenario._mpisppy_data.nonant_indices])

        rho = {k : np.abs(np.divide(cost[k] - w[k], loc_denom[k])) for k in local_snames}                

        local_rhos = {vname: [rho_list[idx] for _, rho_list in rho.items()]
                        for vname, idx in vname_to_idx.items()}

        # Compute a scenario independant rho from a list of rhos using a triangular distribution.
        alpha = self.cfg.grad_order_stat
        assert (alpha >= 0 and alpha <= 1), f"For grad_order_stat 0 is the min, 0.5 the average, 1 the max; {alpha=} is invalid."
        vcnt = len(local_rhos)  # variable count
        rho_mins = np.empty(vcnt, dtype='d')  # global
        rho_maxes = np.empty(vcnt, dtype='d')
        rho_means = np.empty(vcnt, dtype='d')

        local_rho_mins = np.fromiter((min(rho_vals) for rho_vals in local_rhos.values()), dtype='d')
        local_rho_maxes = np.fromiter((max(rho_vals) for rho_vals in local_rhos.values()), dtype='d')
        local_prob = np.sum(prob_list)
        local_wgted_means = np.fromiter((np.dot(rho_vals, prob_list) * local_prob for rho_vals in local_rhos.values()), dtype='d')

        self.ph_object.comms["ROOT"].Allreduce([local_rho_mins, MPI.DOUBLE],
                                               [rho_mins, MPI.DOUBLE],
                                               op=MPI.MIN)
        self.ph_object.comms["ROOT"].Allreduce([local_rho_maxes, MPI.DOUBLE],
                                               [rho_maxes, MPI.DOUBLE],
                                               op=MPI.MAX)

        self.ph_object.comms["ROOT"].Allreduce([local_wgted_means, MPI.DOUBLE],
                                               [rho_means, MPI.DOUBLE],
                                               op=MPI.SUM)
        if alpha == 0.5:
            rhos = {vname: float(rho_mean) for vname, rho_mean in zip(local_rhos.keys(), rho_means)}
        elif alpha == 0.0:
            rhos = {vname: float(rho_min) for vname, rho_min in zip(local_rhos.keys(), rho_mins)}
        elif alpha == 1.0:
            rhos = {vname: float(rho_max) for vname, rho_max in zip(local_rhos.keys(), rho_maxes)}
        elif alpha < 0.5:
            rhos = {vname: float(rho_min + alpha * 2 * (rho_mean - rho_min))\
                    for vname, rho_min, rho_mean in zip(local_rhos.keys(), rho_mins, rho_means)}
        elif alpha > 0.5:
            rhos = {vname: float(2 * rho_mean - rho_max) + alpha * 2 * (rho_max - rho_mean)\
                    for vname, rho_mean, rho_max in zip(local_rhos.keys(), rho_means, rho_maxes)}
        else:
            raise RuntimeError("Coding error.")

        return rhos


    def write_grad_rho(self):
        """ Write the computed rhos in the file --grad-rho-file-out.
            Note: this file was originally opened for append access
        """
        if self.cfg.grad_rho_file_out == '':
            raise RuntimeError("write_grad_rho called without grad_rho_file_out")
        else:
            rho_data = self.compute_rho()
            if self.ph_object.cylinder_rank == 0:
                with open(self.cfg.grad_rho_file_out, 'w') as file:
                    writer = csv.writer(file)
                    writer.writerow(['#Rho values'])
                    for (vname, rho) in rho_data.items():
                        writer.writerow([vname, rho])

        comm = self.ph_object.comms['ROOT']                    
        comm.Barrier()                        


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
        assert self.cfg is not None, "you have to give the rho_setter a cfg"
        assert self.cfg.rho_file_in != '', "use --rho-file-in to give the path of your rhos file"
        rhofile = self.cfg.rho_file_in
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
                        raise RuntimeError(f"rho values from {rhofile} found Var {fullname} "
                                           f"that is not found in the scenario given (name={scenario._name})")
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

    return cfg


def get_rho_from_W(mname, original_cfg):
    """ Creates a ph object from cfg and using the module functions. Then computes rhos from the Ws.

    Args:
       mname (str): module name
       original_cfg (Config object): config object
    NOTE: as of July 2024, this function is never called and would not work due
          to the call to Find_Rho.rhos() (and perhaps other things)

    """
    assert original_cfg.grad_rho_file != ''

    try:
        model_module = importlib.import_module(mname)
    except Exception:
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
    print("call find_rho.get_rho_from_W(modulename, cfg) and use --grad_cost_file_in --rho-file to compute and write rhos")
