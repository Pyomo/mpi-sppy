###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Code to compute gradient cost and rhos from the gradient. It also provides a corresponding rho setter.
# To test: /examples/farmer/farmer_rho_demo.py

import inspect
import pyomo.environ as pyo
import importlib
import csv
import copy

from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.utils.wxbarwriter import WXBarWriter
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.confidence_intervals.ciutils as ciutils
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import mpisppy.utils.find_rho as find_rho


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
           grad_dict (dict): a dictionnary {nonant indice: -gradient}

        """

        # grab all discrete variables so they can be restored (out-of-place creation
        # via pyomo transformations is barfing.
        all_discrete_vars = []
        for var in scenario.component_objects(pyo.Var, active=True):
            for var_idx, var_data in var.items():
                if var_data.domain is pyo.Binary:
                    all_discrete_vars.append((var_data, pyo.Binary))
                elif var_data.domain is pyo.Integers:
                    all_discrete_vars.append((var_data, pyo.Integers))                    
        
        relax_int = pyo.TransformationFactory('core.relax_integer_vars')
        relax_int.apply_to(scenario)
        nlp = PyomoNLP(scenario)

        try:
            grad = nlp.evaluate_grad_objective()
        except Exception:
            raise RuntimeError("Cannot compute the gradient")
        grad = nlp.evaluate_grad_objective()
        grad_dict = {ndn_i: -grad[ndn_i[1]]
                     for ndn_i, var in scenario._mpisppy_data.nonant_indices.items()}

        for (var_data, var_domain) in all_discrete_vars:
            var_data.domain = var_domain

        return grad_dict
        

    def find_grad_cost(self):
        """ Computes gradient cost for all scenarios.
        
        ASSUMES:
           The cfg object should contain an xhat path corresponding to the xhat file.

        """
        assert self.cfg.xhatpath != '', "to compute gradient cost, you have to give an xhat path using --xhatpath"

        self.ph_object.disable_W_and_prox()
        xhatfile = self.cfg.xhatpath
        xhat = ciutils.read_xhat(xhatfile)
        self.ph_object._save_nonants()
        self.ph_object._fix_nonants(xhat)
        self.ph_object.solve_loop()
        for (sname, scenario) in self.ph_object.local_scenarios.items():
            for node in scenario._mpisppy_node_list:
                for v in node.nonant_vardata_list:
                    v.unfix()

        grad_dict = {sname: self.compute_grad(sname, scenario)
                     for sname, scenario in self.ph_object.local_scenarios.items()}
        local_costs = {(sname, var.name): grad_dict[sname][node.name, ix]
                       for (sname, scenario) in self.ph_object.local_scenarios.items()
                       for node in scenario._mpisppy_node_list
                       for (ix, var) in enumerate(node.nonant_vardata_list)}
        comm = self.ph_object.comms['ROOT']
        costs = comm.gather(local_costs, root=0)
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
        if (self.ph_object.cylinder_rank == 0):
            with open(self.cfg.grad_cost_file_out, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['#grad cost values'])
                for (key, val) in self.c.items():
                    sname, vname = key[0], key[1]
                    row = ','.join([sname, vname, str(val)]) + '\n'
                    f.write(row)

        # all ranks rely on the existence of this file.
        # barrier perhaps should be imposed at the caller level.
        comm = self.ph_object.comms['ROOT']                    
        comm.Barrier()

#====================================================================================

# TBD from JPW-  why are these two here at all? SHOULD NOT BE IN THE "GRADIENT" FILE
#   (partial answer, the ph_object is needed; but maybe the code could be untangled)

    def find_grad_rho(self):
        """computes rho based on cost file for all variables (does not write ????).

        ASSUMES:
           The cfg object should contain a grad_cost_file_in.

        """
        return find_rho.Find_Rho(self.ph_object, self.cfg).compute_rho()

    def write_grad_rho(self):
         """Writes gradient rho for all variables. (Does not compute it)

        ASSUMES:
           The cfg object should contain a grad_cost_file_in.

        """
         if self.cfg.grad_rho_file_out == '':
             raise RuntimeError("write_grad_rho without grad_rho_file_out")
         else:
             rho_data = self.find_grad_rho()
             if self.ph_object.cylinder_rank == 0:
                 with open(self.cfg.grad_rho_file_out, 'w', newline='') as file:
                     writer = csv.writer(file)
                     writer.writerow(['#grad rho values'])
                     for (vname, rho) in rho_data.items():
                         writer.writerow([vname, rho_data[vname]])

             # barrier added to avoid race-conditions involving other ranks reading.
             comm = self.ph_object.comms['ROOT']                    
             comm.Barrier()

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
    if  (original_cfg.grad_rho_file_out == '') and (original_cfg.grad_cost_file_out == ''):
        raise RuntimeError ("Presently, grad-rho-file-out and grad-cost-file cannot both be empty")

    try:
        model_module = importlib.import_module(mname)
    except Exception:
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
    #### not valid as of July 2024: print("call gradient.grad_cost_and_rho(modulename, cfg) and use --xhatpath --grad-cost-file --grad-rho-file to compute and write gradient cost and rho")
    print("no main")
    
    
