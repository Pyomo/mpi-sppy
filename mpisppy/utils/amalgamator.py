# Amalgamator.py starting point; DLW March 2021
# This software is distributed under the 3-clause BSD License.
# To test : python amalgamator.py 10 --solver-name=cplex --default-rho=1

"""Takes a scenario list and a scenario creator (and options)
as input and produces an outer bound on the objective function (solving the EF directly
or by decomposition). This same object should also be able to return an x-hat. 

This thing basically wraps the functionality of the "standard" *_cylinder examples.

Input options:
2-stage or multistage
EF or not
Assuming not EF:
  - list of cylinders
  - list of hub extensions

The object will then call the appropriate baseparsers functions to set up
the args and assemble the objects needed for spin-the-wheel, which it will call.

WARNING: When updating config and vanilla to add new cylinders/extensions,
you must keep up to date this file, especially the following dicts:
    - hubs_and_multi_compatibility
    - spokes_and_multi_compatibility
    - extensions_classes

"""
"""
Changes summer 2022: Each Amalgamagor object has a
Config object **** that it might modify (mainly add) *****
It no longer has options, just a cfg.
You might want to copy your cfg before passing it in.
"""
import numpy as np
import importlib
import inspect
import pyomo.environ as pyo
import argparse
import copy
import pyomo.common.config as pyofig
from mpisppy.utils import config
import mpisppy.utils.solver_spec as solver_spec

from mpisppy.utils.sputils import get_objs, nonant_cache_from_ef
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.sputils as sputils
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy import global_toc

from mpisppy.extensions.fixer import Fixer

hubs_and_multi_compatibility = {'ph': True,
                                'aph': True, 
                                #'lshaped':False, No parser = not incuded
                                #'cross_scen_hub':False, No parser = not included
                                }

spokes_and_multi_compatibility = {'fwph':False,
                                  'lagrangian':True,
                                  'lagranger':True,
                                  'xhatlooper':False,
                                  'xhatshuffle':True,
                                  'xhatspecific':True,
                                  'xhatlshaped':False,
                                  'slammax':False,
                                  'slammin':False,
                                  'cross_scenario_cuts':False}

default_unused_spokes = ['xhatlooper', 'xhatspecific']

extensions_classes = {'fixer':Fixer,
                      #NOTE: Before adding other extensions classes there, create:
                      #         - a function for it in config.py
                      #         - a function add_EXTNAME in vanila.py (or cfg_vanilla)
                      
                      }

#==========
# Utilities to interact with config

def _bool_option(cfg, oname):
    return oname in cfg and cfg[oname]


def add_options(cfg, parser_choice=None):
    #  parser_choice is a string referring to the component (e.g., "slammin")
    # (note: by "parser" we mean "config")
    assert parser_choice is not None

    parser_name = parser_choice+"_args"
    adder = getattr(cfg, parser_name)
    adder()


#==========
#Cylinder name checks

def find_hub(cylinders, is_multi=False):
    hubs = set(cylinders).intersection(set(hubs_and_multi_compatibility.keys()))
    if len(hubs) == 1:
        hub = list(hubs)[0]
        if is_multi and not hubs_and_multi_compatibility[hub]:
            raise RuntimeError(f"The hub {hub} does not work with multistage problems" )
    else:
        raise RuntimeError("There must be exactly one hub among cylinders")
    return hub


def find_spokes(cylinders, is_multi=False):
    spokes = []
    for c in cylinders:
        if not c in hubs_and_multi_compatibility:
            if c not in spokes_and_multi_compatibility:
                raise RuntimeError(f"The cylinder {c} do not exist or cannot be called via amalgamator.")
            if is_multi and not spokes_and_multi_compatibility[c]:
                raise RuntimeError(f"The spoke {c} does not work with multistage problems" )
            if c in default_unused_spokes:
                print(f"{c} is unused by default. Please specify --with-{c}=True in the command line to activate this spoke")
            spokes.append(c)
    return spokes

#==========
def check_module_ama(module):
    # Complain if the module lacks things needed.
    everything = ["scenario_names_creator",
                 "scenario_creator",
                 "inparser_adder",
                 "kw_creator"]  # start and denouement can be missing.
    you_can_have_it_all = True
    for ething in everything:
        if not hasattr(module, ething):
            print(f"Module {mname} is missing {ething}")
            you_can_have_it_all = False
    if not you_can_have_it_all:
        raise RuntimeError(f"Module {mname} not complete for from_module")


#==========
def from_module(mname, cfg, extraargs_fct=None, use_command_line=True):
    """ Try to get everything from one file (this will not always be possible).
    Args:
        mname (str): the module name (module must have certain functions)
                     or you can pass in a module that has already been imported
        cfg (Config): Amalgamator options or extra arguments to use 
                        in addition with the command line
        extraargs_fct (fct) : a function to add extra arguments, e.g. for MMW
        use_command_line (bool): should we take into account the command line to populate cfg ?
                                 default is True
                    
    Returns:
        ama (Amalgamator): the instantiated object
    
    """
    if not isinstance(cfg, config.Config):
        raise RuntimeError(f"amalgamator from_model bad cfg type={type(cfg)}; should be Config")

    if inspect.ismodule(mname):
        m = mname
    else:
        m = importlib.import_module(mname)
    check_module_ama(m)

    cfg = Amalgamator_parser(cfg, m.inparser_adder,
                                 extraargs_fct=extraargs_fct,
                                 use_command_line=use_command_line)
    cfg.add_and_assign('_mpisppy_probability', description="Uniform prob.", domain=float, default=None, value= 1/cfg['num_scens'])
    start = cfg['start'] if 'start' in cfg else 0
    sn = m.scenario_names_creator(cfg['num_scens'], start=start)
    dn = m.scenario_denouement if hasattr(m, "scenario_denouement") else None
    ama = Amalgamator(cfg,
                      sn,
                      m.scenario_creator,
                      m.kw_creator,
                      scenario_denouement=dn)
    return ama
                
        
#==========
def Amalgamator_parser(cfg, inparser_adder, extraargs_fct=None, use_command_line=True):
    """ Helper function for Amalgamator.
    Args:
        cfg (Config): Amalgamator control options, etc; might be added to or changed
        inparser_adder (fct): returns updated ArgumentParser the problem
        extraargs_fct (fct) : a function to add extra arguments, e.g. for MMW
        use_command_line (bool): should we take into account the command line to add options ?
                                 default is True
    Returns;
        cfg (Cofig): the modifed cfg object containing the options, both parsed values and pre-set options
    """

    # TBD: should we copy?
    
    if use_command_line:
        if _bool_option(cfg, "EF_2stage"):
            cfg.EF2()
        elif _bool_option(cfg, "EF_mstage"):
            cfg.EF_multistage()
        else:
            if _bool_option(cfg, "2stage"):
                cfg.popular_args()
            elif _bool_option(cfg, "mstage"):
                cfg.multistage()
            else:
                raise RuntimeError("The problem type (2stage or mstage) must be specified")
            cfg.two_sided_args()
            cfg.mip_options()
                
            #Adding cylinders
            if not "cylinders" in cfg:
                raise RuntimeError("A cylinder list must be specified")
            
            for cylinder in cfg['cylinders']:
                #NOTE: This returns an error if the cylinder yyyy has no yyyy_args in config.py
                add_options(cfg, cylinder)
            
            #Adding extensions
            if "extensions" in cfg:
                for extension in cfg['extensions']:
                    add_options(cfg, extension)
    
        inparser_adder(cfg)
        
        if extraargs_fct is not None:
            extraargs_fct()
        
        prg = cfg.get("program_name")
        cfg.parse_command_line(prg)

        """
        print("Amalgamator needs work for solver options!!")
        # TBD: deal with proliferation of solver options specifications
        if _bool_option(options_dict, "EF-2stage") or _bool_option(options_dict, "EF-mstage"): 
            if ('EF_solver_options' in options_dict):
                options_dict["EF_solver_options"]["mipgap"] = options_dict["EF_mipgap"]
            else:
                options_dict["EF_solver_options"] = {"mipgap": options_dict["EF_mipgap"]}
        """
    else:
        #Checking if cfg has all the options we need 
        if not (_bool_option(cfg, "EF_2stage") or _bool_option(cfg, "EF_mstage")):
            raise RuntimeError("For now, completly bypassing command line only works with EF." )
        if not ('EF_solver_name' in cfg):
            raise RuntimeError("EF_solver_name must be specified for the amalgamator." )
        if not ('num_scens' in cfg):
            raise RuntimeWarning("cfg should have a number of scenarios to compute a xhat")
        if _bool_option(cfg, 'EF-mstage') and 'branching_factors' not in cfg:
            raise RuntimeError("For a multistage problem, cfg must have a 'branching_factors' attribute with branching factors")

    return cfg
    

#========================================
class Amalgamator():
    """Takes a scenario list and a scenario creator (and options)
    as input. The ides is to produce an outer bound on the objective function (solving the EF directly
    or by decomposition) and/or an x-hat with inner bound; however, what it does is controlled by
    its constructor options and by user options.

    This thing basically wraps the functionality of the "standard" *_cylinder examples.
    
    It may be an extenisble base class, but not abstract.

    Args:
        cfg (Config): controls the amalgamation and may be added to or changed
        scenario_names (list of str) the full set of scenario names
        scenario_creator (fct): returns a concrete model with special things
        kw_creator (fct): takes an options dict and returns scenario_creator kwargs
        scenario_denouement (fct): (optional) called at conclusion
    """

    def __init__(self, cfg,
                 scenario_names, scenario_creator, kw_creator, 
                 scenario_denouement=None, verbose=True):
        self.cfg = cfg
        self.scenario_names = scenario_names
        self.scenario_creator = scenario_creator
        self.scenario_denouement = scenario_denouement
        self.kw_creator = kw_creator
        self.kwargs = self.kw_creator(self.cfg)
        self.verbose = verbose
        self.is_EF = _bool_option(cfg, "EF_2stage") or _bool_option(cfg, "EF_mstage")
        if self.is_EF:
            sroot, self.solver_name, self.solver_options = solver_spec.solver_specification(cfg, ["EF", ""])
        self.is_multi = _bool_option(cfg, "EF-mstage") or _bool_option(cfg, "mstage")
        if self.is_multi and not "all_nodenames" in cfg:
            if "branching_factors" in cfg:
                ndnms = sputils.create_nodenames_from_branching_factors(cfg["branching_factors"])
                self.cfg.quick_assign("all_nodenames", domain=pyofig.ListOf(str), value=ndnms)
            else:
                raise RuntimeError("For a multistage problem, please provide branching_factors or all_nodenames")
        
    def run(self):

        """ Top-level execution."""
        if self.is_EF:
            ef = sputils.create_EF(
                self.scenario_names,
                self.scenario_creator,
                scenario_creator_kwargs=self.kwargs,
                suppress_warnings=True,
            )

            tee_ef_solves = self.cfg.get('tee_ef_solves',False)
            
            solver_name = self.solver_name
            solver = pyo.SolverFactory(solver_name)
            if hasattr(self, "solver_options") and (self.solver_options is not None):
                for option_key,option_value in self.solver_options.items():
                    solver.options[option_key] = option_value
            if self.verbose :
                global_toc("Starting EF solve")
            if 'persistent' in solver_name:
                solver.set_instance(ef, symbolic_solver_labels=True)
                results = solver.solve(tee=tee_ef_solves)
            else:
                results = solver.solve(ef, tee=tee_ef_solves, symbolic_solver_labels=True,)
            if self.verbose:
                global_toc("Completed EF solve")

            
            self.EF_Obj = pyo.value(ef.EF_Obj)

            objs = sputils.get_objs(ef)
            
            self.is_minimizing = objs[0].is_minimizing
            #TBD : Write a function doing this
            if self.is_minimizing:
                self.best_outer_bound = results.Problem[0]['Lower bound']
                self.best_inner_bound = results.Problem[0]['Upper bound']
            else:
                self.best_inner_bound = results.Problem[0]['Upper bound']
                self.best_outer_bound = results.Problem[0]['Lower bound']
            self.ef = ef
            
            if 'write_solution' in self.cfg:
                if 'first_stage_solution' in self.cfg['write_solution']:
                    sputils.write_ef_first_stage_solution(self.ef,
                                                          self.cfg['write_solution']['first_stage_solution'])
                if 'tree_solution' in self.cfg['write_solution']:
                    sputils.write_ef_tree_solution(self.ef,
                                                   self.cfg['write_solution']['tree_solution'])
            
            self.xhats = sputils.nonant_cache_from_ef(ef)
            self.local_xhats = self.xhats  # Every scenario is local for EF
            self.first_stage_solution = {"ROOT": self.xhats["ROOT"]}

        else:
            self.ef = None

            #Create a hub dict
            hub_name = find_hub(self.cfg['cylinders'], self.is_multi)
            hub_creator = getattr(vanilla, hub_name+'_hub')
            beans = {"cfg": self.cfg,
                     "scenario_creator": self.scenario_creator,
                     "scenario_denouement": self.scenario_denouement,
                     "all_scenario_names": self.scenario_names,
                     "scenario_creator_kwargs": self.kwargs}
            if self.is_multi:
                beans["all_nodenames"] = self.cfg["all_nodenames"]
            hub_dict = hub_creator(**beans)
            
            #Add extensions
            if 'extensions' in self.cfg:
                for extension in self.cfg['extensions']:
                    extension_creator = getattr(vanilla, 'add_'+extension)
                    hub_dict = extension_creator(hub_dict, self.cfg)
            
            #Create spoke dicts
            potential_spokes = find_spokes(self.cfg['cylinders'],
                                           self.is_multi)
            #We only use the spokes with an associated command line arg set to True
            spokes = [spoke for spoke in potential_spokes if self.cfg[spoke]]
            list_of_spoke_dict = list()
            for spoke in spokes:
                spoke_creator = getattr(vanilla, spoke+'_spoke')
                spoke_beans = copy.deepcopy(beans)
                if spoke == "xhatspecific":
                    spoke_beans["scenario_dict"] = self.cfg["scenario_dict"]
                spoke_dict = spoke_creator(**spoke_beans)
                list_of_spoke_dict.append(spoke_dict)
                
            ws =  WheelSpinner(hub_dict, list_of_spoke_dict)
            ws.run()

            spcomm = ws.spcomm
            
            self.opt = spcomm.opt
            self.on_hub = ws.on_hub()
            
            if self.on_hub:  # we are on a hub rank
                self.best_inner_bound = spcomm.BestInnerBound
                self.best_outer_bound = spcomm.BestOuterBound
                #NOTE: We do not get bounds on every rank, only on hub
                #      This should change if we want to use cylinders for MMW
                
            # prior to June 2022, the wite options were a two-level dictionary; now flattened
            if 'first_stage_solution_csv' in self.cfg:
                ws.write_first_stage_solution(self.cfg['first_stage_solution_csv'])
            if 'tree_solution_csv' in self.cfg:
                ws.write_tree_solution(self.cfg['tree_solution_csv'])
            
            if self.on_hub: #we are on a hub rank
                a_sname = self.opt.local_scenario_names[0]
                root = self.opt.local_scenarios[a_sname]._mpisppy_node_list[0]
                self.first_stage_solution = {"ROOT":[pyo.value(var) for var in root.nonant_vardata_list]}
                self.local_xhats = ws.local_nonant_cache()
                
            #TODO: Add a xhats attribute, similar to the output of nonant_cache_from_ef
            #      It means doing a MPI operation over hub ranks


if __name__ == "__main__":
    # For use by developers doing ad hoc testing, our example is farmer
    import mpisppy.tests.examples.farmer as farmer
    # EF, PH, L-shaped, APH flags, and then boolean multi-stage
    """
    ama_options = {"2stage": True,   # 2stage vs. mstage
                   "cylinders": ['ph'],
                   "extensions": [],
                   "program_name": "amalgamator test main",
                   "num_scens_reqd": True,
                   }
    """
    cfg = config.Config()
    cfg.add_and_assign("2stage", description="2stage vsus mstage", domain=bool, default=None, value=True)
    cfg.add_and_assign("cylinders", description="list of cylinders", domain=pyofig.ListOf(str), default=None, value=["ph"])
    cfg.add_and_assign("extensions", description="list of extensions", domain=pyofig.ListOf(str), default=None, value= [])
    # num_scens_reqd has been deprecated

    ama = from_module("mpisppy.tests.examples.farmer", cfg)
    cfg.default_rho = 1.0
    ama.run()
    # print(f"inner bound=", ama.best_inner_bound)
    # print(f"outer bound=", ama.best_outer_bound)
    # print(sputils.nonant_cache_from_ef(ama.ef))
    # wish list: allow for integer relaxation using the Pyomo inplace transformation
    """ Issues:
    0. What do we want to put in ama_options and what do we want to discover
       by looking at data? I am inclined to put things in the options and maybe
       check against data. Here are the things:
       - 2-stage versus multi-stage [SOLVED: put it in ama_options]
       - MIP versus continuous only
       - linear/quadratic versus non-linear
    """
    # these options need to be tested
    ama_options = {"2stage": True,   # 2stage vs. mstage
                   "cylinders": ['ph','cross_scenario_cuts'],
                   "extensions": ['cross_scenario_cuts'],
                   "program_name": "amalgamator test main",
                   "num_scens_reqd": True,
                   }
