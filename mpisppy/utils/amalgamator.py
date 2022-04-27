# Amalgamator.py starting point; DLW March 2021
# This software is distributed under the 3-clause BSD License.
# To test : python amalgamator.py --num-scens=10

"""Takes a scenario list and a scenario creator (and options)
as input and produces an outer bound on the objective function (solving the EF directly
or by decomposition). This same object should also be able to return an x-hat. 

This thing basically wraps the functionality of the "standard" *_cylinder examples.

It may be an extenisble base class, but not abstract.

"""

"""
Input options:
2-stage or multistage
EF or not
Assuming not EF:
  - list of cylinders
  - list of hub extensions

The object will then call the appropriate baseparsers functions to set up
the args and assemble the objects needed for spin-the-wheel, which it will call.

WARNING: When updating baseparsers and vanilla to add new cylinders/extensions,
you must keep up to date this file, especially the following dicts:
    - hubs_and_multi_compatibility
    - spokes_and_multi_compatibility
    - extensions_classes

"""
import numpy as np
import importlib
import inspect
import pyomo.environ as pyo
import argparse
import copy

from mpisppy.utils.sputils import get_objs, nonant_cache_from_ef
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils import config
import mpisppy.utils.sputils as sputils
from mpisppy.utils import vanilla
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
                                  'slamup':False,
                                  'slamdown':False,
                                  'cross_scenario_cuts':False}

default_unused_spokes = ['xhatlooper', 'xhatspecific']

extensions_classes = {'fixer':Fixer,
                      #NOTE: Before adding other extensions classes there, create:
                      #         - a parser for it in baseparsers.py
                      #         - a function add_EXTNAME in vanila.py
                      
                      }

#==========
# Utilities to interact with config

def _bool_option(options, oname):
    return oname in options and options[oname]


def add_options(parser_choice=None):
    #  parser_choice is a string referring to the component (e.g., "slamdown")
    assert parser_choice is not None

    parser_name = parser_choice+"_args"
    adder = getattr(config, parser_name)
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
def from_module(mname, options, extraargs=None, use_command_line=True):
    """ Try to get everything from one file (this will not always be possible).
    Args:
        mname (str): the module name (module must have certain functions)
                     or you can pass in a module that has already been imported
        options (dict): Amalgamator options or extra arguments to use 
                        in addition with the command line
        extraargs (ArgumentParser) : extra arguments to specify in the command line, e.g. for MMW
        use_command_line (bool): should we take into account the command line to add options ?
                                 default is True
                    
    Returns:
        ama (Amalgamator): the instantiated object
    
    """
    assert(options is not None)

    if inspect.ismodule(mname):
        m = mname
    else:
        m = importlib.import_module(mname)
    check_module_ama(m)
    
    options = Amalgamator_parser(options, m.inparser_adder,
                                 extraargs=extraargs,
                                 use_command_line=use_command_line)
    options['_mpisppy_probability'] = 1/options['num_scens']
    start = options['start'] if 'start' in options else 0
    sn = m.scenario_names_creator(options['num_scens'], start=start)
    dn = m.scenario_denouement if hasattr(m, "scenario_denouement") else None
    ama = Amalgamator(options,
                      sn,
                      m.scenario_creator,
                      m.kw_creator,
                      scenario_denouement=dn)
    return ama
                
        
#==========
def Amalgamator_parser(options, inparser_adder, extraargs=None, use_command_line=True):
    """ Helper function for Amalgamator.  This gives us flexibility (e.g., get scen count)
    Args:
        options (dict): Amalgamator control options
        inparser_adder (fct): returns updated ArgumentParser the problem
        extraargs (ArgumentParser) : extra arguments to specify in the command line, e.g. for MMW
        use_command_line (bool): should we take into account the command line to add options ?
                                 default is True
    Returns;
        options (dict): a dict containing the options, both parsed values and pre-set options
    """

    options_dict = {**options}
    cfg = config.global_config
    
    if use_command_line:
        if _bool_option(options, "EF-2stage"):
            config.EF2()
        elif _bool_option(options, "EF-mstage"):
            config.EF_multistage()
        else:
            if _bool_option(options, "2stage"):
                config.popular_args()
            elif _bool_option(options, "mstage"):
                config.multistage()
            else:
                raise RuntimeError("The problem type (2stage or mstage) must be specified")
            config.two_sided_args()
            config.mip_options()
                
            #Adding cylinders
            if not "cylinders" in options:
                raise RuntimeError("A cylinder list must be specified")
            
            for cylinder in options['cylinders']:
                #NOTE: This returns an error if the cylinder xxxx has not xxxx_args in config.py
                parser = add_options(cylinder)
            
            #Adding extensions
            if "extensions" in options:
                for extension in options['extensions']:
                    parser = add_options(extension)
    
        inparser_adder()
        
        if extraargs is not None:
            raise RuntimeError("TBD: add extraargs support to amalgamator")
            ##parser = argparse.ArgumentParser(parents = [parser,extraargs],
            ##                                 conflict_handler='resolve')
        
        prg = options.get("program_name")
        parser = config.create_parser(prg)
        if _bool_option(options, "num_scens_reqd"):
            parser.add_argument("num_scens", help="Number of scenarios", type=int)

        #Changes made via the command line overwrite what is in options             
        ### options_dict.update(vars(cfg)) ???
        xxxx
        if _bool_option(options, "EF-2stage") or _bool_option(options, "EF-mstage"): 
            if ('EF_solver_options' in opt):
                opt["EF_solver_options"]["mipgap"] = opt["EF_mipgap"]
            else:
                opt["EF_solver_options"] = {"mipgap": opt["EF_mipgap"]}
    
    else:
        #Checking if options has all the options we need 
        if not (_bool_option(options, "EF-2stage") or _bool_option(options, "EF-mstage")):
            raise RuntimeError("For now, completly bypassing command line only works with EF." )
        if not ('EF_solver_name' in opt):
            raise RuntimeError("EF_solver_name must be specified for the amalgamator." )
        if not ('EF_solver_options' in opt):
            opt['EF_solver_options'] = {'mipgap': None}
        if not ('num_scens' in opt):
            raise RuntimeWarning("options should have a number of scenarios to compute a xhat")
        if _bool_option(options, 'EF-mstage') and 'branching_factors' not in options:
            raise RuntimeError("For a multistage problem, otpions must have a 'branching_factors' attribute with branching factors")

    return opt
    

#========================================
class Amalgamator():
    """Takes a scenario list and a scenario creator (and options)
    as input. The ides is to produce an outer bound on the objective function (solving the EF directly
    or by decomposition) and/or an x-hat with inner bound; however, what it does is controlled by
    its constructor options and by user options.

    This thing basically wraps the functionality of the "standard" *_cylinder examples.
    
    It may be an extenisble base class, but not abstract.

    Args:
        options (dict): controls the amalgamation
        scenario_names (list of str) the full set of scenario names
        scenario_creator (fct): returns a concrete model with special things
        kw_creator (fct): takes an args object and returns scenario_creator kwargs
        scenario_denouement (fct): (optional) called at conclusion
    """

    def __init__(self, options,
                 scenario_names, scenario_creator, kw_creator, 
                 scenario_denouement=None, verbose=True):
        self.options = options
        self.scenario_names = scenario_names
        self.scenario_creator = scenario_creator
        self.scenario_denouement = scenario_denouement
        self.kw_creator = kw_creator
        self.kwargs = self.kw_creator(self.options)
        self.verbose = verbose
        self.is_EF = _bool_option(options, "EF-2stage") or _bool_option(options, "EF-mstage")
        if self.is_EF:
            self.solvername = options.get('EF_solver_name', None)
            self.solver_options = options['EF_solver_options'] \
                if ('EF_solver_options' in options) else {}
        self.is_multi = _bool_option(options, "EF-mstage") or _bool_option(options, "mstage")
        if self.is_multi and not "all_nodenames" in options:
            if "branching_factors" in options:
                self.options["all_nodenames"] = sputils.create_nodenames_from_branching_factors(options["branching_factors"])
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

            tee_ef_solves = self.options.get('tee_ef_solves',False)
            
            solvername = self.solvername
            solver = pyo.SolverFactory(solvername)
            if hasattr(self, "solver_options") and (self.solver_options is not None):
                for option_key,option_value in self.solver_options.items():
                    if option_value is not None:
                        solver.options[option_key] = option_value
            if self.verbose :
                global_toc("Starting EF solve")
            if 'persistent' in solvername:
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
            
            if 'write_solution' in self.options:
                if 'first_stage_solution' in self.options['write_solution']:
                    sputils.write_ef_first_stage_solution(self.ef,
                                                          self.options['write_solution']['first_stage_solution'])
                if 'tree_solution' in self.options['write_solution']:
                    sputils.write_ef_tree_solution(self.ef,
                                                   self.options['write_solution']['tree_solution'])
            
            self.xhats = sputils.nonant_cache_from_ef(ef)
            self.local_xhats = self.xhats #Every scenario is local for EF
            self.first_stage_solution = {"ROOT": self.xhats["ROOT"]}

        else:
            self.ef = None
            args = argparse.Namespace(**self.options)
            
            #Create a hub dict
            hub_name = find_hub(self.options['cylinders'], self.is_multi)
            hub_creator = getattr(vanilla, hub_name+'_hub')
            beans = {"args":args,
                           "scenario_creator": self.scenario_creator,
                           "scenario_denouement": self.scenario_denouement,
                           "all_scenario_names": self.scenario_names,
                           "scenario_creator_kwargs": self.kwargs}
            if self.is_multi:
                beans["all_nodenames"] = self.options["all_nodenames"]
            hub_dict = hub_creator(**beans)
            
            #Add extensions
            if 'extensions' in self.options:
                for extension in self.options['extensions']:
                    extension_creator = getattr(vanilla, 'add_'+extension)
                    hub_dict = extension_creator(hub_dict,
                                                 args)
            
            #Create spoke dicts
            potential_spokes = find_spokes(self.options['cylinders'],
                                           self.is_multi)
            #We only use the spokes with an associated command line arg set to True
            spokes = [spoke for spoke in potential_spokes if self.options['with_'+spoke]]
            list_of_spoke_dict = list()
            for spoke in spokes:
                spoke_creator = getattr(vanilla, spoke+'_spoke')
                spoke_beans = copy.deepcopy(beans)
                if spoke == "xhatspecific":
                    spoke_beans["scenario_dict"] = self.options["scenario_dict"]
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
                
            
            if 'write_solution' in self.options:
                if 'first_stage_solution' in self.options['write_solution']:
                    ws.write_first_stage_solution(self.options['write_solution']['first_stage_solution'])
                if 'tree_solution' in self.options['write_solution']:
                    ws.write_tree_solution(self.options['write_solution']['tree_solution'])
            
            if self.on_hub: #we are on a hub rank
                a_sname = self.opt.local_scenario_names[0]
                root = self.opt.local_scenarios[a_sname]._mpisppy_node_list[0]
                self.first_stage_solution = {"ROOT":[pyo.value(var) for var in root.nonant_vardata_list]}
                self.local_xhats = ws.local_nonant_cache()
                
            #TODO: Add a xhats attribute, similar to the output of nonant_cache_from_ef
            #      It means doing a MPI operation over hub ranks


if __name__ == "__main__":
    # Our example is farmer
    import mpisppy.tests.examples.farmer as farmer
    # EF, PH, L-shaped, APH flags, and then boolean multi-stage
    ama_options = {"2stage": True,   # 2stage vs. mstage
                   "cylinders": ['ph','cross_scenario_cuts'],
                   "extensions": ['cross_scenario_cuts'],
                   "program_name": "amalgamator test main",
                   }
    ama = from_module("mpisppy.tests.examples.farmer", ama_options)
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
