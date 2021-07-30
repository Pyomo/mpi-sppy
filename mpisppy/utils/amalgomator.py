# Amalgomator.py starting point; DLW March 2021
# Copyright 2021 by D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# To test : python amalgomator.py --num-scens=10

"""Takes a scenario list and a scenario creator (and options)
as input and produces an outer bound on the objective function (solving the EF directly
or by decomposition). This same object should also be able to return an x-hat. 

This thing basically wraps the functionality of the "standard" *_cylinder examples.

It may be an extenisble base class, but not abstract.


Wait. I guess I need a function that takes an argparse object and returns the kwargs dict for the
scenario creator function and I need to be passed in a starting argparse object that
has the needed stuff for the problem at hand.

"""

"""
Input options:
EF|PH|APH|L, etc
Assuming not EF:
  - list of cylinders
  - list of hub extensions

The object will then call the appropriate baseparsers functions to set up
the args and assemble the objects needed for spin-the-wheel, which it will call.

[NOTE: argparse can be passed a command list and can generate a command list that can be passed]


Things that will be needed to support this:
  - mapping from cylinder name to class?
  - mapping from extension name to class?
  = The same kind of mapping to the baseparser functions
  = Support in baseparsers for "all" cylinders and extensions with a mapping to the function
 ==== we can't keep hanging at the termination barrier!

*mapping*: I think the cylinder "name" and extension "name" should just be the module
name to import in string from (e.g., "mpisppy.opt.ph") and then use importlib to 
import it.

? Should baseparser functions "register themselves" in a import-name-string: function
map that is global to baseparsers (i.e. importable)?


"""
import numpy as np
import importlib
import pyomo.environ as pyo
import argparse

from mpisppy.utils.sputils import spin_the_wheel, get_objs, nonant_cache_from_ef
import mpisppy.utils.baseparsers as baseparsers
import mpisppy.utils.sputils as sputils
from mpisppy.utils import vanilla
from mpisppy import global_toc

#==========
def from_module(mname, options, extraargs=None, use_command_line=True):
    """ Try to get everything from one file (this will not always be possible).
    Args:
        mname (str): the module name (module must have certain functions)
        options (dict): Amalgomator options or extra arguments to use 
                        in addition with the commande line
        extraargs (ArgumentParser) : extra arguments to specify in the command line, e.g. for MMW
        use_command_line (bool): should we take into account the command line to add options ?
                                 default is True
                    
    Returns:
        ama (Amalgomator): the instantiated object
    
    Note :
        Use adict iif you want to bypass the command line.
    """
    everything = ["scenario_names_creator",
                 "scenario_creator",
                 "inparser_adder",
                 "kw_creator"]  # start and denouement can be missing.

    assert(options is not None)
    m = importlib.import_module(mname)

    you_can_have_it_all = True
    for ething in everything:
        if not hasattr(m, ething):
            print(f"Module {mname} is missing {ething}")
            you_can_have_it_all = False
    if not you_can_have_it_all:
        raise RuntimeError(f"Module {mname} not complete for from_module")
        
    options = Amalgomator_parser(options, m.inparser_adder,
                                 extraargs=extraargs,
                                 use_command_line=use_command_line)
    options['_mpisppy_probability'] = 1/options['num_scens']
    start = options['start'] if(('start' in options)) else None
    sn = m.scenario_names_creator(options['num_scens'], start=start)
    dn = m.scenario_denouement if hasattr(m, "scenario_denouement") else None
    ama = Amalgomator(options,
                      sn,
                      m.scenario_creator,
                      m.kw_creator,
                      scenario_denouement=dn)
    return ama


#==========
def _bool_option(options, oname):
    return oname in options and options[oname]

def _basic_parse_args(progname = None, is_multi=False,  num_scens_reqd=False):
    if is_multi:
        parser = baseparsers.make_multistage_parser(progname= progname)
    else:
        parser = baseparsers.make_parser(progname=progname, 
                                         num_scens_reqd=num_scens_reqd)
        
    return parser

def add_parser(inparser, parser_choice=None):
    if (parser_choice is None) or not hasattr(baseparsers,parser_choice+"_args"):
        return inparser
    else:
        adder = getattr(baseparsers,parser_choice+"_args")
        parser = adder(inparser)
        return parser
    
hubs_and_multi_compatibility = {'ph': True,
                                'aph': True, 
                                'lshaped':False, #No parser
                                'cross_scen_hub':False} #No parser

spokes_and_multi_compatibility = {'fwph':True,
                                  'lagrangian':False,
                                  'lagranger':False,
                                  'xhatlooper':False,
                                  'xhatshuffle':False,
                                  'xhatspecific':True,
                                  'xhatlshaped':False,
                                  'slamup':False,
                                  'slamdown':False,
                                  'cross_scenario_cuts':False}

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
            if is_multi and not spokes_and_multi_compatibility[c]:
                raise RuntimeError(f"The spoke {c} does not work with multistage problems" )
            spokes.append(c)
    return spokes
                
        
#==========
def Amalgomator_parser(options, inparser_adder, extraargs=None, use_command_line=True):
    """ Helper function for Amalgomator.  This gives us flexibility (e.g., get scen count)
    Args:
        options (dict): Amalgomator control options
        inparser_adder (fct): returns updated ArgumentParser the problem
        extraargs (ArgumentParser) : extra arguments to specify in the command line, e.g. for MMW
        use_command_line (bool): should we take into account the command line to add options ?
                                 default is True
    Returns;
        options (dict): a dict containing the options, both parsed values and pre-set options
    """

    opt = {**options}
    
    if use_command_line:
        num_scens_reqd=_bool_option(options, "num_scens_reqd")
        if _bool_option(options, "EF-2stage"):
            parser = baseparsers.make_EF2_parser(num_scens_reqd=num_scens_reqd)
        elif _bool_option(options, "EF-mstage"):
            parser = baseparsers.make_EF_multistage_parser(num_scens_reqd=num_scens_reqd)
        else:
            if _bool_option(options, "2stage"):
                parser = _basic_parse_args(is_multi=False, num_scens_reqd=num_scens_reqd)
            elif _bool_option(options, "mstage"):
                parser = _basic_parse_args(progname = parser, is_multi=True)
            else:
                raise RuntimeError("The problem type (2stage or mstage) must be specified")
            parser = baseparsers.two_sided_args(parser)
                
            #Adding cylinders
            if not "cylinders" in options:
                raise RuntimeError("A cylinder list must be specified")
            else:
                for cylinder in options['cylinders']:
                    #TODO: Add a list of existing cylinders value and check if they exists
                    parser = add_parser(parser,cylinder)
            if "extensions" in options:
                for extension in options["extensions"]:
                    parser = add_parser(parser,extension)
                raise RuntimeWarning("The extensions are not passed to cylinders for now.")
    
        # TBD add args for everything else that is not EF, which is a lot
    
        # call inparser last (so it can delete args if it needs to)
        inparser_adder(parser)
        
        if extraargs is not None:
            parser = argparse.ArgumentParser(parents = [parser,extraargs],
                                             conflict_handler='resolve')
        
        args = parser.parse_args()
    
        opt.update(vars(args)) #Changes made via the command line overwrite what is in options 
        
        print("Hello", opt)
        
        if _bool_option(options, "EF-2stage") or _bool_option(options, "EF-mstage"): 
            if ('EF_solver_options' in opt):
                opt["EF_solver_options"]["mipgap"] = opt["EF_mipgap"]
            else:
                opt["EF_solver_options"] = {"mipgap": opt["EF_mipgap"]}
    
    else:
        #Checking if options has all the options we need 
        if not _bool_option(options, "EF-2stage") or _bool_option(options, "EF-mstage"):
            raise RuntimeError("For now, completly bypassing command line only works with EF." )
        if not ('EF_solver_name' in opt):
            opt['EF_solver_name'] = "gurobi"
        if not ('EF_solver_options' in opt):
            opt['EF_solver_options'] = {'mipgap': None}
        if not ('num_scens' in opt):
            raise RuntimeWarning("options should have a number of scenarios to compute a xhat")
        if _bool_option(options, 'EF-mstage') and 'BFs' not in options:
            raise RuntimeError("For a multistage problem, otpions must have a 'BFs' attribute with branching factors")

    return opt
    

#========================================
class Amalgomator():
    """Takes a scenario list and a scenario creator (and options)
    as input. The ides is to produce an outer bound on the objective function (solving the EF directly
    or by decomposition) and/or an x-hat with inner bound; however, what it does is controlled by
    its constructor options and by user options.

    This thing basically wraps the functionality of the "standard" *_cylinder examples.
    
    It may be an extenisble base class, but not abstract.

    Args:
        options (dict): controls the amalgomation
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
            self.solvername = options['EF_solver_name'] if  ('EF_solver_name' in options) else 'gurobi'
            self.solver_options = options['EF_solver_options'] \
                if ('EF_solver_options' in options) else {}
        self.is_multi = _bool_option(options, "EF-mstage") or _bool_option(options, "mstage")
        
    def run(self):
        
        """ Top-level execution."""
        if self.is_EF:
            ef = sputils.create_EF(
                self.scenario_names,
                self.scenario_creator,
                scenario_creator_kwargs=self.kwargs,
                suppress_warnings=True,
            )
            
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
                results = solver.solve(tee=False)
            else:
                results = solver.solve(ef, tee=False, symbolic_solver_labels=True,)
            if self.verbose:
                global_toc("Completed EF solve")

            
            self.EF_Obj = pyo.value(ef.EF_Obj)

            objs = get_objs(ef)
            
            self.is_minimizing = objs[0].is_minimizing
            #TBD : Write a function doing this
            if self.is_minimizing:
                self.best_outer_bound = results.Problem[0]['Lower bound']
                self.best_inner_bound = results.Problem[0]['Upper bound']
            else:
                self.best_inner_bound = results.Problem[0]['Upper bound']
                self.best_outer_bound = results.Problem[0]['Lower bound']
            self.ef = ef

        else:
            self.ef = None
            args = argparse.Namespace(**self.options)
            
            hub_name = find_hub(self.options['cylinders'], self.is_multi)
            hub_creator = getattr(vanilla, hub_name+'_hub')
            hub_dict = hub_creator(args = args,
                                   scenario_creator = self.scenario_creator,
                                   scenario_denouement = self.scenario_denouement,
                                   all_scenario_names = self.scenario_names,
                                   scenario_creator_kwargs= self.kwargs,
                                   ph_extensions=None,
                                   rho_setter=None,
                                   variable_probability=None)
            
            potential_spokes = find_spokes(self.options['cylinders'],
                                           self.is_multi)
            spokes = [spoke for spoke in potential_spokes if self.options['with_'+spoke]]
            list_of_spoke_dict = list()
            for spoke in spokes:
                spoke_creator = getattr(vanilla, spoke+'_hub')
                spoke_dict = spoke_creator(args = args,
                                           scenario_creator = self.scenario_creator,
                                           scenario_denouement = self.scenario_denouement,
                                           all_scenario_names = self.scenario_names,
                                           scenario_creator_kwargs= self.kwargs,)
                list_of_spoke_dict.append(spoke_dict)
                
            spcomm, opt_dict = sputils.spin_the_wheel(hub_dict, list_of_spoke_dict)
            


if __name__ == "__main__":
    # for debugging
    import mpisppy.tests.examples.farmer as farmer
    # EF, PH, L-shaped, APH flags, and then boolean multi-stage
    ama_options = {"2stage": True,   # 2stage vs. mstage
                   "cylinders": ['ph'],
                   }
    ama = from_module("mpisppy.tests.examples.farmer", ama_options)
    ama.run()
    # print(f"inner bound=", ama.best_inner_bound)
    # print(f"outer bound=", ama.best_outer_bound)
    # print(nonant_cache_from_ef(ama.ef))
    # wish list: allow for integer relaxation using the Pyomo inplace transformation
    """ Issues:
    0. What do we want to put in ama_options and what do we want to discover
       by looking at data? I am inclined to put things in the data and maybe
       check against data. Here are the things:
       - 2-stage versus multi-stage
       - MIP versus continuous only
       - linear/quadratic versus non-linear
    """
