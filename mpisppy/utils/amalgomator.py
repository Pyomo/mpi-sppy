# Amalgomator.py starting point; DLW March 2021
# Copyright 2021 by D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

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

import importlib
import pyomo.environ as pyo

from mpisppy.utils.sputils import spin_the_wheel, get_objs
from mpisppy.utils import baseparsers
from mpisppy.utils import vanilla
from mpisppy.opt.ef import ExtensiveForm
from mpisppy import global_toc

#==========
def from_module(mname, options, alist=None):
    """ Try to get everything from one file (this will not always be possible).
    Args:
        mname (str): the module name (module must have certain functions)
        options (dict): Amalgomator options
        alist (list): optional list of commands so as to bypass the command line
    Returns:
        ama (Amalgomator): the instantiated object
    """
    everything = ["scenario_names_creator",
                 "scenario_creator",
                 "inparser_adder",
                 "kw_creator"]  # denouement can be missing.

    assert(options is not None)
    m = importlib.import_module(mname)

    you_can_have_it_all = True
    for ething in everything:
        if not hasattr(m, ething):
            print(f"Module {mname} is missing {ething}")
            you_can_have_it_all = False
    if not you_can_have_it_all:
        raise RuntimeError(f"Module {mname} not complete for from_module")

    args = Amalgomator_parser(options, m.inparser_adder, alist)

    sn = m.scenario_names_creator(args.num_scens)
    dn = m.scenario_denouement if hasattr(m, "scenario_denouement") else None
    ama = Amalgomator(options,
                      args,
                      sn,
                      m.scenario_creator,
                      m.kw_creator,
                      scenario_denouement=dn)
    return ama


#==========
def _bool_option(options, oname):
    return oname in options and options[oname]


#==========
def Amalgomator_parser(options, inparser_adder, alist=None):
    """ Helper function for Amalgomator.  This gives us flexibility (e.g., get scen count)
    Args:
        options (dict): Amalgomator control options
        inparser_adder (fct): returns updated ArgumentParser the problem
        alist (list): optional list of commands so as to bypass the command line

    Returns;
        args (ArgumentParser.parse_args return): the parsed values
    """
    if alist is not None:
        raise RuntimeError("alist not supported yet")

    if _bool_option(options, "EF-2stage"):
        parser = baseparsers.make_EF2_parser(num_scens_reqd=_bool_option(options, "num_scens_reqd"))
    else:
        raise RuntimeError("only EF-2Stage is supported right now")

    # TBD add args for everything else that is not EF, which is a lot

    # call inparser last (so it can delete args if it needs to)
    inparser_adder(parser)

    args = parser.parse_args()
    return args
    

#========================================
class Amalgomator(object):
    """Takes a scenario list and a scenario creator (and options)
    as input. The ides is to produce an outer bound on the objective function (solving the EF directly
    or by decomposition) and/or an x-hat with inner bound; however, what it does is controlled by
    its constructor options and by user options.

    This thing basically wraps the functionality of the "standard" *_cylinder examples.
    
    It may be an extenisble base class, but not abstract.

    Args:
        options (dict): controls the amalgomation
        args (return from ArgumentParser.parse_args()): the parsed command line args
        scenario_names (list of str) the full set of scenario names
        scenario_creator (fct): returns a concrete model with special things
        kw_creator (fct): takes an args object and returns scenario_creator kwargs
        scenario_denouement (fct): (optional) called at conclusion
    """

    def __init__(self, options, args,
                 scenario_names, scenario_creator, kw_creator, scenario_denouement=None):
        self.options = options
        self.args = args
        self.scenario_names = scenario_names
        self.scenario_creator = scenario_creator
        self.scenario_denouement = scenario_denouement
        self.kw_creator = kw_creator
        self.is_EF = _bool_option(options, "EF-2stage") or _bool_option(options, "EF-mstage")

    def run(self):
        """ Top-level execution."""
        if self.is_EF:
            ef_options = {"solver": self.args.EF_solver_name}
            kwargs = self.kw_creator(self.args)
            ef = ExtensiveForm(ef_options,
                               self.scenario_names,
                               self.scenario_creator,
                               scenario_creator_kwargs=kwargs)
            solver_options = {"mipgap": self.args.EF_mipgap}
            global_toc("Starting EF solve")
            results = ef.solve_extensive_form()
            global_toc("Completed EF solve")
            if not pyo.check_optimal_termination(results):
                print(f"WARNING: EF was not solved to optimality; results={results}")
            self.EF_Obj = pyo.value(ef.ef.EF_Obj)
            objs = get_objs(ef.ef)
            self.is_minimizing = objs[0].is_minimizing
            if self.is_minimizing:
                self.best_outer_bound = results.Problem[0]['Lower bound']
                self.best_inner_bound = results.Problem[0]['Upper bound']
            else:
                self.best_inner_bound = results.Problem[0]['Upper bound']
                self.best_outer_bound = results.Problem[0]['Lower bound']
        else:
            raise RuntimeError("We can only do EF right now")


if __name__ == "__main__":
    # for debugging
    import mpisppy.tests.examples.farmer as farmer
    print("hello")
    ama_options = {"EF-2stage": True}   # 2stage vs. mstage
    ama = from_module("mpisppy.tests.examples.farmer", ama_options)
    ama.run()
    print(f"inner bound=", ama.best_inner_bound)
    print(f"outer bound=", ama.best_outer_bound)
