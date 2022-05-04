# This software is distributed under the 3-clause BSD License.
# Started 12 April by DLW
# Replace baseparsers.py and enhance functionality.
# The vision is that global_config ConfigDict can be accessed easily
# by all code.
# NOTE: at first I am just copying directly from baseparsers.py
# TBD: reorganize some of this.

""" Notes 
The default for all 'with' options is False
       (and we are dropping the `no` side that was in baseparsers.py)
       (so we are also dropping the use of with_)

Now you assemble the args you want and call the create_parser function,
   which returns an argparse object, E.g.:
parser = config.create_parser("myprog")

After you you parse with this argparse object (assume it is called parser), 
you need to call global_config.import_argparse 
E.g.
args = parser.parse_args()
args = global_config.import_argparse(parser)

If you want to add args, you need to call the add_to_config function

If you want a required arg, you have to DYI:
    parser = config.create_parser("tester")
    parser.add_argument(
            "num_scens", help="Number of scenarios", type=int,
        )
    args=parser.parse_args(['3', '--max-iterations', '99', '--solver-name', 'cplex'])
    print(f"{args.num_scens =}")
(Note: you can still attach it to global_config, but that is also DYI)

"""

"""
TBD:
  - Do a better job with solver options.

  - There is some discussion of name spaces, but I think for
things coming off the command line, we can't *really* do that since
there is just one command line.

"""

""" Notes on conversion (delete this docstring by summer 2022)

See hydro_cylinders_config.py compared to hydro_cylinders.py

import config (not baseparsers)
parser = baseparsers.  --> config. (sort of)
add_argument --> config.add_to_config

Parse_args returns args for the sake of vanilla (which will fail anyway... so vanilla needs to be updated); 
I want to make a clean break.
Every use of args gets replaced with  cfg = config.global_config in the main() function

"""

import argparse
import pyomo.common.config as pyofig

global_config = pyofig.ConfigDict()

#===============
def add_to_config(name, description, domain, default,
                  argparse=True,
                  complain=False,
                  group=None):
    """ Add an arg to the global_config dict.
    Args:
        name (str): the argument name, underscore seperated
        description (str): free text description
        argparse (bool): if True put on command ine
        complain (bool): if True, output a message for a duplicate
        group (str): argparse group name (optional; PRESENTLY NOT USED)
    """
    if name in global_config:
        if complain:
            print(f"Trying to add duplicate {name} to global_config.")
            # raise RuntimeError(f"Trying to add duplicate {name} to global_config.")
    else:
        c = global_config.declare(name, pyofig.ConfigValue(
            description = description,
            domain = domain,
            default = default))
    if argparse:
        c.declare_as_argument()
        #argname = "--" + name.replace("_","-")
        #if group is not None:
        #    c.declare_as_argument(argname, group=group)
        #else:
        #    c.declare_as_argument(argname)


def _common_args():
    raise RuntimeError("_common_args is no longer used. See comments at top of config.py")

def popular_args():
    add_to_config("max_iterations", 
                        description="ph max iiterations (default 1)",
                        domain=int,
                        default=1)

    add_to_config("solver_name", 
                        description= "solver name (default gurobi)",
                        domain = str,
                        default="gurobi")

    add_to_config("seed", 
                        description="Seed for random numbers (default is 1134)",
                        domain=int,
                        default=1134)

    add_to_config("default_rho", 
                        description="Global rho for PH (default None)",
                        domain=float,
                        default=None)

    add_to_config("bundles_per_rank", 
                        description="bundles per rank (default 0 (no bundles))",
                        domain=int,
                        default=0)                

    add_to_config('verbose', 
                          description="verbose output",
                          domain=bool,
                          default=False)

    add_to_config('display_progress', 
                          description="display progress at each iteration",
                          domain=bool,
                          default=False)

    add_to_config('display_convergence_detail', 
                          description="display non-anticipative variable convergence statistics at each iteration",
                          domain=bool,
                          default=False)

    add_to_config("max_solver_threads", 
                        description="Limit on threads per solver (default None)",
                        domain=int,
                        default=None)

    add_to_config("intra_hub_conv_thresh", 
                        description="Within hub convergence threshold (default 1e-10)",
                        domain=float,
                        default=1e-10)

    add_to_config("trace_prefix", 
                        description="Prefix for bound spoke trace files. If None "
                             "bound spoke trace files are not written.",
                        domain=str,
                        default=None)

    add_to_config("tee_rank0_solves", 
                          description="Some cylinders support tee of rank 0 solves."
                          "(With multiple cylinders this could be confusing.)",
                          domain=bool,
                          default=False)

    add_to_config("auxilliary", 
                        description="Free text for use by hackers (default '').",
                        domain=str,
                        default='')

    add_to_config("linearize_binary_proximal_terms", 
                          description="For PH, linearize the proximal terms for "
                          "all binary nonanticipative variables",
                          domain=bool,
                          default=False)


    add_to_config("linearize_proximal_terms", 
                          description="For PH, linearize the proximal terms for "
                          "all nonanticipative variables",
                          domain=bool,
                          default=False)


    add_to_config("proximal_linearization_tolerance", 
                        description="For PH, when linearizing proximal terms, "
                        "a cut will be added if the proximal term approximation "
                        "is looser than this value (default 1e-1)",
                        domain=float,
                        default=1.e-1)


def make_parser(progname=None, num_scens_reqd=False):
    raise RuntimeError("make_parser is no longer used. See comments at top of config.py")


def num_scens_optional():
        add_to_config(
            "num_scens", 
            description="Number of scenarios (default None)",
            domain=int,
            default=None,
            )
    

def _basic_multistage(progname=None, num_scens_reqd=False):
    raise RuntimeError("_basic_multistage is no longer used. See comments at top of config.py")
    parser = argparse.ArgumentParser(prog=progname, conflict_handler="resolve")

def branching_factors():
    add_to_config("branching_factors", 
                        description="Spaces delimited branching factors (e.g., 2 2)",
                        domain=pyofig.ListOf(int, pyofig.PositiveInt),
                        default=None)
        

def make_multistage_parser(progname=None):
    raise RuntimeError("make_multistage_parser is no longer used. See comments at top of config.py")    

def multistage():
    parser = branching_factors()
    parser = popular_args()
    

#### EF ####
def make_EF2_parser(progname=None, num_scens_reqd=False):
    raise RuntimeError("make_EF2_parser is no longer used. See comments at top of config.py")    

def EF2():
    add_to_config(
        "num_scens", 
            description="Number of scenarios (default None)",
            domain=int,
            default=None,
        )
        
    add_to_config("EF_solver_name", 
        description= "solver name (default gurobi)",
        domain = str,
        default="gurobi")


    add_to_config("EF_mipgap", 
                        description="mip gap option for the solver if needed (default None)",
                        domain=float,
                        default=None)
    

def make_EF_multistage_parser(progname=None, num_scens_reqd=False):
    raise RuntimeError("make_EF_multistage_parser is no longer used. See comments at top of config.py")        

def EF_multistage():
    add_to_config(
        "num_scens", 
            description="Number of scenarios (default None)",
            domain=int,
            default=None,
        )
        
    add_to_config("EF_solver_name", 
        description= "solver name (default gurobi)",
        domain = str,
        default="gurobi")


    add_to_config("EF_mipgap", 
        description="mip gap option for the solver if needed (default None)",
        domain=float,
        default=None)
    
##### common additions to the command line #####

def two_sided_args():
    # add commands to  and also return the result
    
    add_to_config("rel_gap", 
                        description="relative termination gap (default 0.05)",
                        domain=float,
                        default=0.05)

    add_to_config("abs_gap", 
                        description="absolute termination gap (default 0)",
                        domain=float,
                        default=0.)
    
    add_to_config("max_stalled_iters", 
                        description="maximum iterations with no reduction in gap (default 100)",
                        domain=int,
                        default=100)

    

def mip_options():
    
    add_to_config("iter0_mipgap", 
                        description="mip gap option for iteration 0 (default None)",
                        domain=float,
                        default=None)

    add_to_config("iterk_mipgap", 
                        description="mip gap option non-zero iterations (default None)",
                        domain=float,
                        default=None)
    

def aph_args():
    
    add_to_config('aph_gamma', 
                        description='Gamma parameter associated with asychronous projective hedging (default 1.0)',
                        domain=float,
                        default=1.0)
    add_to_config('aph_nu', 
                        description='Nu parameter associated with asychronous projective hedging (default 1.0)',
                        domain=float,
                        default=1.0)
    add_to_config('aph_frac_needed', 
                        description='Fraction of sub-problems required before computing projective step (default 1.0)',
                        domain=float,
                        default=1.0)
    add_to_config('aph_dispatch_frac', 
                        description='Fraction of sub-problems to dispatch at each step of asychronous projective hedging (default 1.0)',
                        domain=float,
                        default=1.0)
    add_to_config('aph_sleep_seconds', 
                        description='Spin-lock sleep time for APH (default 0.01)',
                        domain=float,
                        default=0.01)    
        

def fixer_args():
    
    add_to_config('fixer', 
                          description="have an integer fixer extension (default)",
                          domain=bool,
                          default=False)

    add_to_config("fixer_tol", 
                        description="fixer bounds tolerance  (default 1e-4)",
                        domain=float,
                        default=1e-2)
    


def fwph_args():
    
    add_to_config('fwph', 
                          description="have an fwph spoke (default)",
                          domain=bool,
                          default=False)

    add_to_config("fwph_iter_limit", 
                        description="maximum fwph iterations (default 10)",
                        domain=int,
                        default=10)

    add_to_config("fwph_weight", 
                        description="fwph weight (default 0)",
                        domain=float,
                        default=0.0)

    add_to_config("fwph_conv_thresh", 
                        description="fwph convergence threshold  (default 1e-4)",
                        domain=float,
                        default=1e-4)

    add_to_config("fwph_stop_check_tol", 
                        description="fwph tolerance for Gamma^t (default 1e-4)",
                        domain=float,
                        default=1e-4)

    add_to_config("fwph_mipgap", 
                        description="mip gap option FW subproblems iterations (default None)",
                        domain=float,
                        default=None)

    

def lagrangian_args():
    
    add_to_config('lagrangian', 
                          description="have a lagrangian spoke (default)",
                          domain=bool,
                          default=False)

    add_to_config("lagrangian_iter0_mipgap", 
                        description="lgr. iter0 solver option mipgap (default None)",
                        domain=float,
                        default=None)

    add_to_config("lagrangian_iterk_mipgap", 
                        description="lgr. iterk solver option mipgap (default None)",
                        domain=float,
                        default=None)

    


def lagranger_args():
    
    add_to_config('lagranger', 
                        description="have a special lagranger spoke (default)",
                          domain=bool,
                          default=False)

    add_to_config("lagranger_iter0_mipgap", 
                        description="lagranger iter0 mipgap (default None)",
                        domain=float,
                        default=None)

    add_to_config("lagranger_iterk_mipgap", 
                        description="lagranger iterk mipgap (default None)",
                        domain=float,
                        default=None)

    add_to_config("lagranger_rho_rescale_factors_json", 
                        description="json file: rho rescale factors (default None)",
                        domain=str,
                        default=None)

    


def xhatlooper_args():
    
    add_to_config('xhatlooper', 
                          description="have an xhatlooper spoke (default)",
                          domain=bool,
                          default=False)

    add_to_config("xhat_scen_limit", 
                        description="scenario limit xhat looper to try (default 3)",
                        domain=int,
                        default=3)

    


def xhatshuffle_args():
    
    add_to_config('xhatshuffle', 
                          description="have an xhatshuffle spoke (default)",
                          domain=bool,
                          default=False)
    
    add_to_config('add_reversed_shuffle', 
                        description="using also the reversed shuffling (multistage only, default True)",
                          domain=bool,
                          default=False)
    
    add_to_config('xhatshuffle_iter_step', 
                        description="step in shuffled list between 2 scenarios to try (default None)",
                        domain=int,
                        default=None)

    


def xhatspecific_args():
    # we will not try to get the specification from the command line
    
    add_to_config('xhatspecific', 
                          description="have an xhatspecific spoke (default)",
                          domain=bool,
                          default=False)

    


def xhatlshaped_args():
    # we will not try to get the specification from the command line
    
    add_to_config('xhatlshaped', 
                          description="have an xhatlshaped spoke (default)",
                          domain=bool,
                          default=False)

    

def slamup_args():
    # we will not try to get the specification from the command line
    
    add_to_config('slamup', 
                        description="have an slamup spoke (default)",
                          domain=bool,
                          default=False)

    

def slamdown_args():
    # we will not try to get the specification from the command line
    
    add_to_config('slamdown', 
                        description="have an slamdown spoke (default)",
                          domain=bool,
                          default=False)

    

def cross_scenario_cuts_args():
    # we will not try to get the specification from the command line
    
    add_to_config('cross_scenario_cuts', 
                          description="have a cross scenario cuts spoke (default)",
                          domain=bool,
                          default=False)

    add_to_config("cross_scenario_iter_cnt", 
                          description="cross scen check bound improve iterations "
                          "(default 4)",
                          domain=int,
                          default=4)

    add_to_config("eta_bounds_mipgap", 
                          description="mipgap for determining eta bounds for cross "
                          "scenario cuts (default 0.01)",
                          domain=float,
                          default=0.01)


#================
def create_parser(progname=None):
    if len(global_config) == 0:
        raise RuntimeError("create parser called before global_config is populated")
    parser = argparse.ArgumentParser(progname, conflict_handler="resolve")
    global_config.initialize_argparse(parser)
    return parser

#=================
if __name__ == "__main__":
    # a place for ad hoc testing by developers
    popular_args() # populates global_config
    global_config.display()
    for i,j in global_config.items():
        print(i, j)
    print(dir(global_config))
    print(global_config._all_slots)
    print(global_config._domain)
    print(f"{global_config['max_iterations'] =}")

    parser = create_parser("tester")
    parser.add_argument(
            "num_scens", help="Number of scenarios", type=int,
        )

    args=parser.parse_args(['3', '--max-iterations', '99', '--solver-name', 'cplex'])

    print(f"{args.num_scens =}")
    
    args = global_config.import_argparse(args)
    
    global_config.display()    

    #parser.parse_args(['--help'])




    
