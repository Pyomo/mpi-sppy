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
  Do a better job with solver options.

"""

""" Notes on conversion (delete this docstring by summer 2022)

See hydro_cylinders_config.py compared to hydro_cylinders.py

import config (not baseparsers)
parser = baseparsers.  --> config. (sort of)
add_argument --> config.add_to_config

Parse_args returns args for the sake of vanilla; but I want to make a clean break
Every use of args gets replaced with  cfg = config.global_config in the main() function

Here's the bummer: args.xxx_yyy becomes cfg["xxx-yyy"], but it's really not too bad 
"""

import argparse
import pyomo.common.config as pyofig

global_config = pyofig.ConfigDict()

def _common_args():
    raise RuntimeError("_common_args is no longer used. See comments at top of config.py")

def popular_args():
    global_config.declare("max-iterations", pyofig.ConfigValue(
                        description="ph max-iterations (default 1)",
                        domain=int,
                        default=1)).declare_as_argument()

    global_config.declare("solver-name", pyofig.ConfigValue(
                        description= "solver name (default gurobi)",
                        domain = str,
                        default="gurobi")).declare_as_argument().declare_as_argument()

    global_config.declare("seed", pyofig.ConfigValue(
                        description="Seed for random numbers (default is 1134)",
                        domain=int,
                        default=1134)).declare_as_argument()

    global_config.declare("default-rho", pyofig.ConfigValue(
                        description="Global rho for PH (default None)",
                        domain=float,
                        default=None)).declare_as_argument()

    global_config.declare("bundles-per-rank", pyofig.ConfigValue(
                        description="bundles per rank (default 0 (no bundles))",
                        domain=int,
                        default=0)).declare_as_argument()                

    global_config.declare('with-verbose', pyofig.ConfigValue(
                          description="verbose output",
                          domain=bool,
                          default=False)).declare_as_argument()

    global_config.declare('with-display-progress', pyofig.ConfigValue(
                          description="display progress at each iteration",
                          domain=bool,
                          default=False)).declare_as_argument()

    global_config.declare('with-display-convergence-detail', pyofig.ConfigValue(
                          description="display non-anticipative variable convergence statistics at each iteration",
                          domain=bool,
                          default=False)).declare_as_argument()

    global_config.declare("max-solver-threads", pyofig.ConfigValue(
                        description="Limit on threads per solver (default None)",
                        domain=int,
                        default=None)).declare_as_argument()

    global_config.declare("intra-hub-conv-thresh", pyofig.ConfigValue(
                        description="Within hub convergence threshold (default 1e-10)",
                        domain=float,
                        default=1e-10)).declare_as_argument()

    global_config.declare("trace-prefix", pyofig.ConfigValue(
                        description="Prefix for bound spoke trace files. If None "
                             "bound spoke trace files are not written.",
                        domain=str,
                        default=None)).declare_as_argument()

    global_config.declare("with-tee-rank0-solves", pyofig.ConfigValue(
                          description="Some cylinders support tee of rank 0 solves."
                          "(With multiple cylinder this could be confusing.)",
                          domain=bool,
                          default=False)).declare_as_argument()

    global_config.declare("auxilliary", pyofig.ConfigValue(
                        description="Free text for use by hackers (default '').",
                        domain=str,
                        default='')).declare_as_argument()

    global_config.declare("linearize-binary-proximal-terms", pyofig.ConfigValue(
                          description="For PH, linearize the proximal terms for "
                          "all binary nonanticipative variables",
                          domain=bool,
                          default=False)).declare_as_argument()


    global_config.declare("linearize-proximal-terms", pyofig.ConfigValue(
                          description="For PH, linearize the proximal terms for "
                          "all nonanticipative variables",
                          domain=bool,
                          default=False)).declare_as_argument()


    global_config.declare("proximal-linearization-tolerance", pyofig.ConfigValue(
                        description="For PH, when linearizing proximal terms, "
                        "a cut will be added if the proximal term approximation "
                        "is looser than this value (default 1e-1)",
                        domain=float,
                        default=1.e-1)).declare_as_argument()


def make_parser(progname=None, num_scens_reqd=False):
    raise RuntimeError("make_parser is no longer used. See comments at top of config.py")


def num_scens_optional():
        global_config.declare(
            "num-scens", pyofig.ConfigValue(
            description="Number of scenarios (default None)",
            domain=int,
            default=None,
            )).declare_as_argument()
    

def _basic_multistage(progname=None, num_scens_reqd=False):
    raise RuntimeError("_basic_multistage is no longer used. See comments at top of config.py")
    parser = argparse.ArgumentParser(prog=progname, conflict_handler="resolve")

def branching_factors():
    global_config.declare("branching-factors", pyofig.ConfigValue(
                        description="Spaces delimited branching factors (e.g., 2 2)",
                        nargs="*",
                        domain=int,
                        default=None)).declare_as_argument()
        

def make_multistage_parser(progname=None):
    raise RuntimeError("make_multistage_parser is no longer used. See comments at top of config.py")    

def multistage():
    parser = _basic_multistage(progname=None)
    parser = _common_args(parser)
    

#### EF ####
def make_EF2_parser(progname=None, num_scens_reqd=False):
    raise RuntimeError("make_EF2_parser is no longer used. See comments at top of config.py")    

def EF2():
    global_config.declare(
        "num-scens", pyofig.ConfigValue(
            description="Number of scenarios (default None)",
            domain=int,
            default=None,
        )).declare_as_argument()
        
    global_config.declare("EF-solver-name", pyofig.ConfigValue(
        description= "solver name (default gurobi)",
        domain = str,
        default="gurobi")).declare_as_argument()


    global_config.declare("EF-mipgap", pyofig.ConfigValue(
                        description="mip gap option for the solver if needed (default None)",
                        domain=float,
                        default=None)).declare_as_argument()
    

def make_EF_multistage_parser(progname=None, num_scens_reqd=False):
    raise RuntimeError("make_EF_multistage_parser is no longer used. See comments at top of config.py")        

def EF_multistage():
    global_config.declare(
        "num-scens", pyofig.ConfigValue(
            description="Number of scenarios (default None)",
            domain=int,
            default=None,
        )).declare_as_argument()
        
    global_config.declare("EF-solver-name", pyofig.ConfigValue(
        description= "solver name (default gurobi)",
        domain = str,
        default="gurobi")).declare_as_argument()


    global_config.declare("EF-mipgap", pyofig.ConfigValue(
        description="mip gap option for the solver if needed (default None)",
        domain=float,
        default=None)).declare_as_argument()
    
##### common additions to the command line #####

def two_sided_args():
    # add commands to  and also return the result
    
    global_config.declare("rel-gap", pyofig.ConfigValue(
                        description="relative termination gap (default 0.05)",
                        domain=float,
                        default=0.05)).declare_as_argument()

    global_config.declare("abs-gap", pyofig.ConfigValue(
                        description="absolute termination gap (default 0)",
                        domain=float,
                        default=0.)).declare_as_argument()
    
    global_config.declare("max-stalled-iters", pyofig.ConfigValue(
                        description="maximum iterations with no reduction in gap (default 100)",
                        domain=int,
                        default=100)).declare_as_argument()

    

def mip_options():
    
    global_config.declare("iter0-mipgap", pyofig.ConfigValue(
                        description="mip gap option for iteration 0 (default None)",
                        domain=float,
                        default=None)).declare_as_argument()

    global_config.declare("iterk-mipgap", pyofig.ConfigValue(
                        description="mip gap option non-zero iterations (default None)",
                        domain=float,
                        default=None)).declare_as_argument()
    

def aph_args():
    
    global_config.declare('aph-gamma', pyofig.ConfigValue(
                        description='Gamma parameter associated with asychronous projective hedging (default 1.0)',
                        domain=float,
                        default=1.0)).declare_as_argument()
    global_config.declare('aph-nu', pyofig.ConfigValue(
                        description='Nu parameter associated with asychronous projective hedging (default 1.0)',
                        domain=float,
                        default=1.0)).declare_as_argument()
    global_config.declare('aph-frac-needed', pyofig.ConfigValue(
                        description='Fraction of sub-problems required before computing projective step (default 1.0)',
                        domain=float,
                        default=1.0)).declare_as_argument()
    global_config.declare('aph-dispatch-frac', pyofig.ConfigValue(
                        description='Fraction of sub-problems to dispatch at each step of asychronous projective hedging (default 1.0)',
                        domain=float,
                        default=1.0)).declare_as_argument()
    global_config.declare('aph-sleep-seconds', pyofig.ConfigValue(
                        description='Spin-lock sleep time for APH (default 0.01)',
                        domain=float,
                        default=0.01)).declare_as_argument()    
        

def fixer_args():
    
    global_config.declare('with-fixer', pyofig.ConfigValue(
                          description="have an integer fixer extension (default)",
                          domain=bool,
                          default=False)).declare_as_argument()

    global_config.declare("fixer-tol", pyofig.ConfigValue(
                        description="fixer bounds tolerance  (default 1e-4)",
                        domain=float,
                        default=1e-2)).declare_as_argument()
    


def fwph_args():
    
    global_config.declare('with-fwph', pyofig.ConfigValue(
                          description="have an fwph spoke (default)",
                          domain=bool,
                          default=False)).declare_as_argument()

    global_config.declare("fwph-iter-limit", pyofig.ConfigValue(
                        description="maximum fwph iterations (default 10)",
                        domain=int,
                        default=10)).declare_as_argument()

    global_config.declare("fwph-weight", pyofig.ConfigValue(
                        description="fwph weight (default 0)",
                        domain=float,
                        default=0.0)).declare_as_argument()

    global_config.declare("fwph-conv-thresh", pyofig.ConfigValue(
                        description="fwph convergence threshold  (default 1e-4)",
                        domain=float,
                        default=1e-4)).declare_as_argument()

    global_config.declare("fwph-stop-check-tol", pyofig.ConfigValue(
                        description="fwph tolerance for Gamma^t (default 1e-4)",
                        domain=float,
                        default=1e-4)).declare_as_argument()

    global_config.declare("fwph-mipgap", pyofig.ConfigValue(
                        description="mip gap option FW subproblems iterations (default None)",
                        domain=float,
                        default=None)).declare_as_argument()

    

def lagrangian_args():
    
    global_config.declare('with-lagrangian', pyofig.ConfigValue(
                          description="have a lagrangian spoke (default)",
                          domain=bool,
                          default=False)).declare_as_argument()

    global_config.declare("lagrangian-iter0-mipgap", pyofig.ConfigValue(
                        description="lgr. iter0 solver option mipgap (default None)",
                        domain=float,
                        default=None)).declare_as_argument()

    global_config.declare("lagrangian-iterk-mipgap", pyofig.ConfigValue(
                        description="lgr. iterk solver option mipgap (default None)",
                        domain=float,
                        default=None)).declare_as_argument()

    


def lagranger_args():
    
    global_config.declare('with-lagranger', pyofig.ConfigValue(
                        description="have a special lagranger spoke (default)",
                          domain=bool,
                          default=False)).declare_as_argument()

    global_config.declare("lagranger-iter0-mipgap", pyofig.ConfigValue(
                        description="lagranger iter0 mipgap (default None)",
                        domain=float,
                        default=None)).declare_as_argument()

    global_config.declare("lagranger-iterk-mipgap", pyofig.ConfigValue(
                        description="lagranger iterk mipgap (default None)",
                        domain=float,
                        default=None)).declare_as_argument()

    global_config.declare("lagranger-rho-rescale-factors-json", pyofig.ConfigValue(
                        description="json file: rho rescale factors (default None)",
                        domain=str,
                        default=None)).declare_as_argument()

    


def xhatlooper_args():
    
    global_config.declare('with-xhatlooper', pyofig.ConfigValue(
                          description="have an xhatlooper spoke (default)",
                          domain=bool,
                          default=False)).declare_as_argument()

    global_config.declare("xhat-scen-limit", pyofig.ConfigValue(
                        description="scenario limit xhat looper to try (default 3)",
                        domain=int,
                        default=3)).declare_as_argument()

    


def xhatshuffle_args():
    
    global_config.declare('with-xhatshuffle', pyofig.ConfigValue(
                          description="have an xhatshuffle spoke (default)",
                          domain=bool,
                          default=False)).declare_as_argument()
    
    global_config.declare('add-reversed-shuffle', pyofig.ConfigValue(
                        description="using also the reversed shuffling (multistage only, default True)",
                        dest = 'add_reversed_shuffle',
                          domain=bool,
                          default=False)).declare_as_argument()
    
    global_config.declare('xhatshuffle-iter-step', pyofig.ConfigValue(
                        description="step in shuffled list between 2 scenarios to try (default None)",
                        domain=int,
                        default=None)).declare_as_argument()

    


def xhatspecific_args():
    # we will not try to get the specification from the command line
    
    global_config.declare('with-xhatspecific', pyofig.ConfigValue(
                          description="have an xhatspecific spoke (default)",
                          domain=bool,
                          default=False)).declare_as_argument()

    


def xhatlshaped_args():
    # we will not try to get the specification from the command line
    
    global_config.declare('with-xhatlshaped', pyofig.ConfigValue(
                          description="have an xhatlshaped spoke (default)",
                          domain=bool,
                          default=False)).declare_as_argument()

    

def slamup_args():
    # we will not try to get the specification from the command line
    
    global_config.declare('with-slamup', pyofig.ConfigValue(
                        description="have an slamup spoke (default)",
                          domain=bool,
                          default=False)).declare_as_argument()

    

def slamdown_args():
    # we will not try to get the specification from the command line
    
    global_config.declare('with-slamdown', pyofig.ConfigValue(
                        description="have an slamdown spoke (default)",
                          domain=bool,
                          default=False)).declare_as_argument()

    

def cross_scenario_cuts_args():
    # we will not try to get the specification from the command line
    
    global_config.declare('with-cross-scenario-cuts', pyofig.ConfigValue(
                          description="have a cross scenario cuts spoke (default)",
                          domain=bool,
                          default=False)).declare_as_argument()

    global_config.declare("cross-scenario-iter-cnt", pyofig.ConfigValue(
                          description="cross scen check bound improve iterations "
                          "(default 4)",
                          domain=int,
                          default=4)).declare_as_argument()

    global_config.declare("eta-bounds-mipgap", pyofig.ConfigValue(
                          description="mipgap for determining eta bounds for cross "
                          "scenario cuts (default 0.01)",
                          domain=float,
                          default=0.01)).declare_as_argument()

def add_to_config(name, description, domain, default,
                  argparse=True,
                  complain=False):
    # add an arg to the global_config dict
    if name in global_confict:
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

def create_parser(progname=None):
    if len(global_config) == 0:
        raise RuntimeError("create parser called before global_config is populated")
    parser = argparse.ArgumentParser(progname, conflict_handler="resolve")
    global_config.initialize_argparse(parser)
    return parser
                                  
if __name__ == "__main__":
    # a place for ad hoc testing by developers
    popular_args() # populates global_config
    global_config.display()
    for i,j in global_config.items():
        print(i, j)
    print(dir(global_config))
    print(global_config._all_slots)
    print(global_config._domain)
    print(f"{global_config['max-iterations'] =}")

    parser = create_parser("tester")
    parser.add_argument(
            "num_scens", help="Number of scenarios", type=int,
        )

    args=parser.parse_args(['3', '--max-iterations', '99', '--solver-name', 'cplex'])

    print(f"{args.num_scens =}")
    
    args = global_config.import_argparse(args)
    
    global_config.display()    

    #parser.parse_args(['--help'])




    
