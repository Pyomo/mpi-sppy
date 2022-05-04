# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# set up the most common parser args for mpi-sppy examples
""" NOTE TO NEW USERS: just using these parsers will not, itself, do anything.
    You have to use the values when you create the dictionaries that are passed
    to WheelSpinner. Further note that not all examples use all values
    in the parsers.
"""
import argparse

def _common_args(inparser):
    # NOTE: if you want abbreviations, override the arguments in your example
    # do not add abbreviations here.

    parser = inparser
    parser.add_argument("--max-iterations",
                        help="ph max-iterations (default 1)",
                        dest="max_iterations",
                        type=int,
                        default=1)

    parser.add_argument("--solver-name",
                        help = "solver name (default gurobi)",
                        dest="solver_name",
                        type = str,
                        default="gurobi")

    parser.add_argument("--seed",
                        help="Seed for random numbers (default is 1134)",
                        dest="seed",
                        type=int,
                        default=1134)

    parser.add_argument("--default-rho",
                        help="Global rho for PH (default None)",
                        dest="default_rho",
                        type=float,
                        default=None)

    parser.add_argument("--bundles-per-rank",
                        help="bundles per rank (default 0 (no bundles))",
                        dest="bundles_per_rank",
                        type=int,
                        default=0)                

    parser.add_argument('--with-verbose',
                        help="verbose output",
                        dest='with_verbose',
                        action='store_true')
    parser.add_argument('--no-verbose',
                        help="do not verbose output (default)",
                        dest='with_verbose',
                        action='store_false')
    parser.set_defaults(with_verbose=False)

    parser.add_argument('--with-display-progress',
                        help="display progress at each iteration",
                        dest='with_display_progress',
                        action='store_true')
    parser.add_argument('--no-display-progress',
                        help="do not display progress at each iteration (default)",
                        dest='with_display_progress',
                        action='store_false')
    parser.set_defaults(with_display_progress=False)

    parser.add_argument('--with-display-convergence-detail',
                        help="display non-anticipative variable convergence statistics at each iteration",
                        dest='with_display_convergence_detail',
                        action='store_true')
    parser.add_argument('--no-display-convergence-detail',
                        help="do not display non-anticipative variable convergence statistics at each iteration (default)",
                        dest='with_display_convergence_detail',
                        action='store_false')
    parser.set_defaults(with_display_convergence_detail=False)    

    parser.add_argument("--max-solver-threads",
                        help="Limit on threads per solver (default None)",
                        dest="max_solver_threads",
                        type=int,
                        default=None)

    parser.add_argument("--intra-hub-conv-thresh",
                        help="Within hub convergence threshold (default 1e-10)",
                        dest="intra_hub_conv_thresh",
                        type=float,
                        default=1e-10)

    parser.add_argument("--trace-prefix",
                        help="Prefix for bound spoke trace files. If None "
                             "bound spoke trace files are not written.",
                        dest="trace_prefix",
                        type=str,
                        default=None)

    parser.add_argument("--with-tee-rank0-solves",
                        help="Some cylinders support tee of rank 0 solves."
                        "(With multiple cylinder this could be confusing.)",
                        dest="tee_rank0_solves",
                        action='store_true')
    parser.add_argument("--no-tee-rank0-solves",
                        help="Some cylinders support tee of rank 0 solves.",
                        dest="tee_rank0_solves",
                        action='store_false')
    parser.set_defaults(tee_rank0_solves=False)

    parser.add_argument("--auxilliary",
                        help="Free text for use by hackers (default '').",
                        dest="auxilliary",
                        type=str,
                        default='')

    parser.add_argument("--linearize-binary-proximal-terms",
                        help="For PH, linearize the proximal terms for "
                        "all binary nonanticipative variables",
                        dest="linearize_binary_proximal_terms",
                        action='store_true')

    parser.add_argument("--linearize-proximal-terms",
                        help="For PH, linearize the proximal terms for "
                        "all nonanticipative variables",
                        dest="linearize_proximal_terms",
                        action='store_true')

    parser.add_argument("--proximal-linearization-tolerance",
                        help="For PH, when linearizing proximal terms, "
                        "a cut will be added if the proximal term approximation "
                        "is looser than this value (default 1e-1)",
                        dest="proximal_linearization_tolerance",
                        type=float,
                        default=1.e-1)

    return parser
    
def make_parser(progname=None, num_scens_reqd=False):
    # make a parser for the program named progname
    # NOTE: if you want abbreviations, override the arguments in your example
    # do not add abbreviations here.
    parser = argparse.ArgumentParser(prog=progname, conflict_handler="resolve")

    if num_scens_reqd:
        parser.add_argument(
            "num_scens", help="Number of scenarios", type=int
        )
    else:
        parser.add_argument(
            "--num-scens",
            help="Number of scenarios (default None)",
            dest="num_scens",
            type=int,
            default=None,
        )
    parser = _common_args(parser)
    return parser

def _basic_multistage(progname=None, num_scens_reqd=False):
    parser = argparse.ArgumentParser(prog=progname, conflict_handler="resolve")

    parser.add_argument("--branching-factors",
                        help="Spaces delimited branching factors (e.g., 2 2)",
                        dest="branching_factors",
                        nargs="*",
                        type=int,
                        default=None)
        
    return parser


def make_multistage_parser(progname=None):
    # make a parser for the program named progname
    # NOTE: if you want abbreviations, override the arguments in your example
    # do not add abbreviations here.
    parser = _basic_multistage(progname=None)
    parser = _common_args(parser)
    return parser

#### EF ####
def make_EF2_parser(progname=None, num_scens_reqd=False):
    # create a parser just for EF two-stage (does not call _common_args)
    # NOTE: if you want abbreviations, override the arguments in your example
    # do not add abbreviations here.
    parser = argparse.ArgumentParser(prog=progname, conflict_handler="resolve")

    if num_scens_reqd:
        parser.add_argument(
            "num_scens", help="Number of scenarios", type=int
        )
    else:
        parser.add_argument(
            "--num-scens",
            help="Number of scenarios (default None)",
            dest="num_scens",
            type=int,
            default=None,
        )
        
    parser.add_argument("--EF-solver-name",
                        help = "solver name (default gurobi)",
                        dest="EF_solver_name",
                        type = str,
                        default="gurobi")


    parser.add_argument("--EF-mipgap",
                        help="mip gap option for the solver if needed (default None)",
                        dest="EF_mipgap",
                        type=float,
                        default=None)
    return parser

def make_EF_multistage_parser(progname=None, num_scens_reqd=False):
    # create a parser just for EF multi-stage (does not call _common_args)
    # NOTE: if you want abbreviations, override the arguments in your example
    # do not add abbreviations here.
    parser = _basic_multistage(progname=None)
    
    if num_scens_reqd:
        parser.add_argument(
            "num_scens", help="Number of scenarios", type=int
        )
    else:
        parser.add_argument(
            "--num-scens",
            help="Number of scenarios (default None)",
            dest="num_scens",
            type=int,
            default=None,
        )
        
    parser.add_argument("--EF-solver-name",
                        help = "solver name (default gurobi)",
                        dest="EF_solver_name",
                        type = str,
                        default="gurobi")


    parser.add_argument("--EF-mipgap",
                        help="mip gap option for the solver if needed (default None)",
                        dest="EF_mipgap",
                        type=float,
                        default=None)
    return parser
##### common additions to the command line #####

def two_sided_args(inparser):
    # add commands to inparser and also return the result
    parser = inparser
    parser.add_argument("--rel-gap",
                        help="relative termination gap (default 0.05)",
                        dest="rel_gap",
                        type=float,
                        default=0.05)

    parser.add_argument("--abs-gap",
                        help="absolute termination gap (default 0)",
                        dest="abs_gap",
                        type=float,
                        default=0.)
    
    parser.add_argument("--max-stalled-iters",
                        help="maximum iterations with no reduction in gap (default 100)",
                        dest="max_stalled_iters",
                        type=int,
                        default=100)

    return parser

def mip_options(inparser):
    parser = inparser
    parser.add_argument("--iter0-mipgap",
                        help="mip gap option for iteration 0 (default None)",
                        dest="iter0_mipgap",
                        type=float,
                        default=None)

    parser.add_argument("--iterk-mipgap",
                        help="mip gap option non-zero iterations (default None)",
                        dest="iterk_mipgap",
                        type=float,
                        default=None)
    return parser

def aph_args(inparser):
    parser = inparser
    parser.add_argument('--aph-gamma',
                        help='Gamma parameter associated with asychronous projective hedging (default 1.0)',
                        dest='aph_gamma',
                        type=float,
                        default=1.0)
    parser.add_argument('--aph-nu',
                        help='Nu parameter associated with asychronous projective hedging (default 1.0)',
                        dest="aph_nu",
                        type=float,
                        default=1.0)
    parser.add_argument('--aph-frac-needed',
                        help='Fraction of sub-problems required before computing projective step (default 1.0)',
                        dest='aph_frac_needed',
                        type=float,
                        default=1.0)
    parser.add_argument('--aph-dispatch-frac',
                        help='Fraction of sub-problems to dispatch at each step of asychronous projective hedging (default 1.0)',
                        dest='aph_dispatch_frac',
                        type=float,
                        default=1.0)
    parser.add_argument('--aph-sleep-seconds',
                        help='Spin-lock sleep time for APH (default 0.01)',
                        dest='aph_sleep_seconds',
                        type=float,
                        default=0.01)    
    return parser    

def fixer_args(inparser):
    parser = inparser
    parser.add_argument('--with-fixer',
                        help="have an integer fixer extension (default)",
                        dest='with_fixer',
                        action='store_true')
    parser.add_argument('--no-fixer',
                        help="do not have an integer fixer extension",
                        dest='with_fixer',
                        action='store_false')
    parser.set_defaults(with_fixer=True)

    parser.add_argument("--fixer-tol",
                        help="fixer bounds tolerance  (default 1e-4)",
                        dest="fixer_tol",
                        type=float,
                        default=1e-2)
    return parser


def fwph_args(inparser):
    parser = inparser
    parser.add_argument('--with-fwph',
                        help="have an fwph spoke (default)",
                        dest='with_fwph',
                        action='store_true')
    parser.add_argument('--no-fwph',
                        help="do not have an fwph spoke",
                        dest='with_fwph',
                        action='store_false')
    parser.set_defaults(with_fwph=True)

    parser.add_argument("--fwph-iter-limit",
                        help="maximum fwph iterations (default 10)",
                        dest="fwph_iter_limit",
                        type=int,
                        default=10)

    parser.add_argument("--fwph-weight",
                        help="fwph weight (default 0)",
                        dest="fwph_weight",
                        type=float,
                        default=0.0)

    parser.add_argument("--fwph-conv-thresh",
                        help="fwph convergence threshold  (default 1e-4)",
                        dest="fwph_conv_thresh",
                        type=float,
                        default=1e-4)

    parser.add_argument("--fwph-stop-check-tol",
                        help="fwph tolerance for Gamma^t (default 1e-4)",
                        dest="fwph_stop_check_tol",
                        type=float,
                        default=1e-4)

    parser.add_argument("--fwph-mipgap",
                        help="mip gap option FW subproblems iterations (default None)",
                        dest="fwph_mipgap",
                        type=float,
                        default=None)

    return parser

def lagrangian_args(inparser):
    parser = inparser
    parser.add_argument('--with-lagrangian',
                        help="have a lagrangian spoke (default)",
                        dest='with_lagrangian',
                        action='store_true')
    parser.add_argument('--no-lagrangian',
                        help="do not have a lagrangian spoke",
                        dest='with_lagrangian',
                        action='store_false')
    parser.set_defaults(with_lagrangian=True)

    parser.add_argument("--lagrangian-iter0-mipgap",
                        help="lgr. iter0 solver option mipgap (default None)",
                        dest="lagrangian_iter0_mipgap",
                        type=float,
                        default=None)

    parser.add_argument("--lagrangian-iterk-mipgap",
                        help="lgr. iterk solver option mipgap (default None)",
                        dest="lagrangian_iterk_mipgap",
                        type=float,
                        default=None)

    return parser


def lagranger_args(inparser):
    parser = inparser
    parser.add_argument('--with-lagranger',
                        help="have a special lagranger spoke (default)",
                        dest='with_lagranger',
                        action='store_true')
    parser.add_argument('--no-lagranger',
                        help="do not have a special lagranger spoke",
                        dest='with_lagranger',
                        action='store_false')
    parser.set_defaults(with_lagranger=True)

    parser.add_argument("--lagranger-iter0-mipgap",
                        help="lagranger iter0 mipgap (default None)",
                        dest="lagranger_iter0_mipgap",
                        type=float,
                        default=None)

    parser.add_argument("--lagranger-iterk-mipgap",
                        help="lagranger iterk mipgap (default None)",
                        dest="lagranger_iterk_mipgap",
                        type=float,
                        default=None)

    parser.add_argument("--lagranger-rho-rescale-factors-json",
                        help="json file: rho rescale factors (default None)",
                        dest="lagranger_rho_rescale_factors_json",
                        type=str,
                        default=None)

    return parser


def xhatlooper_args(inparser):
    parser = inparser
    parser.add_argument('--with-xhatlooper',
                        help="have an xhatlooper spoke (default)",
                        dest='with_xhatlooper',
                        action='store_true')
    parser.add_argument('--no-xhatlooper',
                        help="do not have an xhatlooper spoke",
                        dest='with_xhatlooper',
                        action='store_false')
    parser.set_defaults(with_xhatlooper=False)
    parser.add_argument("--xhat-scen-limit",
                        help="scenario limit xhat looper to try (default 3)",
                        dest="xhat_scen_limit",
                        type=int,
                        default=3)

    return parser


def xhatshuffle_args(inparser):
    parser = inparser
    parser.add_argument('--with-xhatshuffle',
                        help="have an xhatshuffle spoke (default)",
                        dest='with_xhatshuffle',
                        action='store_true')
    parser.add_argument('--no-xhatshuffle',
                        help="do not have an xhatshuffle spoke",
                        dest='with_xhatshuffle',
                        action='store_false')
    parser.set_defaults(with_xhatshuffle=True)
    parser.add_argument('--add-reversed-shuffle',
                        help="using also the reversed shuffling (multistage only, default True)",
                        dest = 'add_reversed_shuffle',
                        action='store_true')
    parser.set_defaults(add_reversed_shuffle=True)
    parser.add_argument('--xhatshuffle-iter-step',
                        help="step in shuffled list between 2 scenarios to try (default None)",
                        dest="xhatshuffle_iter_step",
                        type=int,
                        default=None)

    return parser


def xhatspecific_args(inparser):
    # we will not try to get the specification from the command line
    parser = inparser
    parser.add_argument('--with-xhatspecific',
                        help="have an xhatspecific spoke (default)",
                        dest='with_xhatspecific',
                        action='store_true')
    parser.add_argument('--no-xhatspecific',
                        help="do not have an xhatspecific spoke",
                        dest='with_xhatspecific',
                        action='store_false')
    parser.set_defaults(with_xhatspecific=False)

    return parser


def xhatlshaped_args(inparser):
    # we will not try to get the specification from the command line
    parser = inparser
    parser.add_argument('--with-xhatlshaped',
                        help="have an xhatlshaped spoke (default)",
                        dest='with_xhatlshaped',
                        action='store_true')
    parser.add_argument('--no-xhatlshaped',
                        help="do not have an xhatlshaped spoke",
                        dest='with_xhatlshaped',
                        action='store_false')
    parser.set_defaults(with_xhatlshaped=True)

    return parser

def slamup_args(inparser):
    # we will not try to get the specification from the command line
    parser = inparser
    parser.add_argument('--with-slamup',
                        help="have an slamup spoke (default)",
                        dest='with_slamup',
                        action='store_true')
    parser.add_argument('--no-slamup',
                        help="do not have an slamup spoke",
                        dest='with_slamup',
                        action='store_false')
    parser.set_defaults(with_slamup=True)

    return parser

def slamdown_args(inparser):
    # we will not try to get the specification from the command line
    parser = inparser
    parser.add_argument('--with-slamdown',
                        help="have an slamdown spoke (default)",
                        dest='with_slamdown',
                        action='store_true')
    parser.add_argument('--no-slamdown',
                        help="do not have an slamdown spoke",
                        dest='with_slamdown',
                        action='store_false')
    parser.set_defaults(with_slamdown=True)

    return parser

def cross_scenario_cuts_args(inparser):
    # we will not try to get the specification from the command line
    parser = inparser
    parser.add_argument('--with-cross-scenario-cuts',
                        help="have a cross scenario cuts spoke (default)",
                        dest='with_cross_scenario_cuts',
                        action='store_true')
    parser.add_argument('--no-cross-scenario-cuts',
                        help="do not have a cross scenario cuts spoke",
                        dest='with_cross_scenario_cuts',
                        action='store_false')
    parser.set_defaults(with_cross_scenario_cuts=True)

    parser.add_argument("--cross-scenario-iter-cnt",
                        help="cross scen check bound improve iterations "
                        "(default 4)",
                        dest="cross_scenario_iter_cnt",
                        type=int,
                        default=4)

    parser.add_argument("--eta-bounds-mipgap",
                        help="mipgap for determining eta bounds for cross "
                        "scenario cuts (default 0.01)",
                        dest="eta_bounds_mipgap",
                        type=float,
                        default=0.01)

    return parser
