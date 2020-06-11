# This software is distributed under the 3-clause BSD License.
# set up the most common parser args for mpi-sppy examples
""" NOTE TO NEW USERS: just using these parsers will not, itself, do anything.
    You have to use the values when you create the dictionaries that are passed
    to spin_the_wheel. Further note that not all examples use all values
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

    parser.add_argument('--with-display-timing',
                        help="display timing at each iteration",
                        dest='with_display_timing',
                        action='store_true')
    parser.add_argument('--no-display-timing',
                        help="do not display timing at each iteration (default)",
                        dest='with_display_timing',
                        action='store_false')
    parser.set_defaults(with_display_timing=False)

    parser.add_argument('--with-display-progress',
                        help="display progress at each iteration",
                        dest='with_display_progress',
                        action='store_true')
    parser.add_argument('--no-display-progress',
                        help="do not display progress at each iteration (default)",
                        dest='with_display_timing',
                        action='store_false')
    parser.set_defaults(with_display_timing=False)

    parser.add_argument("--intra-hub-conv-thresh",
                        help="Within hub convergence threshold (default 0)",
                        dest="intra_hub_conv_thresh",
                        type=float,
                        default=0)

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

def make_multistage_parser(progname=None):
    # make a parser for the program named progname
    # NOTE: if you want abbreviations, override the arguments in your example
    # do not add abbreviations here.
    parser = argparse.ArgumentParser(prog=progname, conflict_handler="resolve")

    # the default is intended more as an example than as a default
    parser.add_argument("--BFs",
                        help="Comma delimied branching factors (default 2,2)",
                        dest="BFs",
                        type=str,
                        default="2,2")
    parser = _common_args(parser)
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
                        help="absolute termination gap (default 8)",
                        dest="abs_gap",
                        type=float,
                        default=8.)

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

    return parser

def lagrangian_args(inparser):
    parser = inparser
    parser.add_argument('--with-lagrangian',
                        help="have an lagrangian spoke (default)",
                        dest='with_lagrangian',
                        action='store_true')
    parser.add_argument('--no-lagrangian',
                        help="do not have an lagrangian spoke",
                        dest='with_lagrangian',
                        action='store_false')
    parser.set_defaults(with_lagrangian=True)

    return parser


def xhatlooper_args(inparser):
    parser = inparser
    parser.add_argument('--with-xhatlooper',
                        help="have an xhatlooper spoke",
                        dest='with_xhatlooper',
                        action='store_true')
    parser.add_argument('--no-xhatlooper',
                        help="do not have an xhatlooper spoke (default)",
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
    parser.set_defaults(with_xhatspecific=True)

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
