# This software is distributed under the 3-clause BSD License.
# Utilities to support formation and use of "proper" bundles

# NOTE: if/because we require the bundles to consume entire
#       second stage tree nodes, the resulting problem is two stage.
#  This adds complications when working with multi-stage problems.

# BTW: ssn (not in this repo) uses this as of March 2022

import os
import dill  
from mpisppy import global_toc

def dill_pickle(model, fname):
    """ serialize model using dill to file name"""
    global_toc(f"about to pickle to {fname}")
    with open(fname, "wb") as f:
        dill.dump(model, f)
    global_toc(f"done with pickle {fname}")


def dill_unpickle(fname):
    """ load a model from fname"""
    
    global_toc(f"about to unpickle {fname}")
    with open(fname, "rb") as f:
        m = dill.load(f)
    global_toc(f"done with unpickle {fname}")
    return m


def pickle_bundle_parser(parser):
    """ Add command line options for creation and use of "proper" bundles"""
    parser.add_argument('--pickle-bundles-dir',
                        help="Write bundles to a dill pickle files in this dir (default None)",
                        dest='pickle_bundles_dir',
                        type=str,
                        default=None)
    
    parser.add_argument('--unpickle-bundles-dir',
                        help="Read bundles from a dill pickle files in this dir; (default None)",
                        dest='unpickle_bundles_dir',
                        type=str,
                        default=None)
    parser.add_argument("--scenarios-per-bundle",
                        help="Used for `proper` bundles only (default None)",
                        dest="scenarios_per_bundle",
                        type=int,
                        default=None)
    return parser

def check_args(args):
    """ make sure the pickle bundle args make sense"""
    assert(args.pickle_bundles_dir is None or args.unpickle_bundles_dir is None)
    if args.scenarios_per_bundle is None:
        raise RuntimeError("For proper bundles, --scenarios-per-bundle must be specified")
    if args.bundles_per_rank is not None and args.bundles_per_rank != 0:
        raise RuntimeError("For proper bundles, --scenarios-per-bundle must be specified "
                           "and --bundles-per-rank cannot be")
    if args.pickle_bundles_dir is not None and not os.path.isdir(args.pickle_bundles_dir):
        raise RuntimeError(f"Directory to pickle into not found: {args.pickle_bundles_dir}")
    if args.unpickle_bundles_dir is not None and not os.path.isdir(args.unpickle_bundles_dir):
        raise RuntimeError(f"Directory to load pickle files from not found: {args.unpickle_bundles_dir}")

def have_proper_bundles(args):
    """ boolean to indicate we have pickled bundles"""
    return (hasattr(args, "pickle_bundles_dir") and args.pickle_bundles_dir is not None)\
       or (hasattr(args, "unpickle_bundles_dir") and args.unpickle_bundles_dir is not None)
