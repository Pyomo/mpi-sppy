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


def pickle_bundle_parser(cfg):
    """ Add command line options for creation and use of "proper" bundles
    args:
        cfg (Config): the Config object to which we add"""
    cfg.add_to_config('pickle_bundles_dir',
                        description="Write bundles to a dill pickle files in this dir (default None)",
                        domain=str,
                        default=None)
    
    cfg.add_to_config('unpickle_bundles_dir',
                        description="Read bundles from a dill pickle files in this dir; (default None)",
                        domain=str,
                        default=None)
    cfg.add_to_config("scenarios_per_bundle",
                        description="Used for `proper` bundles only (default None)",
                        domain=int,
                        default=None)

def check_args(cfg):
    """ make sure the pickle bundle args make sense"""
    assert(cfg.pickle_bundles_dir is None or cfg.unpickle_bundles_dir is None)
    if cfg.scenarios_per_bundle is None:
        raise RuntimeError("For proper bundles, --scenarios-per-bundle must be specified")
    if cfg.get("bundles_per_rank") is not None and cfg.bundles_per_rank != 0:
        raise RuntimeError("For proper bundles, --scenarios-per-bundle must be specified "
                           "and --bundles-per-rank cannot be")
    if cfg.pickle_bundles_dir is not None and not os.path.isdir(cfg.pickle_bundles_dir):
        raise RuntimeError(f"Directory to pickle into not found: {cfg.pickle_bundles_dir}")
    if cfg.unpickle_bundles_dir is not None and not os.path.isdir(cfg.unpickle_bundles_dir):
        raise RuntimeError(f"Directory to load pickle files from not found: {cfg.unpickle_bundles_dir}")

def have_proper_bundles(cfg):
    """ boolean to indicate we have pickled bundles"""
    return (hasattr(cfg, "pickle_bundles_dir") and cfg.pickle_bundles_dir is not None)\
       or (hasattr(cfg, "unpickle_bundles_dir") and cfg.unpickle_bundles_dir is not None)
