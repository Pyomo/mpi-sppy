
###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Utilities to support formation and use of "proper" bundles

# NOTE: if/because we require the bundles to consume entire
#       second stage tree nodes, the resulting problem is two stage.
#  This adds complications when working with multi-stage problems.

# BTW: ssn (not in this repo) uses this as of March 2022

import os
import dill  

def dill_pickle(model, fname):
    """ serialize model using dill to file name"""
    # global_toc(f"about to pickle to {fname}")
    with open(fname, "wb") as f:
        dill.dump(model, f)
    # global_toc(f"done with pickle {fname}")


def dill_unpickle(fname):
    """ load a model from fname"""
    
    # global_toc(f"about to unpickle {fname}")
    with open(fname, "rb") as f:
        m = dill.load(f)
    # global_toc(f"done with unpickle {fname}")
    return m


def check_args(cfg):
    """ make sure the pickle bundle args make sense"""
    assert(cfg.pickle_bundles_dir is None or cfg.unpickle_bundles_dir is None)
    if cfg.proper_no_files:
        assert cfg.pickle_bundles_dir is None and cfg.unpickle_bundles_dir is None, "For proper bundles with no files, do not specify a directory"
    if cfg.scenarios_per_bundle is None:
        raise RuntimeError("For proper bundles, --scenarios-per-bundle must be specified")
    if cfg.get("bundles_per_rank") is not None and cfg.bundles_per_rank != 0:
        raise RuntimeError("For proper bundles, --scenarios-per-bundle must be specified "
                           "and --bundles-per-rank cannot be")
    if cfg.unpickle_bundles_dir is not None and not os.path.isdir(cfg.unpickle_bundles_dir):
        raise RuntimeError(f"Directory to load pickle files from not found: {cfg.unpickle_bundles_dir}")

def have_proper_bundles(cfg):
    """ boolean to indicate we have pickled bundles"""
    return (hasattr(cfg, "pickle_bundles_dir") and cfg.pickle_bundles_dir is not None)\
       or (hasattr(cfg, "unpickle_bundles_dir") and cfg.unpickle_bundles_dir is not None)
