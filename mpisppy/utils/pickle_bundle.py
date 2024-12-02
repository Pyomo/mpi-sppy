###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Utilities to support pickling and unpickling "proper" bundles
# This file also provides support for pickled scenarios

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
    """ Make sure the pickle bundle args make sense; this assumes the config
    has all the appropriate fields."""
    assert cfg.get("pickle_bundles_dir") is None or cfg.get("unpickle_bundles_dir") is None
    assert cfg.get("pickle_scenarios_dir") is None or cfg.get("unpickle_scenarios_dir") is None
    assert cfg.get("unpickle_scenarios_dir") is None or cfg.get("bundles_per_rank") != 0, "Unpickled scenarios in proper bundles are not supported"
    if cfg.get("bundles_per_rank") is not None and cfg.bundles_per_rank != 0:
        raise RuntimeError("For proper bundles, --scenarios-per-bundle must be specified "
                           "and --bundles-per-rank, which is for loose bundles, cannot be")
    if cfg.get("unpickle_bundles_dir") is not None and not os.path.isdir(cfg.unpickle_bundles_dir):
        raise RuntimeError(f"Directory to load pickled bundle files from not found: {cfg.unpickle_bundles_dir}")
    if cfg.get("unpickle_scenarios_dir") is not None and not os.path.isdir(cfg.unpickle_scenarios_dir):
        raise RuntimeError(f"Directory to load pickled scenarios files from not found: {cfg.unpickle_scenarios_dir}")
    

def have_proper_bundles(cfg):
    """ boolean to indicate we have pickled bundles"""
    return cfg.get("scenarios_per_bundle") is not None\
        and cfg.scenarios_per_bundle > 0
        

