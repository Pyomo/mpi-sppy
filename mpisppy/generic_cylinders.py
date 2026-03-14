###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Generic cylinder driver for mpi-sppy.
# Adapted from run_all.py by dlw August 2024.
# Refactored into mpisppy/generic/ package March 2026.

import sys

from mpisppy import MPI

# Re-export public API for backwards compatibility
from mpisppy.generic.decomp import do_decomp  # noqa: F401
from mpisppy.generic.mmw import do_mmw  # noqa: F401

from mpisppy.generic.parsing import model_fname, load_module, parse_args, proper_bundles
from mpisppy.generic.ef import do_EF
from mpisppy.generic.scenario_io import (write_bundles, write_scenarios,
                                          read_pickled_scenario,
                                          write_scenario_lp_mps_files_only)
from mpisppy.generic.mmw import mmw_requested


##########################################################################
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("The python model file module name (no .py) must be given.")
        print("usage, e.g.: python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --help")
        quit()
    fname = model_fname()
    module = load_module(fname)

    cfg = parse_args(module)

    # Perhaps use an object as the so-called module.
    if hasattr(module, "get_mpisppy_helper_object"):
        module = module.get_mpisppy_helper_object(cfg)

    bundle_wrapper = None  # the default
    if proper_bundles(cfg):
        import mpisppy.utils.proper_bundler as proper_bundler
        bundle_wrapper = proper_bundler.ProperBundler(module)
        bundle_wrapper.set_bunBFs(cfg)
        scenario_creator = bundle_wrapper.scenario_creator
        # The scenario creator is wrapped, so these kw_args will not go the original
        # creator (the kw_creator will keep the original args)
        scenario_creator_kwargs = bundle_wrapper.kw_creator(cfg)
    elif cfg.unpickle_scenarios_dir is not None:
        # So reading pickled scenarios cannot be composed with proper bundles
        scenario_creator = read_pickled_scenario
        scenario_creator_kwargs = {"cfg": cfg}
    else:  # the most common case
        scenario_creator = module.scenario_creator
        scenario_creator_kwargs = module.kw_creator(cfg)

    # ADMM setup: wraps scenario_creator and attaches variable_probability to cfg
    if cfg.get("admm", ifmissing=False) or cfg.get("stoch_admm", ifmissing=False):
        from mpisppy.generic.admm import setup_admm, setup_stoch_admm, \
            _count_cylinders, _check_admm_compatibility
        _check_admm_compatibility(cfg)
        n_cylinders = _count_cylinders(cfg)
        if cfg.admm:
            scenario_creator, scenario_creator_kwargs, _, _ = \
                setup_admm(module, cfg, n_cylinders)
        else:
            scenario_creator, scenario_creator_kwargs, _, _ = \
                setup_stoch_admm(module, cfg, n_cylinders)

    assert hasattr(module, "scenario_denouement"), "The model file must have a scenario_denouement function"
    scenario_denouement = module.scenario_denouement

    if cfg.pickle_bundles_dir is not None:
        global_comm = MPI.COMM_WORLD
        write_bundles(module,
                      cfg,
                      scenario_creator,
                      scenario_creator_kwargs,
                      global_comm)
    elif cfg.pickle_scenarios_dir is not None:
        global_comm = MPI.COMM_WORLD
        write_scenarios(module,
                        cfg,
                        scenario_creator,
                        scenario_creator_kwargs,
                        scenario_denouement,
                        global_comm)
    elif cfg.write_scenario_lp_mps_files_dir is not None:
        write_scenario_lp_mps_files_only(
            module,
            cfg,
            scenario_creator,
            scenario_creator_kwargs,
            scenario_denouement,
            bundle_wrapper=bundle_wrapper,
        )
    elif cfg.EF:
        do_EF(module, cfg, scenario_creator, scenario_creator_kwargs,
              scenario_denouement, bundle_wrapper=bundle_wrapper)
        if mmw_requested(cfg) and cfg.get("mmw_xhat_input_file_name") is not None:
            do_mmw(fname, cfg)
    else:
        wheel = do_decomp(module, cfg, scenario_creator, scenario_creator_kwargs,
                          scenario_denouement, bundle_wrapper=bundle_wrapper)
        if mmw_requested(cfg):
            do_mmw(fname, cfg, wheel=wheel)
