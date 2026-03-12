###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Scenario/bundle pickling and LP/MPS file writing for generic_cylinders."""

import os

import mpisppy.utils.sputils as sputils
from mpisppy import global_toc, MPI

from mpisppy.generic.parsing import name_lists


def write_scenarios(module, cfg, scenario_creator, scenario_creator_kwargs,
                    scenario_denouement, comm):
    """Pickle scenarios to disk.

    Args:
        module: model module
        cfg: Config object
        scenario_creator: scenario creator function
        scenario_creator_kwargs: kwargs for scenario creator
        scenario_denouement: denouement function
        comm: MPI communicator
    """
    import mpisppy.utils.pickle_bundle as pickle_bundle
    import shutil
    assert hasattr(cfg, "num_scens")
    ScenCount = cfg.num_scens

    n_proc = comm.Get_size()
    my_rank = comm.Get_rank()
    avg = ScenCount / n_proc
    slices = [list(range(int(i * avg), int((i + 1) * avg))) for i in range(n_proc)]

    local_slice = slices[my_rank]
    my_start = local_slice[0]   # zero based
    inum = sputils.extract_num(module.scenario_names_creator(1)[0])

    local_scenario_names = module.scenario_names_creator(len(local_slice),
                                                         start=inum + my_start)
    if my_rank == 0:
        if os.path.exists(cfg.pickle_scenarios_dir):
            shutil.rmtree(cfg.pickle_scenarios_dir)
        os.makedirs(cfg.pickle_scenarios_dir)
    comm.Barrier()
    for sname in local_scenario_names:
        scen = scenario_creator(sname, **scenario_creator_kwargs)
        fname = os.path.join(cfg.pickle_scenarios_dir, sname+".pkl")
        pickle_bundle.dill_pickle(scen, fname)
    global_toc(f"Pickled Scenarios written to {cfg.pickle_scenarios_dir}")


def read_pickled_scenario(sname, cfg):
    """Read a single pickled scenario from disk.

    Args:
        sname (str): scenario name
        cfg: Config object with unpickle_scenarios_dir

    Returns:
        the unpickled scenario model
    """
    import mpisppy.utils.pickle_bundle as pickle_bundle
    fname = os.path.join(cfg.unpickle_scenarios_dir, sname+".pkl")
    scen = pickle_bundle.dill_unpickle(fname)
    return scen


def write_bundles(module, cfg, scenario_creator, scenario_creator_kwargs, comm):
    """Pickle bundles to disk.

    Args:
        module: model module
        cfg: Config object
        scenario_creator: scenario creator function
        scenario_creator_kwargs: kwargs for scenario creator
        comm: MPI communicator
    """
    import mpisppy.utils.pickle_bundle as pickle_bundle
    import shutil
    assert hasattr(cfg, "num_scens")
    ScenCount = cfg.num_scens
    bsize = int(cfg.scenarios_per_bundle)
    numbuns = ScenCount // bsize
    n_proc = comm.Get_size()
    my_rank = comm.Get_rank()

    if numbuns < n_proc:
        raise RuntimeError(
            "More MPI ranks (%d) supplied than needed given the number of bundles (%d) "
            % (n_proc, numbuns)
        )

    avg = numbuns / n_proc
    slices = [list(range(int(i * avg), int((i + 1) * avg))) for i in range(n_proc)]

    local_slice = slices[my_rank]
    # We need to know if scenarios (not bundles) are one-based.
    inum = sputils.extract_num(module.scenario_names_creator(1)[0])

    local_bundle_names = [f"Bundle_{bn*bsize+inum}_{(bn+1)*bsize-1+inum}" for bn in local_slice]

    if my_rank == 0:
        if os.path.exists(cfg.pickle_bundles_dir):
            shutil.rmtree(cfg.pickle_bundles_dir)
        os.makedirs(cfg.pickle_bundles_dir)
    comm.Barrier()
    for bname in local_bundle_names:
        bundle = scenario_creator(bname, **scenario_creator_kwargs)
        fname = os.path.join(cfg.pickle_bundles_dir, bname+".pkl")
        pickle_bundle.dill_pickle(bundle, fname)
    global_toc(f"Bundles written to {cfg.pickle_bundles_dir}")


def write_scenario_lp_mps_files_only(module, cfg, scenario_creator,
                                     scenario_creator_kwargs,
                                     scenario_denouement,
                                     bundle_wrapper=None):
    """Construct scenarios and write per-scenario LP/MPS + nonants JSON
    (and rho.csv using either scenario rhos or cfg.default_rho) WITHOUT solving.

    Args:
        module: model module
        cfg: Config object
        scenario_creator: scenario creator function
        scenario_creator_kwargs: kwargs for scenario creator
        scenario_denouement: denouement function
        bundle_wrapper: ProperBundler or None
    """
    from mpisppy.spbase import SPBase
    from mpisppy.extensions.scenario_lp_mps_files import Scenario_lp_mps_files

    all_scenario_names, all_nodenames = name_lists(module, cfg,
                                                   bundle_wrapper=bundle_wrapper)

    # SPBase builds the scenarios and scenario-tree bookkeeping; no solves happen here.
    sp_options = {
        "verbose": cfg.verbose,
        "toc": cfg.get("toc", True),
    }

    sp = SPBase(
        options=sp_options,
        all_scenario_names=all_scenario_names,
        scenario_creator=scenario_creator,
        scenario_denouement=scenario_denouement,
        all_nodenames=all_nodenames,
        mpicomm=MPI.COMM_WORLD,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    # Make SPBase look PH-like for this extension
    sp.local_subproblems = sp.local_scenarios
    sp.options["write_lp_mps_extension_options"] = {
        "write_scenario_lp_mps_files_dir": cfg.write_scenario_lp_mps_files_dir,
        "cfg": cfg,   # IMPORTANT: pass cfg so extension can use default_rho
    }

    ext = Scenario_lp_mps_files(sp)
    ext.pre_iter0()

    if sp.cylinder_rank == 0:
        global_toc(f"Wrote scenario lp/mps/nonants to {cfg.write_scenario_lp_mps_files_dir}")
