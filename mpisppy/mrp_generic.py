###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Generic driver for sequential sampling (MRP) in mpi-sppy.
# Analogous to generic_cylinders.py but for the multiple replication procedure.

import sys
import numpy as np

from mpisppy import MPI
from mpisppy.generic.parsing import model_fname, load_module, add_decomp_args
from mpisppy.generic.mrp import mrp_args, do_mrp

import mpisppy.utils.config as config


def parse_mrp_args(m):
    """Parse CLI args for sequential sampling given the model module m.

    MRP shares the full decomposition arg set with generic_cylinders via
    add_decomp_args, so that --xhat-method=cylinders accepts the same command
    line as generic_cylinders.  It adds the sequential sampling args and skips
    the args specific to generic_cylinders (mmw, admm).

    Returns a Config object.
    """
    cfg = config.Config()

    # Bundling / pickling args (needed when --xhat-method=cylinders reaches
    # name_lists, which reads cfg.unpickle_bundles_dir / scenarios_per_bundle).
    cfg.proper_bundle_config()
    cfg.pickle_scenarios_config()
    cfg.pre_pickle_args()

    cfg.add_to_config(name="module_name",
                      description="Name of the file that has the scenario creator, etc.",
                      domain=str,
                      default=None,
                      argparse=True)

    assert hasattr(m, "inparser_adder"), \
        "The model file must have an inparser_adder function"
    m.inparser_adder(cfg)

    cfg.add_to_config(name="solution_base_name",
                      description="Base name for solution output files (default None)",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="write_scenario_lp_mps_files_dir",
                      description="Directory for LP/MPS files (default None)",
                      domain=str,
                      default=None)

    # EF solver specs (always needed, since gap estimation uses EF)
    cfg.EF_base()

    # Sequential sampling args
    mrp_args(cfg)

    # Full hub-and-spoke decomposition args, shared with generic_cylinders, so
    # --xhat-method=cylinders works end-to-end with the same flags.
    add_decomp_args(cfg)

    cfg.parse_command_line(f"mpi-sppy MRP for {cfg.module_name}")

    return cfg


##########################################################################
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("The python model file module name (no .py) must be given.")
        print("usage, e.g.: python -m mpisppy.mrp_generic --module-name farmer"
              " --num-scens 3 --solver-name cplex --stopping-criterion BM")
        quit()

    fname = model_fname()
    module = load_module(fname)

    cfg = parse_mrp_args(module)

    # Perhaps use an object as the so-called module.
    if hasattr(module, "get_mpisppy_helper_object"):
        module = module.get_mpisppy_helper_object(cfg)

    result = do_mrp(fname, module, cfg)

    global_rank = MPI.COMM_WORLD.Get_rank()
    if global_rank == 0:
        print("\n===== MRP Results =====")
        print(f"Iterations (T): {result['T']}")
        print(f"Confidence interval on gap: {result['CI']}")
        if cfg.solution_base_name is not None:
            xhat = result["Candidate_solution"]
            root_nonants = np.array(xhat["ROOT"])
            np.save(f"{cfg.solution_base_name}.npy", root_nonants)
            print(f"Wrote xhat to {cfg.solution_base_name}.npy")
