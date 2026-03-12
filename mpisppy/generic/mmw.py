###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""MMW confidence interval computation for generic_cylinders."""

import mpisppy.utils.sputils as sputils
from mpisppy import global_toc, MPI


def mmw_requested(cfg):
    """Return True iff the required MMW args (num_batches, batch_size, start) are all set.

    Raises ValueError if only some of the three options are provided.
    """
    mmw_opts = {
        "mmw_num_batches": cfg.get("mmw_num_batches"),
        "mmw_batch_size": cfg.get("mmw_batch_size"),
        "mmw_start": cfg.get("mmw_start"),
    }
    given = [k for k, v in mmw_opts.items() if v is not None]
    if 0 < len(given) < 3:
        missing = [k for k in mmw_opts if k not in given]
        raise ValueError(
            f"Partial MMW configuration: {given} provided but {missing} missing. "
            "Either provide all three (mmw_num_batches, mmw_batch_size, mmw_start) "
            "or none of them."
        )
    return len(given) == 3


def do_mmw(module_fname, cfg, wheel=None):
    """Run an MMW confidence interval computation after the main algorithm.

    Args:
        module_fname (str): module name or path (e.g. 'farmer' or '/path/to/farmer')
        cfg (Config): config with mmw_* and EF_* options; modified in place
        wheel (WheelSpinner or None): if None, cfg.mmw_xhat_input_file_name must be set
    """
    import os
    import tempfile
    from mpisppy.confidence_intervals import mmw_ci
    from mpisppy.confidence_intervals import ciutils

    # MMWConfidenceIntervals expects an importable module name, not a file path.
    # load_module (in parsing.py) already added the directory to sys.path,
    # so the basename is importable.
    module_fname = os.path.basename(module_fname)

    global_comm = MPI.COMM_WORLD
    global_rank = global_comm.Get_rank()

    xhat_fname = cfg.get("mmw_xhat_input_file_name")
    if xhat_fname is not None:
        xhat = ciutils.read_xhat(xhat_fname)
    else:
        if wheel is None:
            raise RuntimeError(
                "do_mmw: mmw_xhat_input_file_name must be set when no wheel is provided"
            )
        # Write the wheel's best xhat to a temporary file readable by all ranks.
        if global_rank == 0:
            tmp_path = tempfile.mktemp(suffix=".npy")
        else:
            tmp_path = None
        tmp_path = global_comm.bcast(tmp_path, root=0)

        wheel.write_first_stage_solution(
            tmp_path,
            first_stage_solution_writer=sputils.first_stage_nonant_npy_serializer,
        )
        global_comm.Barrier()

        if not os.path.exists(tmp_path):
            global_toc("MMW CI skipped: no feasible solution found by the main algorithm.")
            return
        xhat = ciutils.read_xhat(tmp_path)
        if global_rank == 0:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Default the EF solver to the main solver when not explicitly set.
    if cfg.get("EF_solver_name") is None:
        cfg.quick_assign("EF_solver_name", str, cfg.solver_name)

    # Tell MMW whether this is a 2-stage or multi-stage problem.
    if cfg.get("branching_factors") is not None:
        cfg.quick_assign("EF_mstage", bool, True)
    else:
        cfg.quick_assign("EF_2stage", bool, True)

    mmw = mmw_ci.MMWConfidenceIntervals(
        module_fname,
        cfg,
        xhat,
        cfg.mmw_num_batches,
        batch_size=cfg.mmw_batch_size,
        start=cfg.mmw_start,
        verbose=True,
    )
    result = mmw.run()
    global_toc(f"MMW CI result: {result}")
    return result
