###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Sequential sampling (MRP) for mrp_generic.

This module provides model-agnostic sequential sampling using the
SeqSampling (two-stage) and IndepScens_SeqSampling (multi-stage)
classes from mpisppy.confidence_intervals.
"""

import os
import tempfile
import functools

import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgamator as amalgamator
import mpisppy.confidence_intervals.confidence_config as confidence_config
from mpisppy import global_toc, MPI


def mrp_args(cfg):
    """Register sequential sampling CLI arguments on cfg."""
    cfg.add_to_config("stopping_criterion",
                      description="BM (Bayraksan-Morton) or BPL (Bayraksan-Pierre-Louis)",
                      domain=str,
                      default="BM")
    cfg.add_to_config("mrp_max_iterations",
                      description="Safety cap on sequential sampling iterations (default 200)",
                      domain=int,
                      default=200)
    cfg.add_to_config("xhat_method",
                      description="Method to generate xhat: EF or cylinders (default EF)",
                      domain=str,
                      default="EF")
    confidence_config.confidence_config(cfg)
    confidence_config.sequential_config(cfg)
    confidence_config.BM_config(cfg)
    confidence_config.BPL_config(cfg)


def _ef_xhat_generator(scenario_names, solver_name=None, solver_options=None,
                        cfg=None, module_name=None, start_seed=None,
                        branching_factors=None, **kwargs):
    """Model-agnostic xhat generator using Extensive Form.

    Args:
        scenario_names (list of str): scenario names for the sample
        solver_name (str): solver to use
        solver_options (dict): solver options
        cfg (Config): the main Config (a safe copy will be made)
        module_name (str): importable module name
        start_seed (int, optional): starting seed for multi-stage
        branching_factors (list of int, optional): for multi-stage problems

    Returns:
        dict: nonant cache, e.g. {'ROOT': [v1, v2, ...]}
    """
    num_scens = len(scenario_names)
    local_cfg = cfg()  # safe copy

    if branching_factors is not None:
        local_cfg.quick_assign("EF_mstage", bool, True)
    else:
        local_cfg.quick_assign("EF_2stage", bool, True)

    local_cfg.quick_assign("EF_solver_name", str, solver_name)
    if solver_options is not None:
        solver_options_str = sputils.option_dict_to_string(solver_options)
        local_cfg.quick_assign("EF_solver_options", str, solver_options_str)
    else:
        local_cfg.quick_assign("EF_solver_options", dict, {})
    local_cfg.quick_assign("num_scens", int, num_scens)
    local_cfg.quick_assign("_mpisppy_probability", float, 1 / num_scens)
    if start_seed is not None:
        local_cfg.quick_assign("start_seed", int, start_seed)

    ama = amalgamator.from_module(module_name, local_cfg, use_command_line=False)
    ama.scenario_names = scenario_names
    ama.verbose = False
    ama.run()

    xhat = sputils.nonant_cache_from_ef(ama.ef)
    return xhat


def _cylinder_xhat_generator(scenario_names, solver_name=None,
                              solver_options=None, cfg=None,
                              module_name=None, module=None,
                              start_seed=None, branching_factors=None,
                              **kwargs):
    """Model-agnostic xhat generator using hub-and-spoke decomposition.

    This spins up a WheelSpinner for each sample, which is heavier than
    EF but enables solving problems too large for the extensive form.

    Args:
        scenario_names (list of str): scenario names for the sample
        solver_name (str): solver to use
        solver_options (dict): solver options
        cfg (Config): the main Config (must have cylinder args configured)
        module_name (str): importable module name
        module: the loaded module object
        start_seed (int, optional): starting seed for multi-stage
        branching_factors (list of int, optional): for multi-stage problems

    Returns:
        dict: nonant cache, e.g. {'ROOT': [v1, v2, ...]}
    """
    from mpisppy.generic.decomp import do_decomp

    global_comm = MPI.COMM_WORLD
    global_rank = global_comm.Get_rank()

    # We need a cfg that has the right num_scens for this sample
    local_cfg = cfg()  # safe copy
    local_cfg['num_scens'] = len(scenario_names)

    scenario_creator = module.scenario_creator
    scenario_creator_kwargs = module.kw_creator(local_cfg)
    scenario_denouement = module.scenario_denouement

    wheel = do_decomp(module, local_cfg, scenario_creator,
                      scenario_creator_kwargs, scenario_denouement)

    # Extract xhat from the wheel, using the same approach as generic/mmw.py
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

    from mpisppy.confidence_intervals import ciutils
    if not os.path.exists(tmp_path):
        raise RuntimeError("Cylinder xhat generator: no feasible solution found")
    xhat = ciutils.read_xhat(tmp_path)
    # Sync before rank 0 removes the file so peer ranks finish reading it.
    global_comm.Barrier()
    if global_rank == 0:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return xhat


def do_mrp(module_fname, module, cfg):
    """Run sequential sampling (MRP) and return results.

    Args:
        module_fname (str): module file name or path as given on CLI
        module: the loaded module object
        cfg (Config): parsed config with sequential sampling args

    Returns:
        dict: {"T": T, "Candidate_solution": xhat, "CI": [0, upper_bound]}
    """
    from mpisppy.confidence_intervals import seqsampling
    from mpisppy.confidence_intervals import multi_seqsampling

    global_comm = MPI.COMM_WORLD
    global_rank = global_comm.Get_rank()

    # The CI stack needs an importable module name string.
    # load_module already added the directory to sys.path.
    importable_name = os.path.basename(module_fname)

    # Determine two-stage vs multi-stage
    is_multistage = cfg.get("branching_factors") is not None

    # Set the solving_type for SeqSampling
    if is_multistage:
        solving_type = "EF_mstage"
        cfg.quick_assign("EF_mstage", bool, True)
    else:
        solving_type = "EF_2stage"
        cfg.quick_assign("EF_2stage", bool, True)

    # Default the EF solver to the main solver when not explicitly set
    if cfg.get("EF_solver_name") is None:
        cfg.quick_assign("EF_solver_name", str, cfg.solver_name)

    # Build the xhat_generator with the right method
    xhat_method = cfg.get("xhat_method", "EF")
    if xhat_method == "EF":
        xhat_gen = functools.partial(
            _ef_xhat_generator,
            cfg=cfg,
            module_name=importable_name,
        )
    elif xhat_method == "cylinders":
        xhat_gen = functools.partial(
            _cylinder_xhat_generator,
            cfg=cfg,
            module_name=importable_name,
            module=module,
        )
    else:
        raise ValueError(f"Unknown xhat_method: {xhat_method}. Use 'EF' or 'cylinders'.")

    # Build xhat_gen_kwargs from the module
    scenario_creator_kwargs = module.kw_creator(cfg)
    xhat_gen_kwargs = dict(scenario_creator_kwargs)
    cfg.quick_assign("xhat_gen_kwargs", dict, xhat_gen_kwargs)

    # Configure stopping criterion
    stopping = cfg.stopping_criterion
    if stopping not in ("BM", "BPL"):
        raise ValueError(f"--stopping-criterion must be BM or BPL, got: {stopping}")

    stochastic_sampling = False
    if stopping == "BPL":
        stochastic_sampling = (cfg.get("BPL_n0min", 0) != 0)

    # Instantiate the appropriate sampler
    if is_multistage:
        sampler = multi_seqsampling.IndepScens_SeqSampling(
            importable_name,
            xhat_gen,
            cfg,
            stopping_criterion=stopping,
            stochastic_sampling=stochastic_sampling,
            solving_type=solving_type,
        )
    else:
        sampler = seqsampling.SeqSampling(
            importable_name,
            xhat_gen,
            cfg,
            stochastic_sampling=stochastic_sampling,
            stopping_criterion=stopping,
            solving_type=solving_type,
        )

    maxit = cfg.get("mrp_max_iterations", 200)
    result = sampler.run(maxit=maxit)

    if global_rank == 0:
        global_toc(f"MRP complete: T={result['T']}, CI={result['CI']}")

    return result
