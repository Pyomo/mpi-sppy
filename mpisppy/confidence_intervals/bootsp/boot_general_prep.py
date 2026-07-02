###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Create npy files for xhat and optimal that are used in simulations.
#
#   python -m mpisppy.confidence_intervals.bootsp.boot_general_prep <json>
#
# Compute the optimal function value with max_count scenarios (or read it from
# a file), then find a candidate solution using the last candidate_sample_size
# scenarios and compute the corresponding optimality gap.

import sys
import json
import numpy as np
import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
import mpisppy.confidence_intervals.ciutils as ciutils
import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils
import mpisppy.confidence_intervals.bootsp.boot_sp as boot_sp


def find_optimal(cfg, module):
    opt_ef = boot_sp.solve_routine(cfg, module, range(cfg.max_count), num_threads=16)
    opt_obj = pyo.value(opt_ef.EF_Obj)
    return opt_obj


def find_candidate(cfg, module):
    scenarios = range(cfg.max_count - cfg.candidate_sample_size, cfg.max_count)
    if len(scenarios) == 1:
        print(f"only one scenario, {scenarios},  for candidate solution")
    candidate_ef = boot_sp.solve_routine(cfg, module, scenarios, num_threads=2, duplication=False)

    xhat = sputils.nonant_cache_from_ef(candidate_ef)
    return xhat


def find_gap(cfg, module, xhat, opt_obj):
    obj_hat = boot_sp.evaluate_scenarios(cfg, module, range(cfg.max_count), xhat, duplication=False)
    opt_gap = obj_hat - opt_obj
    return opt_gap


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("need json file")
        print("usage (e.g.): python -m mpisppy.confidence_intervals.bootsp.boot_general_prep little_schultz.json")
        quit()

    json_fname = sys.argv[1]
    cfg = boot_utils.cfg_from_json(json_fname)

    boot_utils.check_BFs(cfg)

    cfg.add_to_config(name="use_fitted",
                    description="a boolean to control use of fitted distribution",
                    domain=bool,
                    default=None,
                    argparse=False)
    cfg.use_fitted = False

    if "deterministic_data_json" in cfg:
        json_fname = cfg.deterministic_data_json
        try:
            with open(json_fname, "r") as read_file:
                detdata = json.load(read_file)
        except Exception:
            print(f"Could not read the json file: {json_fname}")
            raise
        cfg.add_to_config("detdata",
                        description="determinstic data from json file",
                        domain=dict,
                        default=detdata)

    module = boot_utils.module_name_to_module(cfg.module_name)

    xhat_fname = cfg["xhat_fname"]

    opt_obj = find_optimal(cfg, module)
    xhat = find_candidate(cfg, module)
    opt_gap = find_gap(cfg, module, xhat, opt_obj)

    np.save(cfg.optimal_fname, [opt_obj, opt_gap])
    ciutils.write_xhat(xhat, path=xhat_fname)

    print(f"opt_obj: {opt_obj}")
    print(f"opt_gap: {opt_gap}")
