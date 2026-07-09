###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Run ONE experiment arm on the sizes model and record its metrics.

Invoked as a subprocess by ``run_experiments.py`` (one process per arm, for
clean module state). Reads a single JSON argument describing the arm, builds a
PH-only hub on sizes via the vanilla cylinders helpers, attaches the experiment
monitor (and, for the slam arm, the real Slammer extension), and spins. The
monitor writes the per-iteration metrics CSV.

Arm JSON keys: arm, intervention (none|w_damping|rho_reduce|w_reset|fix|
prox_boost), factor, floorfrac, start_iter, iters_between_slams, boost_factor,
boost_iters, refire_cooldown, escalate_mult, escalate_every, escalate_cap,
escalate_on, escalate_gap_thresh, measure_quality, smoothing,
smoothing_rho_ratio, smoothing_beta, num_scens, iters, solver, default_rho,
out_csv, rho_setter (native|off|perturb), eps, seed, pmode.
"""

import json
import os
import sys

# The sizes model is imported by short name (like sizes_cylinders.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    arm = json.load(open(sys.argv[1]))

    import sizes
    from mpisppy.utils import config
    import mpisppy.utils.cfg_vanilla as vanilla
    from mpisppy.spin_the_wheel import WheelSpinner
    from w_osc_experiment_ext import WOscExperimentMonitor

    # Build a Config as sizes_cylinders.py does, but feed argv programmatically.
    sys.argv = [
        "_run_one_arm",
        "--num-scens", str(arm["num_scens"]),
        "--solver-name", arm["solver"],
        "--max-iterations", str(arm["iters"]),
        "--default-rho", str(arm["default_rho"]),
        "--rel-gap", "1e-6", "--abs-gap", "1e-6",
    ]
    cfg = config.Config()
    cfg.popular_args()
    cfg.num_scens_required()
    cfg.ph_args()
    cfg.two_sided_args()
    cfg.mip_options()
    cfg.parse_command_line("_run_one_arm")

    # smoothed PH (mpi-sppy's built-in smoothing term): anchor each scenario's x
    # to its own EMA trajectory z, penalty p = smoothing_rho_ratio * rho, EMA
    # memory beta. vanilla.ph_hub reads these off cfg, so set them before it.
    if arm.get("smoothing", False):
        cfg.smoothing = True
        cfg.smoothing_rho_ratio = float(arm.get("smoothing_rho_ratio", 0.1))
        cfg.smoothing_beta = float(arm.get("smoothing_beta", 0.2))

    num_scen = cfg.num_scens
    scenario_creator = sizes.scenario_creator
    scenario_denouement = sizes.scenario_denouement
    all_scenario_names = [f"Scenario{i + 1}" for i in range(num_scen)]

    # rho profile: native setter, disabled (uniform --default-rho), or perturbed.
    mode = arm.get("rho_setter", "native")
    if mode == "off":
        rho_setter = None
    elif mode == "perturb":
        import numpy as np
        rng = np.random.default_rng(int(arm.get("seed", 0)))
        eps = float(arm.get("eps", 0.0))
        pmode = arm.get("pmode", "per_var")
        pert = {}

        def rho_setter(scen, **kw):
            lst = sizes._rho_setter(scen, **kw)
            out = []
            for k, (idv, rho) in enumerate(lst):
                if pmode == "per_var":
                    if k not in pert:
                        pert[k] = 1.0 + rng.uniform(-eps, eps)
                    f = pert[k]
                else:
                    f = 1.0 + rng.uniform(-eps, eps)
                out.append((idv, rho * f))
            return out
    else:
        rho_setter = sizes._rho_setter

    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)
    hub_dict = vanilla.ph_hub(
        *beans,
        scenario_creator_kwargs={"scenario_count": num_scen},
        rho_setter=rho_setter,
    )
    hub_dict["opt_kwargs"]["options"]["defaultPHrho"] = arm["default_rho"]

    vanilla.extension_adder(hub_dict, WOscExperimentMonitor)
    hub_dict["opt_kwargs"]["options"]["wosc_experiment_options"] = {
        "intervention": arm["intervention"],
        "factor": arm.get("factor", 0.5),
        "floorfrac": arm.get("floorfrac", 1e-3),
        "start_iter": arm.get("start_iter", 10),
        "iters_between_slams": arm.get("iters_between_slams", 3),
        "boost_factor": arm.get("boost_factor", 10.0),
        "boost_iters": arm.get("boost_iters", 5),
        "refire_cooldown": arm.get("refire_cooldown", 0),
        "escalate_mult": arm.get("escalate_mult", 1.0),
        "escalate_every": arm.get("escalate_every", 5),
        "escalate_cap": arm.get("escalate_cap", 1e6),
        "escalate_on": arm.get("escalate_on", "gap"),
        "escalate_gap_thresh": arm.get("escalate_gap_thresh", 5.0),
        "measure_quality": arm.get("measure_quality", False),
        "out_csv": arm["out_csv"],
    }

    WheelSpinner(hub_dict, []).spin()


if __name__ == "__main__":
    main()
