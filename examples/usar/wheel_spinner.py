"""Provides top-level functions for USAR with `WheelSpinner`.

This script is intended to be run from the command line. Command-line
arguments serve to configure the USAR problem being solved. You can run
with ``--help`` to show the command-line arguments. Additionally, the
functions here can be imported and used as documented.
"""
import itertools
import os

import pyomo.common.config

from config import add_wheel_spinner_args, add_common_args
from generate_data import generate_coords, generate_data
import mpisppy.spin_the_wheel
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.config
from scenario_creator import scenario_creator
from scenario_denouement import scenario_denouement
from write_solutions import walks_writer, gantt_writer

SUPPORTED_SPOKES = (
    "fwph",
    "lagrangian",
    "lagranger",
    "xhatlooper",
    "xhatshuffle",
    "xhatlshaped",
    "slammax",
    "slammin",
)


def wheel_spinner(
    cnfg: pyomo.common.config.ConfigDict,
) -> mpisppy.spin_the_wheel.WheelSpinner:
    """Creates a `WheelSpinner` to solve a USAR stochastic program.

    Args:
        cnfg: `WheelSpinner` and general configuration as in config.py.

    Returns:
        An unsolved `WheelSpinner` for the stochastic program.
    """
    if cnfg["num_scens"] <= 0:
        raise ValueError("Provide a positive integer for num_scens")
    scenario_names = list(map(str, range(cnfg["num_scens"])))

    data_dicts = list(
        itertools.islice(generate_data(**cnfg), cnfg["num_scens"]))
    depot_coords, site_coords = generate_coords(**cnfg)
    scenario_creator_kwargs = {
        "data_dicts": data_dicts,
        "depot_coords": depot_coords,
        "site_coords": site_coords,
    }

    vanilla_args = \
        (cnfg, scenario_creator, scenario_denouement, scenario_names)
    vanilla_kwargs = {"scenario_creator_kwargs": scenario_creator_kwargs}

    vanilla_fn = vanilla.aph_hub if cnfg["run_async"] else vanilla.ph_hub
    hub_dict = vanilla_fn(*vanilla_args, **vanilla_kwargs)

    spoke_dicts = []
    for spoke_name in SUPPORTED_SPOKES:
        if getattr(cnfg, spoke_name):
            vanilla_fn = getattr(vanilla, spoke_name + "_spoke")
            spoke_dicts.append(vanilla_fn(*vanilla_args, **vanilla_kwargs))

    return mpisppy.spin_the_wheel.WheelSpinner(hub_dict, spoke_dicts)


def main() -> None:
    """Runs command line-configured `WheelSpinner`; plots solution."""
    cnfg = mpisppy.utils.config.Config()
    cnfg.num_scens_required()
    cnfg.popular_args()
    cnfg.two_sided_args()
    add_wheel_spinner_args(cnfg)
    cnfg.ph_args()
    cnfg.aph_args()
    for spoke_name in SUPPORTED_SPOKES:
        getattr(cnfg, spoke_name + "_args")()
    add_common_args(cnfg)
    cnfg.parse_command_line()

    ws = wheel_spinner(cnfg)

    ws.spin()

    output_dir = cnfg["output_dir"]
    ws.write_tree_solution(os.path.join(output_dir, "walks"), walks_writer)
    ws.write_tree_solution(os.path.join(output_dir, "gantts"), gantt_writer)


if __name__ == "__main__":
    main()
