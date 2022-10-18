"""Provides top-level functions for USAR with `ExtensiveForm`.

This script is intended to be run from the command line. Command-line
arguments serve to configure the USAR problem being solved. You can run
with ``--help`` to show the command-line arguments. Additionally, the
functions here can be imported and used as documented.
"""
import itertools
import os

from config import add_extensive_form_args, add_common_args
from generate_data import generate_coords, generate_data
import mpisppy.opt.ef
import mpisppy.utils.config
from scenario_creator import scenario_creator
from write_solutions import walks_writer, gantt_writer


def extensive_form(
    solver_name: str, num_scens: int, **generate_data_kwargs
) -> mpisppy.opt.ef.ExtensiveForm:
    """Creates an `ExtensiveForm` to solve a USAR stochastic program.

    Args:
        solver_name: Solver to be used for optimization.
        num_scens: Positive number of scenarios generated.
        **generate_data_kwargs: Arguments passed on to `generate_data`.

    Returns:
        An unsolved `ExtensiveForm` for the stochastic program.
    """
    if num_scens <= 0:
        raise ValueError("Provide a positive integer for num_scens")
    scenario_names = list(map(str, range(num_scens)))

    data_dicts = list(
        itertools.islice(generate_data(**generate_data_kwargs), num_scens))
    depot_coords, site_coords = generate_coords(**generate_data_kwargs)
    scenario_creator_kwargs = {
        "data_dicts": data_dicts,
        "depot_coords": depot_coords,
        "site_coords": site_coords,
    }

    return mpisppy.opt.ef.ExtensiveForm(
        {"solver": solver_name},
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs=scenario_creator_kwargs,
        model_name="USAR extensive form",
    )


def main() -> None:
    """Runs command line-configured `ExtensiveForm`; plots solution."""
    cnfg = mpisppy.utils.config.Config()
    add_extensive_form_args(cnfg)
    add_common_args(cnfg)
    cnfg.parse_command_line()

    ef = extensive_form(**cnfg)

    ef.solve_extensive_form(tee=True)

    output_dir = cnfg["output_dir"]
    ef.write_tree_solution(os.path.join(output_dir, "walks"), walks_writer)
    ef.write_tree_solution(os.path.join(output_dir, "gantts"), gantt_writer)


if __name__ == "__main__":
    main()
