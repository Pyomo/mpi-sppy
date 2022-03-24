import argparse
import itertools
import os

from generate_data import generate_coords, generate_data
import mpisppy.opt.ef
from parser import add_extensive_form_args, add_common_args
from scenario_creator import scenario_creator
from write_solutions import walks_writer, gantt_writer


def extensive_form(
    solver_name: str, num_scens: int, **generate_data_kwargs
) -> mpisppy.opt.ef.ExtensiveForm:
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
    parser = argparse.ArgumentParser()
    add_extensive_form_args(parser)
    add_common_args(parser)
    args = parser.parse_args()

    ef = extensive_form(**vars(args))

    ef.solve_extensive_form(tee=True)

    output_dir = args.output_dir
    ef.write_tree_solution(os.path.join(output_dir, "walks"), walks_writer)
    ef.write_tree_solution(os.path.join(output_dir, "gantts"), gantt_writer)


if __name__ == "__main__":
    main()
