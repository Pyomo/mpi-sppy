import argparse
import itertools
from typing import Iterable

from generate_data import generate_coords, generate_data
import mpisppy.spin_the_wheel
from mpisppy.utils import baseparsers, vanilla
from parser import add_wheel_spinner_args, add_common_args
from scenario_creator import scenario_creator
from scenario_denouement import scenario_denouement

SUPPORTED_SPOKES = (
    "fwph",
    "lagrangian",
    "lagranger",
    "xhatlooper",
    "xhatshuffle",
    "xhatlshaped",
    "slamup",
    "slamdown",
)


def wheel_spinner(
    num_scens: int, argparse_args: argparse.Namespace, spokes: Iterable[str]
) -> mpisppy.spin_the_wheel.WheelSpinner:
    if num_scens <= 0:
        raise ValueError("Provide a positive integer for num_scens")
    scenario_names = list(map(str, range(num_scens)))

    data_dicts = list(
        itertools.islice(generate_data(**vars(argparse_args)), num_scens))
    depot_coords, site_coords = generate_coords(**vars(argparse_args))
    scenario_creator_kwargs = {
        "data_dicts": data_dicts,
        "depot_coords": depot_coords,
        "site_coords": site_coords,
    }

    vanilla_args = \
        (argparse_args, scenario_creator, scenario_denouement, scenario_names)
    vanilla_kwargs = {"scenario_creator_kwargs": scenario_creator_kwargs}

    vanilla_fn = vanilla.aph_hub if argparse_args.run_async else vanilla.ph_hub
    hub_dict = vanilla_fn(*vanilla_args, **vanilla_kwargs)

    spoke_dicts = []
    for spoke_name in spokes:
        vanilla_fn = getattr(vanilla, spoke_name + "_spoke")
        spoke_dicts.append(vanilla_fn(*vanilla_args, **vanilla_kwargs))

    return mpisppy.spin_the_wheel.WheelSpinner(hub_dict, spoke_dicts)


def main() -> None:
    parser = baseparsers.make_parser(num_scens_reqd=True)
    baseparsers.two_sided_args(parser)
    baseparsers.aph_args(parser)
    for spoke_name in SUPPORTED_SPOKES:
        getattr(baseparsers, spoke_name + "_args")(parser)
    add_wheel_spinner_args(parser)
    add_common_args(parser)
    args = parser.parse_args()

    spokes = [s for s in SUPPORTED_SPOKES if getattr(args, "with_" + s)]
    ws = wheel_spinner(args.num_scens, args, spokes)

    ws.spin()


if __name__ == "__main__":
    main()
