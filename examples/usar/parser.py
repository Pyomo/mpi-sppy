import argparse


def add_extensive_form_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("solver_name", help="Optimization solver to use")

    parser.add_argument(
        "num_scens",
        type=int,
        help="Positive number of scenarios (int) generated",
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory for output files (current directory by default)",
    )

    generate_data = parser.add_argument_group(
        "generate_data arguments", "Arguments passed on to the data generator"
    )
    generate_data.add_argument(
        "time_horizon",
        type=int,
        help="Number of time steps (int) considered in the optimization model",
    )
    generate_data.add_argument(
        "time_unit_minutes",
        type=float,
        help="Number of minutes (float) per time step",
    )
    generate_data.add_argument(
        "num_depots",
        type=int,
        help="Number of depots (int) generated",
    )
    generate_data.add_argument(
        "num_active_depots",
        type=int,
        help="Number of depots (int) allowed to be active",
    )
    generate_data.add_argument(
        "num_households",
        type=int,
        help="Number of households (int) generated",
    )
    generate_data.add_argument(
        "constant_rescue_time",
        type=int,
        help="Flat time (int) for each household rescue",
    )
    generate_data.add_argument(
        "travel_speed",
        type=float,
        help="Distance on the unit square (float) traveled per time step",
    )
    generate_data.add_argument(
        "constant_depot_inflow",
        type=int,
        help="Number of rescue teams (int) arriving at depots each time step",
    )
    generate_data.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Seed (int) for the random module",
    )
