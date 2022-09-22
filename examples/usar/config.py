from pyomo.common.config import ConfigDict, ConfigValue


def add_extensive_form_args(cnfg: ConfigDict) -> None:
    cnfg.declare("num_scens", ConfigValue(
        domain=int,
        description="Positive number of scenarios (int) generated",
    )).declare_as_argument(required=True)

    cnfg.declare("solver_name", ConfigValue(
        description="Optimization solver to be used",
    )).declare_as_argument(required=True)

    cnfg.declare("seed", ConfigValue(
        domain=int,
        description="Seed (int) for the random module",
    )).declare_as_argument()


def add_wheel_spinner_args(cnfg: ConfigDict) -> None:
    cnfg.declare("run_async", ConfigValue(
        default=False,
        domain=bool,
        description="Run async projective hedging, not progressive hedging",
    )).declare_as_argument()


def add_common_args(cnfg: ConfigDict) -> None:
    cnfg.declare("output_dir", ConfigValue(
        default=".",
        description="Directory for output files (current dir by default)",
    )).declare_as_argument()

    group = "generate_data arguments"
    for name, domain, desc in (
        ("time_horizon", int,
         "Number of time steps (int) considered in the optimization model"),
        ("time_unit_minutes", float,
         "Number of minutes (float) per time step"),
        ("num_depots", int, "Number of depots (int) generated"),
        ("num_active_depots", int,
         "Number of depots (int) allowed to be active"),
        ("num_households", int, "Number of households (int) generated"),
        ("constant_rescue_time", int,
         "Flat time (int) for each household rescue"),
        ("travel_speed", float,
         "Distance on the unit square (float) traveled per time step"),
        ("constant_depot_inflow", int,
         "Number of rescue teams (int) arriving at depots per time step"),
    ):
        val = ConfigValue(domain=domain, description=desc)
        cnfg.declare(name, val) \
            .declare_as_argument(group=group, required=True)
