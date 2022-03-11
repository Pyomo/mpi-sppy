""" Main program to build and run the EMPRISE modelling and optimization framework """
#
# Imports
#


import os

from functools import reduce
from datetime import datetime

from mpisppy.utils import sputils
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils import baseparsers
from mpisppy.utils import vanilla

# import mpisppy.cylinders as cylinders
from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
from mpisppy.convergers.norm_rho_converger import NormRhoConverger

import emprise

# General configuration choices ====================================================================
# use_cluster = False # not used at the moment
SCENARIO_NAME = "test"  # user's choice (needs to be in line with input data names)
SCENARIO_VARIANT = "storage"  # user's choice
EXCLUDE_COMPONENT_LIST = []  # ["electricity_storage"], []
WRITE_SOLUTION = True
SOLVE_EXTENSIVE_FORM = False

# Scenario (tree) configuration ====================================================================
NUMBER_OF_STAGES = 3  # Planning periods, e.g. 3 (representing planning periods 2025, 2035, 2045)

# Operational uncertainty -> System operation (sysop) configuration
SAMPLE_SIZE = 1 * 7 * 24  # hourly time steps of system operation sample # NOTE: will change due to representative and extreme periods
SAMPLE_OFFSET = 8 * 7 * 24  # offset of system operation sample # NOTE: will change due to representative and extreme periods

REFERENCE_YEARS = [2010, 2012]  # [2010, 2012]  # [2010, 2012]  # [2006, 2008]  # [2007, 2008, 2011] # Currently only one set for all stages
OPERATIONAL_UNCERTAINTY_PROBABILITY = [0.4, 0.6]  # needs to sum up to 1.0

# Strategic uncertainty -> Investment (inv) configuration
UNCERTAINTY_EMISSION_PRICE_SCENARIO = {1: [70.0, 100.0, 200.0], 2: [70.0, 70.0, 70.0]}  # EUR/tCO2eq
STRATEGIC_UNCERTAINTY_PROBABILITY = [0.7, 0.3]

def check_and_create_directory(directory):
    """Checks and creates the directory if it does not already exist"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory


def _get_emprise_configuration():
    """Build and return EMPRISE configuration (dict)"""
    number_of_investment_scenarios = len(STRATEGIC_UNCERTAINTY_PROBABILITY)
    number_of_system_operation_scenarios = len(REFERENCE_YEARS)

    # Create scenario probability dictionary ===========================================================
    scenario_probability_sysop = {i: OPERATIONAL_UNCERTAINTY_PROBABILITY for i in range(1, NUMBER_OF_STAGES + 1)}  # reference year probabilities are the same in all stages
    scenario_probability_inv = {i: STRATEGIC_UNCERTAINTY_PROBABILITY for i in range(1, NUMBER_OF_STAGES + 1)}  #
    # scenario_probability_inv = {"inv_" + str(i): [1.0] for i in range(1, NUMBER_OF_STAGES + 1)}
    scenario_probability = {"inv": scenario_probability_inv, "sysop": scenario_probability_sysop}
    # print(scenario_probability)

    # --- Path and directory information
    scenario_name_extended = SCENARIO_NAME + "_" + SCENARIO_VARIANT + "_" + "_".join([str(ry) for ry in REFERENCE_YEARS]) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    data_folder_name = "data"
    plot_folder_name = "plot"
    result_folder_name = "result/" + scenario_name_extended
    # json_file_name = "ef_solution.json"
    cluster_working_directory = "."  # "/export/home/dlwoodruff/Documents/DLWFORKS/mpi-sppy-1/examples/EMPRISE"  # "/home/phaertel/emprise"

    print("### Initializing EMPRISE framework for scenario setup '" + scenario_name_extended + "' ###")
    # --- Check whether running on local windows or other machine (HPCC)
    if os.name == "nt":
        working_dir = ""
        os.environ["EMPRISE_JOB_NAME"] = scenario_name_extended
    else:
        working_dir = cluster_working_directory
        # if "TMP_PYOMO_DIR" in os.environ:
        #     cluster_tmp_directory = os.environ["TMP_PYOMO_DIR"]
        #     TempfileManager.tempdir = cluster_tmp_directory
        #     # print('### Using temporary pyomo directory ', cluster_tmp_directory, ' ###')

    path_data_dir = os.path.join(working_dir, data_folder_name)
    path_plot_dir = check_and_create_directory(os.path.join(working_dir, plot_folder_name))
    path_result_dir = check_and_create_directory(os.path.join(working_dir, result_folder_name))

    # --- Strucural input data directory information
    file_names_structural = {
        "nodes": os.path.join(path_data_dir, SCENARIO_NAME + "_nodes.csv"),
        "branches": os.path.join(path_data_dir, SCENARIO_NAME + "_branches.csv"),
        "generators_thermal": os.path.join(path_data_dir, SCENARIO_NAME + "_generators_thermal.csv"),
        "generators_renewable": os.path.join(path_data_dir, SCENARIO_NAME + "_generators_renewable.csv"),
        "consumers_conventional": os.path.join(path_data_dir, SCENARIO_NAME + "_consumers_conventional.csv"),
        "storage": os.path.join(path_data_dir, SCENARIO_NAME + "_storage.csv"),
        "cost_generation": os.path.join(path_data_dir, SCENARIO_NAME + "_cost_generation.csv"),
        "cost_storage": os.path.join(path_data_dir, SCENARIO_NAME + "_cost_storage.csv"),
        "fuel": os.path.join(path_data_dir, SCENARIO_NAME + "_fuel.csv"),
    }

    # --- Timeseries input data directory information
    file_names_timeseries = {
        "consumers_conventional": os.path.join(
            path_data_dir,
            "ts",
            SCENARIO_NAME + "_consumers_conventional_%NODE%_%REFYEAR%_timeseries.csv",
        ),
        "wind": os.path.join(
            path_data_dir,
            "ts",
            SCENARIO_NAME + "_wind_%NODE%_%TYPE%_%LCOE%_%REFYEAR%_timeseries.csv",
        ),
        "solar": os.path.join(
            path_data_dir,
            "ts",
            SCENARIO_NAME + "_solar_%NODE%_%TYPE%_%LCOE%_%REFYEAR%_timeseries.csv",
        ),
    }

    n_header_rows_timeseries = {
        "consumers_conventional": [0, 1],
        "wind": [0, 1, 2, 3],
        "solar": [0, 1, 2, 3],
    }

    # Create a dictionary with configuration data for multi-stage problems with EMPRISE ================
    emprise_config = {
        "number_of_stages": NUMBER_OF_STAGES,
        "number_of_investment_scenarios": number_of_investment_scenarios,
        "number_of_system_operation_scenarios": number_of_system_operation_scenarios,
        "reference_years": REFERENCE_YEARS,
        "uncertainty_emission_price_scenario": UNCERTAINTY_EMISSION_PRICE_SCENARIO,
        "sample_size": SAMPLE_SIZE,
        "sample_offset": SAMPLE_OFFSET,
        "scenario_probability": scenario_probability,
        "file_names_structural": file_names_structural,
        "file_names_timeseries": file_names_timeseries,
        "n_header_rows_timeseries": n_header_rows_timeseries,
        "path_result_dir": path_result_dir,
        "path_plot_dir": path_plot_dir,
        "path_data_dir": path_data_dir,
        "exclude_component_list": EXCLUDE_COMPONENT_LIST,
        "spoke_sleep_time": None, # do not set spoke sleep time here
        "write_solution": WRITE_SOLUTION,
        "solve_extensive_form": SOLVE_EXTENSIVE_FORM,
    }
    return emprise_config


def _parse_args():
    parser = baseparsers.make_multistage_parser()
    parser = baseparsers.two_sided_args(parser)
    parser = baseparsers.xhatlooper_args(parser)
    parser = baseparsers.xhatshuffle_args(parser)
    parser = baseparsers.lagrangian_args(parser)
    parser = baseparsers.xhatspecific_args(parser)

    parser.add_argument("--use-norm-rho-updater",
                        help="Use the norm rho updater extension",
                        dest="use_norm_rho_updater",
                        action="store_true")
    parser.add_argument("--use-norm-rho-converger",
                        help="Use the norm rho converger",
                        dest="use_norm_rho_converger",
                        action="store_true")

    
    args = parser.parse_args()
    return args


def main():
    """Contains main scripts to configure and run EMPRISE on a distributed computational platform"""
    args = _parse_args()
    emprise_config = _get_emprise_configuration()
    # print(args)

    if args.use_norm_rho_converger:
        if not args.use_norm_rho_updater:
            raise RuntimeError("--use-norm-rho-converger requires --use-norm-rho-updater")
        else:
            ph_converger = NormRhoConverger
    else:
        ph_converger = None


    # Multi-stage scenario tree
    # Example for 3 planning periods (e.g. 2025, 2035, 2045)
    # and 2 strategic uncertainty scenario variants (e.g. high and low investment cost)
    # and 2 system operation uncertainty scenarios (e.g. two meteorological reference years)
    # stage:                                1       2       3       4
    # inv decisions:                        yes     yes     yes     no
    # sysop decisions:                      no      yes     yes     yes
    # #nodes:                               1       4       16      32
    # branching factor (to next stage):     4       4       2       - (leaf nodes)
    emprise_config["branching_factors"] = args.branching_factors  # add branching factor information to emprise_config dict
    if emprise_config["branching_factors"] is None:
        emprise_config["branching_factors"] = []
        for stg in range(1, emprise_config["number_of_stages"] + 1):
            if stg == emprise_config["number_of_stages"]:
                emprise_config["branching_factors"].append(emprise_config["number_of_system_operation_scenarios"])
            else:
                emprise_config["branching_factors"].append(emprise_config["number_of_investment_scenarios"] * emprise_config["number_of_system_operation_scenarios"])
    # emprise_config["branching_factors"] = BFs  # add branching factor information to emprise_config dict

    with_xhatshuffle = args.with_xhatshuffle
    with_lagrangian = args.with_lagrangian

    # EMPRISE is multi-stage, so we need to supply node names
    all_nodenames = sputils.create_nodenames_from_branching_factors(emprise_config["branching_factors"])  # e.g. ['ROOT', 'ROOT_0', 'ROOT_1', 'ROOT_2', 'ROOT_3', 'ROOT_0_0', 'ROOT_0_1', ...]
    scen_count = reduce(lambda x, y: x * y, emprise_config["branching_factors"])  # e.g. 4*4*2 = 32 with emprise_config["branching_factors"] = [4, 4, 2]
    scenario_creator_kwargs = {
        "branching_factors": emprise_config["branching_factors"],
        "emprise_config": emprise_config,
    }  # e.g. {'branching_factors': [4, 4, 2]}
    all_scenario_names = [f"Scen{i+1}" for i in range(scen_count)]  # e.g. ['Scen1', 'Scen2', 'Scen3', 'Scen4', 'Scen5', 'Scen6', 'Scen7', ...]

    # Import callback functions from EMPRISE model
    scenario_creator = emprise.scenario_creator
    scenario_denouement = emprise.scenario_denouement

    # Build single scenario instance for one exemplary scenario ("Scen17") (only used for local debugging)
    # scenario_creator("Scen17", emprise_config["branching_factors"], emprise_config, None)  # For local debugging only - comment line for cluster use

    # Build and solve extensive form of multi-stage EMPRISE instance (if configured)
    # Mainly used for testing
    if emprise_config["solve_extensive_form"] is True:
        import pyomo.environ as pyo

        solver = pyo.SolverFactory("cplex")
        extensive_form_instance = sputils.create_EF(
            all_scenario_names,
            scenario_creator,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )
        results = solver.solve(extensive_form_instance, tee=True)
        print("EF objective value:", pyo.value(extensive_form_instance.EF_Obj))
        sputils.ef_nonants_csv(extensive_form_instance, "vardump.csv")

    rho_setter = None

    # Things needed for vanilla cylinders
    beans = (args, scenario_creator, scenario_denouement, all_scenario_names)

    # Vanilla PH hub # do not set spoke sleep time here (spoke_sleep_time=emprise_config["spoke_sleep_time"])
    hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs, ph_extensions=None, rho_setter=rho_setter, all_nodenames=all_nodenames)

    ## hack in adaptive rho
    if args.use_norm_rho_updater:
        hub_dict['opt_kwargs']['extensions'] = NormRhoUpdater
        hub_dict['opt_kwargs']['options']['norm_rho_options'] = {'verbose': True}

    # Standard Lagrangian bound spoke
    if with_lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs, rho_setter=rho_setter, all_nodenames=all_nodenames, )

    # xhat looper bound spoke

    if with_xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, all_nodenames=all_nodenames, scenario_creator_kwargs=scenario_creator_kwargs, )

    list_of_spoke_dict = list()
    if with_lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if with_xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    # Set solver-specific options for oracle calls =================================================
    # --- PH Hub
    hub_dict["opt_kwargs"]["options"]["iter0_solver_options"]["lpmethod"] = 4
    hub_dict["opt_kwargs"]["options"]["iter0_solver_options"]["solutiontype"] = 2
    hub_dict["opt_kwargs"]["options"]["iterk_solver_options"]["lpmethod"] = 4
    hub_dict["opt_kwargs"]["options"]["iterk_solver_options"]["solutiontype"] = 2

    # --- Spokes (lagrangian plus xhatshuffle)
    for i in range(len(list_of_spoke_dict)):
        list_of_spoke_dict[i]["opt_kwargs"]["options"]["iter0_solver_options"]["lpmethod"] = 4
        list_of_spoke_dict[i]["opt_kwargs"]["options"]["iter0_solver_options"]["solutiontype"] = 2
        list_of_spoke_dict[i]["opt_kwargs"]["options"]["iterk_solver_options"]["lpmethod"] = 4
        list_of_spoke_dict[i]["opt_kwargs"]["options"]["iterk_solver_options"]["solutiontype"] = 2
    #CPLEX-specific feasibility tolerance for the xhatter (tighten wenn moeglich)
    xhatshuffle_spoke["opt_kwargs"]["options"]["iterk_solver_options"]["EpRHS"] = 1e-2
    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    if wheel.global_rank == 0:  # we are the reporting hub rank
        print(f"BestInnerBound={wheel.BestInnerBound} and BestOuterBound={wheel.BestOuterBound}")

    if emprise_config["write_solution"]:
        print("### Writing EMPRISE results to " + os.path.join(emprise_config["path_result_dir"], "emprise_first_stage.csv") + " ! ###")
        wheel.write_first_stage_solution(os.path.join(emprise_config["path_result_dir"], "emprise_first_stage.csv"))
        # wheel.write_tree_solution(os.path.join(emprise_config["path_result_dir"], "emprise_full_solution.csv"))


if __name__ == "__main__":
    main()
