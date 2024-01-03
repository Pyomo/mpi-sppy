#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd  # for pandas.DataFrame objects (https://pandas.pydata.org/)
import numpy as np  # for numpy.matrix objects (https://numpy.org/)


def prepare_data():
    food_df = pd.DataFrame(
        [
            ("BEEF", 3.59, 2, 10),
            ("CHK", 2.59, 2, 10),
            ("FISH", 2.29, 2, 10),
            ("HAM", 2.89, 2, 10),
            ("MCH", 1.89, 2, 10),
            ("MTL", 1.99, 2, 10),
            ("SPG", 1.99, 2, 10),
            ("TUR", 2.49, 2, 10),
        ],
        columns=["FOOD", "cost", "f_min", "f_max"],
    ).set_index("FOOD")

    # Create a pandas.DataFrame with data for n_min, n_max
    nutr_df = pd.DataFrame(
        [
            ("A", 700, 20000),
            ("C", 700, 20000),
            ("B1", 700, 20000),
            ("B2", 700, 20000),
            ("NA", 0, 50000),
            ("CAL", 16000, 24000),
        ],
        columns=["NUTR", "n_min", "n_max"],
    ).set_index("NUTR")

    amt_df = pd.DataFrame(
        np.array(
            [
                [60, 8, 8, 40, 15, 70, 25, 60],
                [20, 0, 10, 40, 35, 30, 50, 20],
                [10, 20, 15, 35, 15, 15, 25, 15],
                [15, 20, 10, 10, 15, 15, 15, 10],
                [928, 2180, 945, 278, 1182, 896, 1329, 1397],
                [295, 770, 440, 430, 315, 400, 379, 450],
            ]
        ),
        columns=food_df.index.to_list(),
        index=nutr_df.index.to_list(),
    )
    return food_df, nutr_df, amt_df


def main(argc, argv):
    # You can install amplpy with "python -m pip install amplpy"
    from amplpy import AMPL

    os.chdir(os.path.dirname(__file__) or os.curdir)

    """
    # If you are not using amplpy.modules, and the AMPL installation directory
    # is not in the system search path, add it as follows:
    from amplpy import add_to_path
    add_to_path(r"full path to the AMPL installation directory")
    """

    # Create an AMPL instance
    ampl = AMPL()

    # Set the solver to use
    solver = argv[1] if argc > 1 else "highs"
    ampl.set_option("solver", solver)

    ampl.eval(
        r"""
        set NUTR;
        set FOOD;

        param cost {FOOD} > 0;
        param f_min {FOOD} >= 0;
        param f_max {j in FOOD} >= f_min[j];

        param n_min {NUTR} >= 0;
        param n_max {i in NUTR} >= n_min[i];

        param amt {NUTR,FOOD} >= 0;

        var Buy {j in FOOD} >= f_min[j], <= f_max[j];

        minimize Total_Cost:
            sum {j in FOOD} cost[j] * Buy[j];

        subject to Diet {i in NUTR}:
            n_min[i] <= sum {j in FOOD} amt[i,j] * Buy[j] <= n_max[i];
    """
    )

    # Load the data from pandas.DataFrame objects:
    food_df, nutr_df, amt_df = prepare_data()
    # 1. Send the data from "amt_df" to AMPL and initialize the indexing set "FOOD"
    ampl.set_data(food_df, "FOOD")
    # 2. Send the data from "nutr_df" to AMPL and initialize the indexing set "NUTR"
    ampl.set_data(nutr_df, "NUTR")
    # 3. Set the values for the parameter "amt" using "amt_df"
    ampl.get_parameter("amt").set_values(amt_df)

    # Solve
    ampl.solve()

    # Get objective entity by AMPL name
    totalcost = ampl.get_objective("Total_Cost")
    # Print it
    print("Objective is:", totalcost.value())

    # Reassign data - specific instances
    cost = ampl.get_parameter("cost")
    cost.set_values({"BEEF": 5.01, "HAM": 4.55})
    print("Increased costs of beef and ham.")

    # Resolve and display objective
    ampl.solve()
    assert ampl.solve_result == "solved"
    print("New objective value:", totalcost.value())

    # Reassign data - all instances
    cost.set_values(
        {
            "BEEF": 3,
            "CHK": 5,
            "FISH": 5,
            "HAM": 6,
            "MCH": 1,
            "MTL": 2,
            "SPG": 5.01,
            "TUR": 4.55,
        }
    )

    print("Updated all costs.")

    # Resolve and display objective
    ampl.solve()
    assert ampl.solve_result == "solved"
    print("New objective value:", totalcost.value())

    # Get the values of the variable Buy in a pandas.DataFrame object
    df = ampl.get_variable("Buy").get_values().to_pandas()
    # Print them
    print(df)

    # Get the values of an expression into a pandas.DataFrame object
    df2 = ampl.get_data("{j in FOOD} 100*Buy[j]/Buy[j].ub").to_pandas()
    # Print them
    print(df2)


if __name__ == "__main__":
    try:
        main(len(sys.argv), sys.argv)
    except Exception as e:
        print(e)
        raise
