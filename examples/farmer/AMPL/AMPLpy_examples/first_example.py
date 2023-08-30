#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os


def main(argc, argv):
    # You can install amplpy with "python -m pip install amplpy"
    from amplpy import AMPL

    os.chdir(os.path.dirname(__file__) or os.curdir)
    model_directory = os.path.join(os.curdir, "models")

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

    # Read the model and data files.
    ampl.read(os.path.join(model_directory, "diet.mod"))
    ampl.read_data(os.path.join(model_directory, "diet.dat"))

    # Solve
    ampl.solve()
    if ampl.solve_result != "solved":
        raise Exception(f"Failed to solve (solve_result: {ampl.solve_result})")

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
    if ampl.solve_result != "solved":
        raise Exception(f"Failed to solve (solve_result: {ampl.solve_result})")
    print("New objective value:", totalcost.value())

    # Reassign data - all instances
    elements = [3, 5, 5, 6, 1, 2, 5.01, 4.55]
    cost.set_values(elements)
    print("Updated all costs.")

    # Resolve and display objective
    ampl.solve()
    if ampl.solve_result != "solved":
        raise Exception(f"Failed to solve (solve_result: {ampl.solve_result})")
    print("New objective value:", totalcost.value())

    # Get the values of the variable Buy in a dataframe object
    buy = ampl.get_variable("Buy")
    df = buy.get_values()
    # Print as pandas dataframe
    print(df.to_pandas())

    # Get the values of an expression into a DataFrame object
    df2 = ampl.get_data("{j in FOOD} 100*Buy[j]/Buy[j].ub")
    # Print as pandas dataframe
    print(df2.to_pandas())


if __name__ == "__main__":
    try:
        main(len(sys.argv), sys.argv)
    except Exception as e:
        print(e)
        raise
