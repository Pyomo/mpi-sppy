#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os


def main(argc, argv):
    # You can install amplpy with "python -m pip install amplpy"
    from amplpy import AMPL

    os.chdir(os.path.dirname(__file__) or os.curdir)
    model_directory = os.path.join(os.curdir, "models", "tracking")

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

    # Load the AMPL model from file
    ampl.read(os.path.join(model_directory, "tracking.mod"))
    # Read data
    ampl.read_data(os.path.join(model_directory, "tracking.dat"))
    # Read table declarations
    ampl.read(os.path.join(model_directory, "trackingbit.run"))
    # Set tables directory (parameter used in the script above)
    ampl.get_parameter("data_dir").set(model_directory)
    # Read tables
    ampl.read_table("assets")
    ampl.read_table("indret")
    ampl.read_table("returns")

    hold = ampl.get_variable("hold")
    ifinuniverse = ampl.get_parameter("ifinuniverse")

    # Relax the integrality
    ampl.set_option("relax_integrality", True)

    # Solve the problem
    ampl.solve()
    if ampl.solve_result != "solved":
        raise Exception(f"Failed to solve (solve_result: {ampl.solve_result})")

    objectives = list(obj for name, obj in ampl.get_objectives())
    assert objectives[0].value() == ampl.get_objective("cst").value()
    print("QP objective value", ampl.get_objective("cst").value())

    lowcutoff = 0.04
    highcutoff = 0.1

    # Get the variable representing the (relaxed) solution vector
    holdvalues = hold.get_values().to_list()
    to_hold = []
    # For each asset, if it was held by more than the highcutoff,
    # forces it in the model, if less than lowcutoff, forces it out
    for _, value in holdvalues:
        if value < lowcutoff:
            to_hold.append(0)
        elif value > highcutoff:
            to_hold.append(2)
        else:
            to_hold.append(1)
    # uses those values for the parameter ifinuniverse, which controls
    # which stock is included or not in the solution
    ifinuniverse.set_values(to_hold)

    # Get back to the integer problem
    ampl.set_option("relax_integrality", False)

    # Solve the (integer) problem
    ampl.solve()
    if ampl.solve_result != "solved":
        raise Exception(f"Failed to solve (solve_result: {ampl.solve_result})")

    print("QMIP objective value", ampl.get_objective("cst").value())


if __name__ == "__main__":
    try:
        main(len(sys.argv), sys.argv)
    except Exception as e:
        print(e)
        raise
