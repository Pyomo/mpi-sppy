#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os


def main(argc, argv):
    # You can install amplpy with "python -m pip install amplpy"
    from amplpy import AMPL

    os.chdir(os.path.dirname(__file__) or os.curdir)
    model_directory = os.path.join(os.curdir, "models", "qpmv")

    """
    # If you are not using amplpy.modules, and the AMPL installation directory
    # is not in the system search path, add it as follows:
    from amplpy import add_to_path
    add_to_path(r"full path to the AMPL installation directory")
    """

    # Create an AMPL instance
    ampl = AMPL()

    # Number of steps of the efficient frontier
    steps = 10

    ampl.set_option("reset_initial_guesses", True)
    ampl.set_option("send_statuses", False)
    ampl.set_option("solver", "cplex")

    # Load the AMPL model from file
    ampl.read(os.path.join(model_directory, "qpmv.mod"))
    ampl.read(os.path.join(model_directory, "qpmvbit.run"))

    # Set tables directory (parameter used in the script above)
    ampl.get_parameter("data_dir").set(model_directory)
    # Read tables
    ampl.read_table("assetstable")
    ampl.read_table("astrets")

    portfolio_return = ampl.getVariable("portret")
    average_return = ampl.get_parameter("averret")
    target_return = ampl.get_parameter("targetret")
    variance = ampl.get_objective("cst")

    # Relax the integrality
    ampl.set_option("relax_integrality", True)
    # Solve the problem
    ampl.solve()
    # Calibrate the efficient frontier range
    minret = portfolio_return.value()
    maxret = ampl.get_value("max {s in stockall} averret[s]")
    stepsize = (maxret - minret) / steps
    returns = [None] * steps
    variances = [None] * steps
    for i in range(steps):
        print(f"Solving for return = {maxret - i * stepsize:g}")
        # Set target return to the desired point
        target_return.set(maxret - i * stepsize)
        ampl.eval("let stockopall:={};let stockrun:=stockall;")
        # Relax integrality
        ampl.set_option("relax_integrality", True)
        ampl.solve()
        print(f"QP result = {variance.value():g}")
        # Adjust included stocks
        ampl.eval("let stockrun:={i in stockrun:weights[i]>0};")
        ampl.eval("let stockopall:={i in stockrun:weights[i]>0.5};")
        # Set integrality back
        ampl.set_option("relax_integrality", False)
        # Solve the problem
        ampl.solve()
        # Check if the problem was solved successfully
        if ampl.solve_result != "solved":
            raise Exception(f"Failed to solve (solve_result: {ampl.solve_result})")
        print(f"QMIP result = {variance.value():g}")
        # Store data of corrent frontier point
        returns[i] = maxret - (i - 1) * stepsize
        variances[i] = variance.value()

    # Display efficient frontier points
    print("RETURN    VARIANCE")
    for i in range(steps):
        print(f"{returns[i]:-6f}  {variances[i]:-6f}")


if __name__ == "__main__":
    try:
        main(len(sys.argv), sys.argv)
    except Exception as e:
        print(e)
        raise
