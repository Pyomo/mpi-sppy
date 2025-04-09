#!/bin/bash
# This is mainly to demonstrate what loose agnostic files look like and
#   how to use them with agnostic_cylinders.py.
# To do that, we write the files based on a Pyomo model, then
# read them in.
# Note: if you actually have a Pyomo model, you probably don't want to do
#  it this way since you would have had to have written most of the
# functions (e.g. scenario_creator) anyway.
# If you are using some other AML, then you migth want to use the second
#   command line to read the files you wrote with your AML and
#   you can use the first command to write files as an example of the format
#   for the json files.

set -e

SOLVER=cplex

# assumes we are in the sizes directory and don't mind polluting it with 6 files
python ../../mpisppy/generic_cylinders.py --module-name sizes_expression --num-scens 3 --default-rho 1 --solver-name ${SOLVER} --max-iterations 0 --scenario-lp-mps-files

# By specifying the module to be mps_module we will read files for the problem
#  from the specified mps-files-directory.
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name ../../mpisppy/utils/mps_module --xhatshuffle --lagrangian --default-rho 1 --solver-name ${SOLVER} --max-iterations 10 --mps-files-directory=.
