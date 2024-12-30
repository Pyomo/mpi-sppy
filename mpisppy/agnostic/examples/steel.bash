#!/bin/bash
#python ../agnostic_cylinders.py --module-name farmer4agnostic --default-rho 1 --num-scens 3 --solver-name cplex --guest-language Pyomo
# NOTE: you need the AMPL solvers!!!
#python -u ../agnostic_cylinders.py --module-name mpisppy.agnostic.examples.steel_ampl_model --default-rho 1 --num-scens 3 --seed 17 --solver-name gurobi --guest-language AMPL --ampl-model-file steel.mod --ampl-data-file "$ampl_data_file" --max-iterations 10
#could i ad a -- data file name 
ampl_data_file="steel.dat"
mpiexec -np 3 python -u ../agnostic_cylinders.py --module-name mpisppy.agnostic.examples.steel_ampl_model --default-rho 1 --num-scens 3 --seed 17 --solver-name gurobi --guest-language AMPL --ampl-model-file steel.mod --ampl-data-file "$ampl_data_file" --max-iterations 10 --rel-gap 0.01 --xhatshuffle --lagrangian --solution-base-name steel
