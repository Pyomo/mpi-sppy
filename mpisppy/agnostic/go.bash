#!/bin/bash
#python agnostic_cylinders.py --module-name farmer4agnostic --default-rho 1 --num-scens 3 --solver-name cplex --guest-language Pyomo
python agnostic_cylinders.py --module-name ../../examples/farmer/farmer_ampl_model --default-rho 1 --num-scens 3 --solver-name cplex --guest-language AMPL --ampl-file-name farmer.mod
