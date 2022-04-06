#!/bin/bash

SOLVERNAME=cplex

mpiexec --oversubscribe --np 3 python -m mpi4py hydro_cylinders.py --branching-factors 3 3 --bundles-per-rank=0 --max-iterations=100 --default-rho=1 --with-xhatshuffle --with-lagrangian --solver-name=${SOLVERNAME} --stage2EFsolvern=${SOLVERNAME}

# including this option will cause the upper bounder to solve the EF since there are only three ranks in total.
#--stage2EFsolvern=${SOLVERNAME}
