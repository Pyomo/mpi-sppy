#!/bin/bash
# runs the version that uses config.py

SOLVERNAME=cplex

mpiexec --oversubscribe --np 3 python -m mpi4py hydro_cylinders_config.py --branching-factors "3 3" --bundles-per-rank=0 --max-iterations=100 --default-rho=1 --xhatshuffle --lagrangian --solver-name=${SOLVERNAME} --stage2EFsolvern=${SOLVERNAME}
#--tee-rank0-solves

# including this option will cause the upper bounder to solve the EF since there are only three ranks in total.
#--stage2EFsolvern=${SOLVERNAME}
