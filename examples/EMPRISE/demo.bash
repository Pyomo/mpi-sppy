#!/bin/bash
SOLVERNAME="cplex_persistent"

mpiexec -n 24 python -m mpi4py /home/phaertel/emprise/emprise_cylinders.py --max-solver-threads=8 --bundles-per-rank=1 --max-iterations=50 --default-rho=75 --with-xhatshuffle --with-lagrangian --solver-name=${SOLVERNAME} >/home/phaertel/emprise/job_files/job_emprise.out 2>&1
