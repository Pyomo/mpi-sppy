#!/bin/bash
### This code is used for a paper

SOLVERNAME=gurobi_persistent
ITERS=100

sizes=(8 128)
scencnts=(16 256)
numprocs=(3 12 36)

for size in "${sizes[@]}"; do
    for np in "${numprocs[@]}"; do
        for scens in "${scencnts[@]}"; do
            # The comparison with EF
            echo "Starting EF for: np=${np}, scens=${scens}, size=${size}"
            python stoch_distr_ef.py --num-stoch-scens ${scens} --num-admm-subproblems 5  --solver-name ${SOLVERNAME} --scalable --mnpr ${size} --ensure-xhat-feas
            # The PH example
            echo "^^^^**** mpiexec -np ${np} python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens  ${scens} --num-admm-subproblems 5 --default-rho 10 --solver-name ${SOLVERNAME} --max-iterations ${ITERS} --scalable --xhatxbar --lagrangian --mnpr ${size} --ensure-xhat-feas --rel-gap 0.01"
            mpiexec -np ${np} python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens  ${scens} --num-admm-subproblems 5 --default-rho 10 --solver-name ${SOLVERNAME} --max-iterations ${ITERS} --scalable --xhatxbar --lagrangian --mnpr ${size} --ensure-xhat-feas --rel-gap 0.01
	      done
    done
done
