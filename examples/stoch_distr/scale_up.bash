# How to run this bash script:
### This code is used for a paper

SOLVERNAME=xpress

# 1st example: This runs a scalable example on 6 spokes without ensuring that xhat is feasible
mpiexec -np 6 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 20 --num-admm-subproblems 5 --default-rho 10 --solver-name ${SOLVERNAME} --max-iterations 100 --scalable --xhatxbar --lagrangian --mnpr 20 --rel-gap 0.01

# 2nd example: This runs a scalable example on 6 spokes by ensuring that xhat is feasible
mpiexec -np 6 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 20 --num-admm-subproblems 5 --default-rho 10 --solver-name ${SOLVERNAME} --max-iterations 100 --scalable --xhatxbar --lagrangian --mnpr 8 --ensure-xhat-feas --rel-gap 0.01