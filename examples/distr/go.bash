#execute with bash go.bash

mpiexec -np 3 python -u -m mpi4py scalable_distr_admm_cylinders.py --num-scens 3 --default-rho 10 --solver-name xpress --max-iterations 200 --xhatxbar --lagrangian --mnpr 40 --rel-gap 0.005