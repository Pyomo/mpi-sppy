#!/bin/bash
# Recommended ADMM runs via generic_cylinders --admm.
# The legacy custom driver distr_admm_cylinders.py is still in this
# directory for reference; see doc/src/generic_admm.rst for the
# rationale and a tutorial.

SOLVERNAME=xpress

# Hard-wired example (2/3/4 regions: change --num-scens accordingly)
#mpiexec -np 3 python -u -m mpi4py ../../mpisppy/generic_cylinders.py \
#    --module-name distr --admm --num-scens 3 \
#    --default-rho 10 --solver-name $SOLVERNAME --max-iterations 50 \
#    --xhatxbar --lagrangian --rel-gap 0.05 --ensure-xhat-feas

# Scalable example with xhatxbar (inner bound) and lagrangian (outer bound)
mpiexec -np 6 python -u -m mpi4py ../../mpisppy/generic_cylinders.py \
    --module-name distr --admm --num-scens 5 \
    --default-rho 10 --solver-name $SOLVERNAME --max-iterations 50 \
    --xhatxbar --lagrangian --mnpr 6 --rel-gap 0.05 --scalable --ensure-xhat-feas
