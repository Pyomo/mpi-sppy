#!/bin/bash

SOLVERNAME="cbc"

# taking defaults for most farmer args
# --reduced-costs --rc-fixer --rc-fix-fraction-iter0=0.2 --rc-fix-fraction-iterK=0.2 --rc-progressive-fix-fraction 
#mpiexec -np 3 python -m mpi4py farmer_cylinders.py  --linearize-proximal-terms --proximal-linearization-tolerance=1e-3 --num-scens 3 --max-iterations=50 --crops-mult=10 --tracking-folder="tracking_test/" --ph-track-progress --track-convergence=1 --track-nonants=1 --xhatshuffle --lagrangian --bundles-per-rank=0 --default-rho=1 --rel-gap=-1 --abs-gap=-1 --solver-name=${SOLVERNAME}

mpiexec -np 3 python -m mpi4py farmer_cylinders.py --num-scens 3 --max-iterations=5 --crops-mult=3 --xhatshuffle --reduced-costs --bundles-per-rank=0 --default-rho=.2 --rel-gap=-1 --abs-gap=-1 --solver-name=${SOLVERNAME} --ph-track-progress --track-reduced-costs=1 --track-xbars=1 --linearize-proximal-terms

#mpiexec -np 3 python -m mpi4py farmer_cylinders.py --linearize-proximal-terms --num-scens 3 --max-iterations=50 --crops-mult=1  --xhatshuffle --lagrangian --bundles-per-rank=0 --default-rho=.1 --rel-gap=-1 --abs-gap=-1 --solver-name=${SOLVERNAME}
