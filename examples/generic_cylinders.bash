#!/bin/bash
# This runs a few command lines to illustrate the use of generic_cylinders.py

SOLVER="xpress_persistent"
SPB=1

#  echo "^^^ sslp bounds ^^^"
# cd sslp
# mpiexec -np 15 python -u -m mpi4py ../../mpisppy/generic_cylinders.py --module-name sslp --sslp-data-path ./data --instance-name sslp_15_45_10 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 1 --default-rho 1 --xhatshuffle --rel-gap -0.01 --fwph-hub --lagrangian #--tee-rank0-solves
# cd ..
# exit

#echo "^^^ netdes bounds ^^^"
#cd netdes
#mpiexec -np 3 python -u -m mpi4py ../../mpisppy/generic_cylinders.py --module-name netdes --netdes-data-path ./data --instance-name network-10-20-L-01 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01

# now UC
echo "^^^ not-so-cool UC bounds (you can do a lot better) ^^^"
# I'm not sure why I can only find uc_funcs from the directory it is in...
cd uc
mpiexec -np 10 python -u -m mpi4py ../../mpisppy/generic_cylinders.py --module-name uc_funcs --num-scens 5 --solver-name ${SOLVER} --max-iterations 50 --max-solver-threads 1 --default-rho 1e-0 --xhatshuffle --rel-gap -0.01 --fwph-hub --use-primal-dual-rho-updater --primal-dual-rho-update-threshold=2.0
cd ..


exit

# begin gradient based rho demo
# NOTE: you need to have pynumero installed!
# gradient rho, and particularly the dynamic version, is agressive,
# so you probably need ph-ob instead of lagrangian
# and you probably need xhat-xbar instead of xhatshuffle
# Note that the order stat is set to zero.
# For farmer, just plain old rho=1 is usually better than gradient rho

# First we are going to get an xhat, which is needed for grad_rho
cd farmer
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01 --solution-base-name farmer_nonants

mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 --bundles-per-rank=0 --max-iterations=100 --default-rho=1 --solver-name=${SOLVER} --xhatpath=./farmer_nonants.npy --grad-order-stat 0.0 --xhatxbar --ph-ob --max-stalled-iters 5000 --grad-rho-setter --rel-gap 0.001

# now do it again, but this time using dynamic rho

mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 --bundles-per-rank=0 --max-iterations=100 --default-rho=1 --solver-name=${SOLVER} --xhatpath=./farmer_nonants.npy --grad-order-stat 0.0 --xhatxbar --ph-ob --max-stalled-iters 5000 --grad-rho-setter --rel-gap 0.001 --dynamic-rho-dual-crit --dynamic-rho-dual-thresh 0.1

cd ..
# end gradient based rho demo

