#!/bin/bash
# This runs a few command lines to illustrate the use of generic_cylinders.py

SOLVER="cplex"
SPB=1

echo "^^^ Multi-stage AirCond ^^^"
mpiexec -np 3 python -m mpi4py ../mpisppy/generic_cylinders.py --module-name mpisppy.tests.examples.aircond --branching-factors "3 3 3" --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatxbar --rel-gap 0.01 --solution-base-name aircond_nonants
# --xhatshuffle --stag2EFsolvern

echo "^^^ Multi-stage AirCond, pickle  the scenarios ^^^"
mpiexec -np 3 python -m mpi4py ../mpisppy/generic_cylinders.py --module-name mpisppy.tests.examples.aircond --branching-factors "3 3 3" --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatxbar --rel-gap 0.01 --solution-base-name aircond_nonants --pickle-scenarios-dir aircond/pickles

echo "^^^ Multi-stage AirCond, bundle the scenarios ^^^"
mpiexec -np 3 python -m mpi4py ../mpisppy/generic_cylinders.py --module-name mpisppy.tests.examples.aircond --branching-factors "3 3 3" --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatxbar --rel-gap 0.01 --solution-base-name aircond_nonants --scenarios-per-bundle 9

echo "^^^ Multi-stage AirCond, bundle the scenarios and write ^^^"
mpiexec -np 3 python -m mpi4py ../mpisppy/generic_cylinders.py --module-name mpisppy.tests.examples.aircond --branching-factors "3 3 3" --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatxbar --rel-gap 0.01 --solution-base-name aircond_nonants --pickle-scenarios-dir aircond/pickles --scenarios-per-bundle 9

#### HEY! check on error messages for bad bundle sizes

echo "^^^ write scenario lp and nonant json files ^^^"
cd sizes
python ../../mpisppy/generic_cylinders.py --module-name sizes --num-scens 3 --default-rho 1 --solver-name ${SOLVER} --max-iterations 0 --scenario-lpfiles
cd ..

echo "^^^ pickle sizes bundles ^^^"
cd sizes
python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name sizes --num-scens 10 --pickle-bundles-dir sizes_pickles --scenarios-per-bundle 5 --default-rho 1
cd ..

echo "^^^ unpickle the sizes bundles and write the lp and nonant files ^^^"
# note that numscens need to match the number before pickling...
# so does scenarios per bundle
cd sizes
python ../../mpisppy/generic_cylinders.py --module-name sizes --num-scens 10 --default-rho 1 --solver-name ${SOLVER} --max-iterations 0 --scenario-lpfiles --unpickle-bundles-dir sizes_pickles --scenarios-per-bundle 5
cd ..

echo "^^^ pickle the scenarios ^^^"
cd farmer
python ../../mpisppy/generic_cylinders.py --module-name farmer --pickle-scenarios-dir farmer_pickles --crops-mult 2 --num-scens 10
cd ..

echo "^^^ pickle the scenarios a little faster ^^^"
# for larger models, the speed up is more
cd farmer
mpiexec -np 2 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --pickle-scenarios-dir farmer_pickles --crops-mult 2 --num-scens 10
cd ..

echo "^^^ used pickled scenarios ^^^"
cd farmer
# note that crops-mult would be ignored
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 10 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01 --unpickle-scenarios-dir farmer_pickles
cd ..

echo "^^^ use proper bundles without writing ^^^"
cd sslp
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name sslp --sslp-data-path ./data --instance-name sslp_15_45_10 --scenarios-per-bundle $SPB --default-rho 1 --solver-name ${SOLVER} --max-iterations 5 --lagrangian --xhatshuffle --rel-gap 0.001
cd ..

echo "^^^ write pickle bundles ^^^"
cd sslp
python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name sslp --sslp-data-path ./data --instance-name sslp_15_45_10 --pickle-bundles-dir sslp_pickles --scenarios-per-bundle $SPB --default-rho 1
cd ..

echo "^^^ write pickle bundles faster ^^^"
# np needs to divide the number of scenarios, btw
cd sslp
mpiexec -np 2 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name sslp --sslp-data-path ./data --instance-name sslp_15_45_10 --pickle-bundles-dir sslp_pickles --scenarios-per-bundle $SPB --default-rho 1
cd ..

echo "^^^ read pickle bundles ^^^"
cd sslp
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name sslp --sslp-data-path ./data --instance-name sslp_15_45_10 --unpickle-bundles-dir sslp_pickles --scenarios-per-bundle $SPB --default-rho 1 --solver-name=${SOLVER} --max-iterations 5 --lagrangian --xhatshuffle --rel-gap 0.001
cd ..

cd farmer

mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01 --solution-base-name farmer_nonants

echo "^^^ sep rho not dynamic ^^^"
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 --bundles-per-rank=0 --max-iterations=100 --default-rho=1 --solver-name=${SOLVER} --xhatpath=./farmer_nonants.npy --grad-order-stat 0.0 --xhatxbar --ph-ob --max-stalled-iters 5000 --sep-rho --rel-gap 0.001


echo "^^^ sep rho dynamic ^^^"
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 --bundles-per-rank=0 --max-iterations=100 --default-rho=1 --solver-name=${SOLVER} --xhatpath=./farmer_nonants.npy --grad-order-stat 0.0 --xhatxbar --ph-ob --max-stalled-iters 5000 --sep-rho --rel-gap 0.001 --dynamic-rho-dual-crit --dynamic-rho-dual-thresh 0.1

mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 --bundles-per-rank=0 --max-iterations=100 --default-rho=1 --solver-name=${SOLVER} --xhatpath=./farmer_nonants.npy --grad-order-stat 0.0 --xhatxbar --ph-ob --max-stalled-iters 5000 --grad-rho-setter --rel-gap 0.001

# now do it again, but this time using dynamic rho
echo "^^^ grad rho dynamic ^^^"
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 --bundles-per-rank=0 --max-iterations=100 --default-rho=1 --solver-name=${SOLVER} --xhatpath=./farmer_nonants.npy --grad-order-stat 0.0 --xhatxbar --ph-ob --max-stalled-iters 5000 --grad-rho-setter --rel-gap 0.001 --dynamic-rho-dual-crit --dynamic-rho-dual-thresh 0.1

echo "^^^ sensi rho dynamic ^^^"
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 --bundles-per-rank=0 --max-iterations=100 --default-rho=1 --solver-name=${SOLVER} --xhatpath=./farmer_nonants.npy --grad-order-stat 0.0 --xhatxbar --ph-ob --max-stalled-iters 5000 --sensi-rho --rel-gap 0.001 --dynamic-rho-dual-crit --dynamic-rho-dual-thresh 0.1

cd ..

echo "^^^ netdes sensi-rho ^^^"
cd netdes
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name netdes --netdes-data-path ./data --instance-name network-10-20-L-01 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01 --sensi-rho
cd ..

echo "^^^ farmer sensi-rho ^^^"
mpiexec -np 3 python -m mpi4py ../mpisppy/generic_cylinders.py --module-name farmer/farmer --num-scens 3 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01 --sensi-rho


# sslp EF
echo "^^^ sslp ef ^^^"
cd sslp
python ../../mpisppy/generic_cylinders.py --module-name sslp --sslp-data-path ./data --EF --instance-name sslp_15_45_10 --EF-solver-name ${SOLVER}
cd ..

echo "^^^ sslp bounds ^^^"
cd sslp
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name sslp --sslp-data-path ./data --instance-name sslp_15_45_10 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01
cd ..

# netdes EF
echo "^^^ netdes ef ^^^"
cd netdes
python ../../mpisppy/generic_cylinders.py --module-name netdes --netdes-data-path ./data --EF --instance-name network-10-20-L-01 --EF-solver-name ${SOLVER}
cd ..

echo "^^^ netdes bounds ^^^"
cd netdes
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name netdes --netdes-data-path ./data --instance-name network-10-20-L-01 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01

# sizes EF
echo "^^^ sizes ef ^^^"
python ../mpisppy/generic_cylinders.py --module-name sizes/sizes --EF --num-scens 3 --EF-solver-name ${SOLVER}

# sizes with a custom rho_setter
echo "^^^ sizes custom rho_setter ^^^"
mpiexec -np 3 python -m mpi4py ../mpisppy/generic_cylinders.py --module-name sizes/sizes --num-scens 3 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.0001

# Note: you still need to name a module to get help
echo "^^^ help ^^^"
python -m mpi4py ../mpisppy/generic_cylinders.py --module-name farmer/farmer --help

# A not-so-useful run that does not use MPI, so only runs a hub
echo "^^^ hub ^^^"
python -m mpi4py ../mpisppy/generic_cylinders.py --module-name farmer/farmer --num-scens 3 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 

# same thing, but with bounds
echo "^^^ farmer bounds ^^^"
mpiexec -np 3 python -m mpi4py ../mpisppy/generic_cylinders.py --module-name farmer/farmer --num-scens 3 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01

# try a simple Hydro
pwd
cd hydro
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name hydro --solver-name ${SOLVER} --max-iterations 100 --bundles-per-rank=0 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.001 --branching-factors "3 3" --stage2EFsolvern ${SOLVER}
cd ..


# now UC
echo "^^^ not-so-cool UC bounds (you can do a lot better) ^^^"
# I'm not sure why I can only find uc_funcs from the directory it is in...
cd uc
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name uc_funcs --num-scens 5 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01
cd ..

echo "^^^ not-so-cool UC with mipgapper ^^^"
# I'm not sure why I can only find uc_funcs from the directory it is in...
cd uc
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name uc_funcs --num-scens 5 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.001 --mipgaps-json phmipgaps.json
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

