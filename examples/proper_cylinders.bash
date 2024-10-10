#!/bin/bash

SOLVER=cplex

# Note: you still need to name a module to get help
echo "^^^ help ^^^"
python -m mpi4py ../mpisppy/proper_cylinders.py --module-name farmer/farmer --help

echo "^^^ use proper bundles without writing ^^^"
cd sslp
mpiexec -np 3 python -m mpi4py ../../mpisppy/proper_cylinders.py --module-name sslp --sslp-data-path ./data --instance-name sslp_15_45_10 --proper-no-files --scenarios-per-bundle 5 --default-rho 1 --solver-name ${SOLVER} --max-iterations 5 --lagrangian --xhatshuffle --rel-gap 0.001
cd ..

echo "^^^ write pickle bundles ^^^"
cd sslp
python -m mpi4py ../../mpisppy/proper_cylinders.py --module-name sslp --sslp-data-path ./data --instance-name sslp_15_45_10 --pickle-bundles-dir sslp_pickles --scenarios-per-bundle 5 --default-rho 1
cd ..

echo ""
echo "^^^ xxxxxx race conditions???? write pickle bundles faster XXXXXX^^^"
echo ""
# np needs to divide the number of scenarios, btw
#cd sslp
#mpiexec -np 2 python -m mpi4py ../../mpisppy/proper_cylinders.py --module-name sslp --sslp-data-path ./data --instance-name sslp_15_45_10 --pickle-bundles-dir sslp_pickles --scenarios-per-bundle 5 --default-rho 1
#cd ..

echo "^^^ read pickle bundles ^^^"
cd sslp
mpiexec -np 3 python -m mpi4py ../../mpisppy/proper_cylinders.py --module-name sslp --sslp-data-path ./data --instance-name sslp_15_45_10 --unpickle-bundles-dir sslp_pickles --scenarios-per-bundle 5 --default-rho 1 --solver-name=${SOLVER} --max-iterations 5 --lagrangian --xhatshuffle --rel-gap 0.001
cd ..
