#!/bin/bash
# This runs a few command lines to illustrate the use of generic_cylinders.py

SOLVER="cplex"

# sizes EF
echo "^^^ sizes ef ^^^"
python ../mpisppy/generic_cylinders.py --module-name sizes/sizes --EF --num-scens 3 --EF-solver-name ${SOLVER}

echo "^^^ not-so-cool UC with mipgapper ^^^"
# I'm not sure why I can only find uc_funcs from the directory it is in...
cd uc
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name uc_funcs --num-scens 5 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.001 --mipgaps-json phmipgaps.json
cd ..
exit


# sizes with gradient-based rho setter and ph_ob
echo("HEY, TBD: we need # sizes with gradient-based rho setter and ph_ob")

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


