#!/bin/bash

SOLVERNAME=gurobi

python -u ../agnostic_cylinders.py --module-name mpisppy.agnostic.examples.farmer_gams_model --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=5 --rel-gap 0.01 --display-progress --guest-language GAMS --gams-model-file farmer_average.gms

echo "^^^^lagrangian only"
mpiexec -np 2 python -m mpi4py ../agnostic_cylinders.py --module-name mpisppy.agnostic.examples.farmer_gams_model --num-scens 3 --default-rho 0.5 --solver-name $SOLVERNAME --max-iterations=30 --lagrangian --rel-gap 0.01 --guest-language GAMS --gams-model-file farmer_average.gms

echo "^^^^ xhat only"
mpiexec -np 2 python -m mpi4py ../agnostic_cylinders.py --module-name mpisppy.agnostic.examples.farmer_gams_model --num-scens 3 --default-rho 0.5 --solver-name $SOLVERNAME --max-iterations=30 --xhatshuffle  --rel-gap 0.01 --guest-language GAMS --gams-model-file farmer_average.gms

echo "^^^^ lagrangian and xhat"
mpiexec -np 3 python -m mpi4py ../agnostic_cylinders.py --module-name mpisppy.agnostic.examples.farmer_gams_model --num-scens 3 --default-rho 0.5 --solver-name $SOLVERNAME --max-iterations=30 --xhatshuffle --lagrangian --rel-gap 0.01 --guest-language GAMS --gams-model-file farmer_average.gms
