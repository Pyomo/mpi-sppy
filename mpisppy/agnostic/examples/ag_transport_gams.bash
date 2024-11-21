#!/bin/bash

SOLVERNAME=gurobi
IT=150

python ../agnostic_cylinders.py --module-name mpisppy.agnostic.examples.transport_gams_model --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=${IT} --rel-gap 0.01 --display-progress --guest-language GAMS --gams-model-file transport_average.gms

mpiexec -np 3 python -m mpi4py ../agnostic_cylinders.py --module-name mpisppy.agnostic.examples.transport_gams_model --num-scens 3 --default-rho 0.1 --solver-name $SOLVERNAME --max-iterations=${IT} --xhatshuffle --lagrangian --rel-gap 0.01 --guest-language GAMS --gams-model-file transport_average.gms


