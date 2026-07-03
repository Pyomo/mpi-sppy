#!/bin/bash
# Run the farmer bootstrap example (needs the statdist library).
# Pass a solver name as the first argument (default: cplex_direct).

SOLVER=${1:-cplex_direct}
BOOT="python -m mpisppy.confidence_intervals.bootsp.user_boot"
COMMON="--max-count 300 --candidate-sample-size 5 --sample-size 50 \
        --subsample-size 10 --nB 20 --alpha 0.1 --seed-offset 100 \
        --crops-multiplier 1 --yield-cv 0.1 --solver-name ${SOLVER}"

echo "Serial, compute xhat within user_boot (empirical Bagging_with_replacement)"
echo
time ${BOOT} farmer ${COMMON} --boot-method Bagging_with_replacement
echo
echo "========================"
echo
echo "Parallel batches with mpiexec -np 2 (empirical Bagging_with_replacement)"
echo
time mpiexec -np 2 python -m mpi4py \
    -m mpisppy.confidence_intervals.bootsp.user_boot \
    farmer ${COMMON} --boot-method Bagging_with_replacement
echo
echo "========================"
echo
echo "Smoothed coverage simulation from a json file (Smoothed_bagging)"
echo
time python -m mpisppy.confidence_intervals.bootsp.simulate_boot smoothed_farmer.json
