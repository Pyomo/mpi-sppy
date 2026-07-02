#!/bin/bash
# Run the (statdist-free) schultz bootstrap example.
# Pass a solver name as the first argument (default: gurobi_direct).

SOLVER=${1:-gurobi_direct}
BOOT="python -m mpisppy.confidence_intervals.bootsp.user_boot"
COMMON="--max-count 50 --candidate-sample-size 1 --sample-size 30 \
        --subsample-size 10 --nB 20 --alpha 0.1 --seed-offset 100 \
        --solver-name ${SOLVER}"

echo "Serial, compute xhat within user_boot (Classical_quantile)"
echo
time ${BOOT} unique_schultz ${COMMON} --boot-method Classical_quantile
echo
echo "========================"
echo
echo "Parallel batches with mpiexec -np 2 (Bagging_with_replacement)"
echo
time mpiexec -np 2 python -m mpi4py \
    -m mpisppy.confidence_intervals.bootsp.user_boot \
    unique_schultz ${COMMON} --boot-method Bagging_with_replacement
echo
echo "========================"
echo
echo "Coverage simulation from a json file"
echo
time python -m mpisppy.confidence_intervals.bootsp.simulate_boot unique_schultz.json
