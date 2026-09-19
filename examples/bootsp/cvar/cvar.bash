#!/bin/bash
# Run the CVaR bootstrap example (needs the statdist library).
# Pass a solver name as the first argument (default: cplex_direct).

SOLVER=${1:-cplex_direct}
BOOT="python -m mpisppy.confidence_intervals.bootsp.user_boot"
COMMON="--max-count 3000 --candidate-sample-size 10 --sample-size 75 \
        --subsample-size 10 --nB 20 --alpha 0.1 --seed-offset 0 \
        --solver-name ${SOLVER}"

echo "Serial, compute xhat within user_boot (empirical Bagging_with_replacement)"
echo
time ${BOOT} cvar ${COMMON} --boot-method Bagging_with_replacement
echo
echo "========================"
echo
echo "Smoothed coverage simulation from a json file (Smoothed_bagging)"
echo
time python -m mpisppy.confidence_intervals.bootsp.simulate_boot smoothed_cvar.json
