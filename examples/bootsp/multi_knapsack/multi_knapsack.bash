#!/bin/bash
# Run the multi-knapsack bootstrap example (needs the statdist library).
# The sample sizes here are small; this is just a demonstration.
# NOTE: do not be alarmed by infeasibility messages during the confidence
# interval calculations. Pass a solver name as the first argument.

SOLVER=${1:-cplex_direct}
BOOT="python -m mpisppy.confidence_intervals.bootsp.user_boot"
COMMON="--max-count 300 --candidate-sample-size 5 --sample-size 50 \
        --subsample-size 10 --nB 20 --alpha 0.1 --seed-offset 100 \
        --deterministic-data-json multi_knapsack_data.json \
        --solver-name ${SOLVER}"

echo "Serial, compute xhat within user_boot (empirical Bagging_with_replacement)"
echo
time ${BOOT} multi_knapsack ${COMMON} --boot-method Bagging_with_replacement
echo
echo "========================"
echo
echo "Smoothed coverage simulation from a json file (Smoothed_bagging)"
echo
time python -m mpisppy.confidence_intervals.bootsp.simulate_boot smoothed_multi_knapsack.json
