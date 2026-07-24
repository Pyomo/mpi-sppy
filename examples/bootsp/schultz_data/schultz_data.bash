#!/bin/bash
# Run the data-file schultz bootstrap example (reads schultz_data.csv).
# Pass a solver name as the first argument (default: gurobi_direct).
# Regenerate the dataset first with: python schultz_data_generator.py

SOLVER=${1:-gurobi_direct}
BOOT="python -m mpisppy.confidence_intervals.bootsp.user_boot"
COMMON="--max-count 200 --candidate-sample-size 5 --sample-size 100 \
        --subsample-size 20 --nB 20 --alpha 0.1 --seed-offset 100 \
        --solver-name ${SOLVER}"

echo "Serial, xhat from part of the data, bootstrap CI on the rest"
echo
time ${BOOT} schultz_data ${COMMON} --boot-method Bagging_with_replacement
echo
echo "========================"
echo
echo "Coverage simulation from a json file"
echo
time python -m mpisppy.confidence_intervals.bootsp.simulate_boot schultz_data.json
