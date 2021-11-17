#!/bin/bash

SOLVERNAME="cplex"
XHF="xhatsequential"

python farmer_seqsampling.py 3 --EF-solver-name ${SOLVERNAME} --BM-h 2  --BM-q 1.3 --xhat1-file ${XHF}
echo "Starting xhat4xhat"
python -m mpisppy.confidence_intervals.zhat4xhat afarmer ${XHF}.npy --solver-name ${SOLVERNAME} --branching-factors 10 --num-samples 5 --confidence-level 0.95
