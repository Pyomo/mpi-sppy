#!/bin/bash

SOLVERNAME="cplex"
XHF="xhatsequential"
CL="0.95"

echo "Starting Bayraksan and Morton sequential sampling."
python aircond_seqsampling.py  --solver-name ${SOLVERNAME} --branching-factors 3 2 --BM-h 2  --BM-q 1.3 --xhat1-file ${XHF} --confidence-level ${CL} --seed=1134 --BM-vs-BPL BM
#echo "Starting xhat4xhat on the xhat"
#python -m mpisppy.confidence_intervals.zhat4xhat afarmer ${XHF}.npy --solver-name ${SOLVERNAME} --branching-factors 10 --num-samples 5 --confidence-level ${CL}
#echo ""
#echo "Starting Bayraksan and Pierre Louis sequential sampling."
#python farmer_seqsampling.py 3 --EF-solver-name ${SOLVERNAME} --BPL-c0 25 --BPL-eps 100 --xhat1-file ${XHF} --confidence-level ${CL} --BM-vs-BPL BPL
#echo "Starting xhat4xhat on the xhat"
#python -m mpisppy.confidence_intervals.zhat4xhat afarmer ${XHF}.npy --solver-name ${SOLVERNAME} --branching-factors 10 --num-samples 5 --confidence-level ${CL}
