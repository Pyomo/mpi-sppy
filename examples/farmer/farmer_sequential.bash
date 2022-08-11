#!/bin/bash

SOLVERNAME="cplex"
XHF="xhatsequential"
CL="0.95"

echo "Starting Bayraksan and Morton sequential sampling."
python farmer_seqsampling.py --num-scens 3 --EF-solver-name ${SOLVERNAME} --BM-h 2  --BM-q 1.3 --xhat1-file ${XHF} --confidence-level ${CL} --BM-vs-BPL BM

echo "Starting zhat4xhat"
python -m mpisppy.confidence_intervals.zhat4xhat farmer --xhatpath ${XHF}.npy --EF-solver-name ${SOLVERNAME} --branching-factors 10 --num-samples 5 --confidence-level ${CL}

echo ""
echo "Starting Bayraksan and Pierre Louis sequential sampling."
python farmer_seqsampling.py --num-scens 3 --EF-solver-name ${SOLVERNAME} --BPL-c0 25 --BPL-eps 100 --xhat1-file ${XHF} --confidence-level ${CL} --BM-vs-BPL BPL
echo "Exit early"
exit

echo "Starting zhat4xhat"
python -m mpisppy.confidence_intervals.zhat4xhat farmer --xhatpath ${XHF}.npy --EF-solver-name ${SOLVERNAME} --branching-factors 10 --num-samples 5 --confidence-level ${CL}
