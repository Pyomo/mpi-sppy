#!/bin/bash

SOLVERNAME="cplex"
XHF="xhatsequential"
CL="0.95"

echo "Starting Bayraksan and Morton sequential sampling."
python aircond_sequential.py  --solver-name ${SOLVERNAME} --branching-factors 3 2 --BM-h 2  --BM-q 1.3 --xhat1-file ${XHF} --confidence-level ${CL} --seed=1134 --BM-eps 0.02 --BM-eps-prime 0.01 --BM-hprime 0.1 --BM-vs-BPL BM
echo ""
echo "Starting zhat4xhat on the xhat"
python -m mpisppy.confidence_intervals.zhat4xhat mpisppy.tests.examples.aircond ${XHF}.npy --solver-name ${SOLVERNAME} --branching-factors 3 2 --num-samples 5 --confidence-level ${CL}
echo ""
echo "Now doing MMW on the same xhat"
python -m mpisppy.confidence_intervals.mmw_conf mpisppy.tests.examples.aircond ${XHF}.npy ${SOLVERNAME} --MMW-num-batches 5 --MMW-batch-size 10 --confidence-level ${CL} --start-scen 6 --branching-factors 3 2
echo ""
echo "EXITING EARLY!!!"
exit
echo "========================================================"
echo "Starting Bayraksan and Pierre Louis sequential sampling."
python aircond_sequential.py --solver-name ${SOLVERNAME} --branching-factors 3 2 --BPL-c0 25 --BPL-eps 100 --xhat1-file ${XHF} --confidence-level ${CL} --seed=1134 --BPL-eps 0.5 --BM-vs-BPL BPL
echo ""
echo "Starting zhat4xhat on the xhat"
python -m mpisppy.confidence_intervals.zhat4xhat mpisppy.tests.examples.aircond ${XHF}.npy --solver-name ${SOLVERNAME} --branching-factors 3 2 --num-samples 5 --confidence-level ${CL}
