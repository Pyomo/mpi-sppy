#!/bin/bash
# Example: sequential sampling using mrp_generic with the farmer model.
# This is the generic-driver equivalent of farmer_sequential.bash.
#
# Solver can be overridden via the first CLI arg or the SOLVERNAME env var.
# Defaults to xpress because that is what the CI images have a usable
# license for; a developer can pass "cplex" or "gurobi" locally.

SOLVERNAME="${1:-${SOLVERNAME:-xpress}}"
CL="0.95"
# Path to the farmer module (relative to this directory)
FARMER="farmer"

echo "===== BM sequential sampling (EF) ====="
python -m mpisppy.mrp_generic \
    --module-name ${FARMER} \
    --num-scens 3 \
    --solver-name ${SOLVERNAME} \
    --stopping-criterion BM \
    --BM-h 2.0 \
    --BM-q 1.3 \
    --confidence-level ${CL} \
    --solution-base-name farmer_mrp_xhat

echo ""
echo "===== BPL sequential sampling (EF) ====="
python -m mpisppy.mrp_generic \
    --module-name ${FARMER} \
    --num-scens 3 \
    --solver-name ${SOLVERNAME} \
    --stopping-criterion BPL \
    --BPL-eps 100.0 \
    --BPL-c0 25 \
    --confidence-level ${CL}

echo ""
echo "===== BM sequential sampling (cylinders) ====="
mpiexec -np 3 python -m mpi4py -m mpisppy.mrp_generic \
    --module-name ${FARMER} \
    --num-scens 3 \
    --solver-name ${SOLVERNAME} \
    --xhat-method cylinders \
    --stopping-criterion BM \
    --BM-h 2.0 \
    --BM-q 1.3 \
    --confidence-level ${CL} \
    --default-rho 1 \
    --max-iterations 10 \
    --lagrangian --xhatshuffle
