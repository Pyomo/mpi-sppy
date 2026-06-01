#!/bin/bash
# Example: stochastic ADMM via generic_cylinders with the stage2_ef_solver_name
# option enabled.
#
# Why stage2ef matters here:
#   stoch-admm wraps the 2-stage stoch_distr problem into a 3-level scenario
#   tree (ROOT -> per-stoch-scenario nodes -> per-admm-subproblem nodes).
#   That makes the problem "multistage" in mpi-sppy's view, so xhatshuffle's
#   stage2_ef path applies: for each stage-2 node it builds and solves an EF
#   that re-couples all ADMM subproblems for that stochastic outcome with
#   the root nonants fixed.  Without stage2ef, xhatshuffle would evaluate
#   the (stoch_scen, admm_sub) "scenarios" independently and lose the ADMM
#   consensus coupling, producing an invalid inner bound.
#
# Note: `num_admm_subproblems == num_stoch_scens` keeps things simple here.
# The stage2ef rank-assignment assertion in mpisppy/extensions/xhatbase.py
# uses `branching_factors[1]` (num_admm_subproblems after wrapper
# augmentation) and compares it to the locally-observed count of stage-2
# nodes (== num_stoch_scens with one rank per cylinder).  Generalizing to
# unequal counts requires care with --np.
#
# We deliberately do *not* pass --branching-factors: setup_stoch_admm
# publishes the augmented BFs back to cfg, so xhatshuffle's stage2ef path
# sees the correct tree shape automatically.

set -e

SOLVER=${SOLVERNAME:-gurobi_persistent}
N=3   # num_admm_subproblems == num_stoch_scens

mpiexec -np 3 python -u -m mpi4py ../../mpisppy/generic_cylinders.py \
    --module-name stoch_distr --stoch-admm \
    --num-admm-subproblems ${N} --num-stoch-scens ${N} \
    --default-rho 10 --max-iterations 50 \
    --solver-name ${SOLVER} \
    --lagrangian --xhatshuffle --stage2-ef-solver-name ${SOLVER} \
    --rel-gap 0.01 --ensure-xhat-feas
