# DCAP (dynamic capacity acquisition and assignment)

DCAP is a two-stage stochastic *integer* program from the SIPLIB test problem
library. It models dynamic capacity acquisition and assignment under
uncertainty: over a planning horizon, capacity is acquired for a set of
resources (the here-and-now, first-stage decision), and then tasks are assigned
to resources in each period once the (random) processing requirements are
observed (the recourse, second-stage decision).

The instances are supplied by SIPLIB in **SMPS format** (`.cor`, `.tim`,
`.sto`), so no Python model file is needed. mpi-sppy reads them directly with
its SMPS facility (`mpisppy.problem_io.smps_module`), which the generic driver
selects automatically when you pass `--smps-dir`.

## Source and references

Data from SIPLIB: https://www2.isye.gatech.edu/~sahmed/siplib/ (dcap set).

- S. Ahmed and R. Garcia, "Dynamic Capacity Acquisition and Assignment under
  Uncertainty," *Annals of Operations Research*, 124:267–283, 2003.
- S. Ahmed, M. Tawarmalani, and N. V. Sahinidis, "A Finite Branch and Bound
  Algorithm for Two-stage Stochastic Integer Programs," *Mathematical
  Programming*, 100:355–377, 2004.

## Model

Minimize expected total cost.

- **First stage** (mixed integer): for each resource `i` and period `t`,
  `x[i,t] >= 0` is the capacity acquired and `u[i,t]` is a binary setup
  indicator. These are the nonanticipative variables.
- **Second stage** (binary): `y[i,j,t]` assigns task `j` to resource `i` in
  period `t`, and `z[j,t]` is a penalty/slack for an unassigned task. Each task
  is assigned to exactly one resource (or takes the penalty), subject to the
  acquired capacity. The processing requirements (the `y` coefficients in the
  capacity constraints) are the random data described by the `.sto` file.

## Instances

Each instance name is `dcap{R}{T}{P}_{S}`, where `R` = number of resources,
`T` = number of tasks, `P` = number of time periods, and `S` = number of
scenarios. All scenarios are equally likely (`prob = 1/S`).

| Base    | Resources | Tasks | Periods | Scenario counts |
|---------|-----------|-------|---------|-----------------|
| dcap233 | 2         | 3     | 3       | 200, 300, 500   |
| dcap243 | 2         | 4     | 3       | 200, 300, 500   |
| dcap332 | 3         | 3     | 2       | 200, 300, 500   |
| dcap342 | 3         | 4     | 2       | 200, 300, 500   |

Each instance lives in its own subdirectory holding exactly one `.cor`/`.tim`/
`.sto` trio (the SMPS reader expects a single trio per directory). The
redundant CPLEX-LP core files shipped by SIPLIB are not included.

## Running

The problems are two-stage MIPs with integer first-stage variables, so they
need a MIP solver (e.g. gurobi, cplex, or xpress). Run from this directory.

Progressive hedging with a Lagrangian outer-bound spoke and an xhat inner-bound
spoke (3 MPI ranks):

```bash
mpiexec -np 3 python -u -m mpi4py ../../mpisppy/generic_cylinders.py \
    --smps-dir dcap233_200 --solver-name gurobi \
    --max-iterations 20 --default-rho 1 \
    --lagrangian --xhatshuffle --rel-gap 1e-4 --intra-hub-conv-thresh -0.1
```

Solve the extensive form directly (small solvers may hit size limits on the
full scenario set):

```bash
python -m mpisppy.generic_cylinders --smps-dir dcap233_200 \
    --solver-name gurobi --EF
```

See `dcap_demo.bash` for a ready-to-run example.
