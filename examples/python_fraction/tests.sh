#!/bin/bash

# wrap the mpiexec run in python so the rank can be determined
# run with $ SOLVER=gurobi mpiexec -np 3 ./tests.sh
# (do chmod once)

set -euo pipefail

SOLVER="${SOLVER:-gurobi}"

# Determine MPI rank from common env vars (OpenMPI / MPICH / Slurm)
RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-}}}"
if [[ -z "${RANK}" ]]; then
  echo "Could not determine MPI rank from environment (OMPI_COMM_WORLD_RANK / PMI_RANK / SLURM_PROCID)." >&2
  exit 1
fi

echo "^^^ farmer ^^^ rank=${RANK}"

python -m scalene run \
  --outfile "scalene_rank_${RANK}.txt" \
  ../../mpisppy/generic_cylinders.py \
  --module-name ../farmer/farmer \
  --num-scens 3 \
  --solver-name "${SOLVER}" \
  --max-iterations 100 \
  --max-solver-threads 4 \
  --default-rho 1 \
  --lagrangian \
  --xhatshuffle \
  --rel-gap 0.0001

exit

==============================================
#!/bin/bash
set -e

SOLVER="gurobi"

echo "^^^ farmer ^^^"
mpiexec -np 3 python -m scalene run --outfile scalene_rank_%r.txt ../../mpisppy/generic_cylinders.py --module-name ../farmer/farmer --num-scens 3 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01 

exit

echo "^^^ sslp bounds ^^^"
cd sslp
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name sslp --sslp-data-path ./data --instance-name sslp_15_45_10 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01
cd ..

scalene is designed to attribute time to Python vs native (it can estimate time spent in compiled code called from Python). It’s often the most straightforward way to get exactly what you asked.

Run:

python -m pip install scalene
python -m scalene your_script.py [args...]


It reports per-file and per-line:

Python time

Native time (extensions, libraries like NumPy, solver bindings, etc.)

MPI note: if you run under mpiexec, you’ll get output per rank (may need to direct to separate files):

mpiexec -np 4 python -m scalene --outfile scalene_rank_%r.txt your_script.py ...


If %r isn’t supported in your shell, just set unique outfile names using env vars per rank.

This is usually the quickest route to “fraction spent in Python.”        
