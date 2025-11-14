#!/bin/bash
# run the example where an GAMSpy script writes scenarios to be read by mpi-sppy

set -e

ODIR="_fromGAMS"
SOLVER="gurobi"
SOLBASE="farmer_solution_output"

empty_or_create_dir() {
  local d=$1

  # Refuse dangerous/meaningless targets
  if [[ -z "$d" || "$d" == "/" || "$d" == "." ]]; then
    echo "Refusing to operate on empty path, /, or ." >&2
    return 1
  fi

  # If the path exists but isn't a directory, bail
  if [[ -e "$d" && ! -d "$d" ]]; then
    echo "Refusing: '$d' exists and is not a directory." >&2
    return 1
  fi

  if [[ -d "$d" ]]; then
    # Empty contents but keep the directory node (preserves perms/ACLs)
    find "$d" -mindepth 1 -exec rm -rf -- {} +
  else
    mkdir -p -- "$d"
  fi
}

# Note that farmer_average.gms has the "base-case"
# Note that using "x(" is based on looking at _fromGAMS/scen0_files/dict.txt
empty_or_create_dir $ODIR
echo "Create the files"
python farmer_writer.py --gms-file farmer_average.gms --nonant-prefix "x(" --output-directory=$ODIR

echo "Use the files (just an interface demo)"
# This is perhaps too clever by about half: the module is the mps_module and its scenario_creator
#  function assumes that mps-files-directory has been set on the command line.
# You can have any generic cylinders commands you like.
# Note that we don't use a lower bound (so only the trivial bound will be there)
###python ../../../mpisppy/generic_cylinders.py --module-name ../../../mpisppy/utils/mps_module --mps-files-directory $ODIR --solver-name ${SOLVER} --max-iterations 2 --default-rho 1 --solution-base-name $SOLBASE
mpiexec -np 2 python -m mpi4py ../../../mpisppy/generic_cylinders.py --module-name ../../../mpisppy/utils/mps_module --mps-files-directory $ODIR --solver-name ${SOLVER} --max-iterations 2 --default-rho 1 --solution-base-name $SOLBASE --xhatshuffle

echo "write the nonant values with GAMS names to nonant_output.csv"
python colmap.py ${ODIR}/scen0_files/dict.txt farmer_solution_output.csv nonant_output.csv --strict

