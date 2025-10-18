#!/bin/bash
# run the example where an AMPLpy script writes scenarios to be read by mpi-sppy

set -e

ODIR="_fromAMPL"
SOLVER="gurobi"


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

empty_or_create_dir $ODIR
echo "Create the files"
python farmer_writer.py --output-directory=$ODIR

echo "Use the files (with only one cylinder... so this is just a demo"
# This is perhaps too clever by about half: the module is the mps_module and its scenario_creator
#  function assumes that mps-files-directory has been set on the command line.
# You can have any generic cylinders commands you like.
python -m mpi4py ../../../mpisppy/generic_cylinders.py --module-name ../../../mpisppy/utils/mps_module --mps-files-directory $ODIR --solver-name ${SOLVER} --max-iterations 2 --default-rho 1 
