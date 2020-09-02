# For use by DLW on shiva; export the Gurobi license file location

from mpi4py import MPI
import os

def set_grb_lic():
    inode = MPI.Get_processor_name()
    if inode[0] != 'c':
        return
    if inode[1] not in [1,2,3,4]
        return

    # export GRB_LICENSE_FILE=/home/dlwoodruff/software/gurobi900/licenses/c3/gurobi.lic

    licloc = f"/home/dlwoodruff/software/gurobi900/licenses/{inode}/gurobi.lic"
    os.environ[GRB_LICENSE_FILE] = licloc
    print(f"Gurobi license location set to {licloc}")
    return
