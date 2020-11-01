# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# For use by DLW on shiva; export the Gurobi license file location

from mpi4py import MPI
import os

def set_grb_lic():
    inode = MPI.Get_processor_name()
    assert inode[0] == 'c', f"{inode} is not a valid shiva node name"
    assert inode[1] in ["1","2","3","4"], f"{inode} is not a valid shiva node name"

    # export GRB_LICENSE_FILE=/home/dlwoodruff/software/gurobi900/licenses/c3/gurobi.lic

    licloc = f"/home/dlwoodruff/software/gurobi900/licenses/{inode}/gurobi.lic"
    os.environ["GRB_LICENSE_FILE"] = licloc
    print(f"Gurobi license location set to {licloc}")
    return
