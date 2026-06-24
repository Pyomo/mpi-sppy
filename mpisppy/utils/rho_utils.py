###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from pyomo.common.dependencies import pandas as pd

def rhos_to_csv(s, filename):
    """ write the rho values to a csv "fullname", rho
    Args:
        s (ConcreteModel): the scenario Pyomo model
        filenaame (str): file to which to write
    """
    with open(filename, "w") as f:
        f.write("fullname,rho\n")
        for ndn_i, rho in s._mpisppy_model.rho.items():
            vdata = s._mpisppy_data.nonant_indices[ndn_i]
            fullname = vdata.name
            f.write(f'"{fullname}",{rho._value}\n')

            
def rho_list_from_csv(s, filename):
    """ read rho values from a file and return a list suitable for rho_setter

    The csv may use a "fullname" or "varname" column for the variable name.
    Parentheses in names are normalized to underscores so names written in the
    lp/mps label form (e.g. ``X(1)``) match the Pyomo components built by the
    reader (``X_1_``); for ordinary Pyomo names this normalization is a no-op.

    Args:
        s (ConcreteModel): scenario whence the id values come
        filename (str): name of the csv file to read ((fullname|varname), rho)
    Returns:
        retlist (list of (id, rho) tuples); list suitable for rho_setter
   """
    rhodf = pd.read_csv(filename)
    namecol = "varname" if "varname" in rhodf.columns else "fullname"
    retlist = list()
    for idx, row in rhodf.iterrows():
        fullname = str(row[namecol]).replace('(', '_').replace(')', '_')
        vo = s.find_component(fullname)
        if vo is not None:
            retlist.append((id(vo), row["rho"]))
        else:
            raise RuntimeError(f"rho values from {filename} found Var {fullname} "
                               f"that is not found in the scenario given (name={s._name})")
    return retlist
