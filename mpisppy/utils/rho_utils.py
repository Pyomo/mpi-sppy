# This software is distributed under the 3-clause BSD License.

import pandas as pd

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
    Args:
        s (ConcreteModel): scenario whence the id values come
        filename (str): name of the csv file to read (fullname, rho)
    Returns:
        retlist (list of (id, rho) tuples); list suitable for rho_setter
   """
    rhodf = pd.read_csv(filename)
    retlist = list()
    for idx, row in rhodf.iterrows():
        fullname = row["fullname"]
        vo = s.find_component(fullname)
        if vo is not None:
            retlist.append((id(vo), row["rho"]))
        else:
            raise RuntimeError(f"rho values from {filename} found Var {fullname} "
                               f"that is not found in the scenario given (name={s._name})")
    return retlist    
