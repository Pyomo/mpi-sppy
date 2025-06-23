###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import pyomo.environ as pyo

from mpisppy.spbase import _put_var_vals_in_component_map_dict 

def test_component_map_usage():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1,2], initialize=2)
    m.y = pyo.Var(["a", "b"], initialize=5)
    m.z = pyo.Var(initialize=42)

    cmap = pyo.ComponentMap()

    _put_var_vals_in_component_map_dict(cmap._dict, m.component_data_objects(pyo.Var))

    assert cmap[m.x[1]] == 2
    assert cmap[m.x[2]] == 2
    assert cmap[m.y["a"]] == 5
    assert cmap[m.y["b"]] == 5
    assert cmap[m.z] == 42
