###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
""" Extension to write an lp file for each scenario and a json file for
the nonant structure for each scenario (yes, for two stage problems this
json file will be the same for all scenarios.)
"""

import json
import mpisppy.extensions.extension
import pyomo.core.base.label as pyomo_label


def lpize(varname):
    # convert varname to the string that will appear in the lp file
    # return varname.replace("[", "(").replace("]", ")").replace(",", "_").replace(".","_")
    return pyomo_label.cpxlp_label_from_name(varname)


class Scenario_lpfiles(mpisppy.extensions.extension.Extension):

    def __init__(self, ph):
        self.ph = ph

    def pre_iter0(self):
        for k, s in self.ph.local_subproblems.items():
            s.write(f"{k}.lp", io_options={'symbolic_solver_labels': True})
            nonants_by_node = {nd.name: [lpize(var.name) for var in nd.nonant_vardata_list] for nd in s._mpisppy_node_list}
            with open(f"{k}_nonants.json", "w") as jfile:
                json.dump(nonants_by_node, jfile)
                                        
    def post_iter0(self):
        return




