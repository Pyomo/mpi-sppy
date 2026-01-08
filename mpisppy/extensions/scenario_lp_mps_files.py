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
import os
import json
import mpisppy.extensions.extension
import pyomo.core.base.label as pyomo_label


def lpize(varname):
    # convert varname to the string that will appear in the lp and mps files
    return pyomo_label.cpxlp_label_from_name(varname)


class Scenario_lp_mps_files(mpisppy.extensions.extension.Extension):

    def __init__(self, ph):
        self.ph = ph
        opts = self.ph.options["write_lp_mps_extension_options"]

        self.dirname = opts.get("write_scenario_lp_mps_files_dir")
        if self.dirname is None:
            raise RuntimeError("Scenario_lp_mps_files extension"
                               " cannot be used without the"
                               " write_scenario_lp_mps_files_dir option")

        # Keep cfg so we can use cfg.default_rho when scenario rhos are absent
        self.cfg = opts.get("cfg", None)

        # MPI-safe directory creation: only rank0 creates it, then barrier
        # (dlw hopes that there is only one rank when this runs ...)
        if getattr(self.ph, "cylinder_rank", 0) == 0:
            os.makedirs(self.dirname, exist_ok=False)
            if hasattr(self.ph, "mpicomm"):
                self.ph.mpicomm.Barrier()


    def pre_iter0(self):
        dn = self.dirname  # typing aid
        for k, s in self.ph.local_subproblems.items():
            s.write(os.path.join(dn, f"{k}.lp"), io_options={'symbolic_solver_labels': True})
            s.write(os.path.join(dn, f"{k}.mps"), io_options={'symbolic_solver_labels': True})
            scenData = {"name": s.name, "scenProb": s._mpisppy_probability} 
            scenDict = {"scenarioData": scenData}
            treeData = {"globalNodeCount": len(self.ph.all_nodenames)}
            treeData["nodes"] = dict()
            rhoList = list()
            for nd in s._mpisppy_node_list:
                treeData["nodes"][nd.name] = {"serialNumber": self.ph.all_nodenames.index(nd.name),
                                              "condProb": nd.cond_prob}
                treeData["nodes"][nd.name].update({"nonAnts": [lpize(var.name) for var in nd.nonant_vardata_list]})
                have_rho = hasattr(s, "_mpisppy_model") and hasattr(s._mpisppy_model, "rho")

                # If no rhos exist on the scenarios, require cfg.default_rho
                if (not have_rho) and (self.cfg is None or self.cfg.default_rho is None):
                    raise RuntimeError(
                        "Scenario_lp_mps_files: no rho values found on scenarios and "
                        "--default-rho was not provided. Please specify --default-rho."
                    )

                default_rho = None if have_rho else self.cfg.default_rho

                for i, var in enumerate(nd.nonant_vardata_list):
                    if have_rho:
                        rho_val = s._mpisppy_model.rho[(nd.name, i)]._value
                    else:
                        rho_val = default_rho
                        rhoList.append((lpize(var.name), rho_val))


            scenDict["treeData"] = treeData
            with open(os.path.join(dn, f"{k}_nonants.json"), "w") as jfile:
                json.dump(scenDict, jfile, indent=2)
            # Note that for two stage problems, all rho files will be the same
            with open(os.path.join(dn, f"{k}_rho.csv"), "w") as rfile:
                rfile.write("varname,rho\n")
                for name, rho in rhoList:
                    rfile.write(f"{name},{rho}\n")
                                        
    def post_iter0(self):
        return




