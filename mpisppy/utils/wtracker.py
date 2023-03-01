# This software is distributed under the 3-clause BSD License.
# Started by DLW, March 2023

"""
- A utility to track W and compute interesting statistics and/or log the w values
- We are going to take a comm as input so we can do a gather,
  so be careful if you use this with APH (you might want to call it from a lagrangian spoke)
"""

import mpisppy.opt.ph
import pyomo.environ as pyo

class WTracker():
    """

    Notes:
    - A utility to track W and compute interesting statistics and/or log the w values
    - We are going to take a comm as input so we can do a gather,
    so be careful if you use this with APH (you might want to call it from a lagrangian spoke)
    """

    
    def __init__(self, PHB, comm = None):
        #### assert comm is not None
        self.comm = comm
        self.varnames = list()  # bound by index to W vals
        self.Ws = dict()  #  [iteration][sname](list of W vals)
        self.PHB = PHB

    def track_Ws(self):
        """ Get the W values from the PHB that we are following
            NOTE: we assume this is called only once per iteration
        """
        
        for (sname, scenario) in self.PHB.local_scenarios.items():
            scenario_Ws = {var.name: pyo.value(scenario._mpisppy_model.W[node.name, ix])
                           for node in scenario._mpisppy_node_list
                           for (ix, var) in enumerate(node.nonant_vardata_list)}

        for node in scenario._mpisppy_node_list:
            for (ix, var) in enumerate(node.nonant_vardata_list):
                for (vname, val) in scenario_Ws.items():
                    print(vname, val)
        

if __name__ == "__main__":
    # for ad hoc developer testing
    solver_name = "cplex"
    
    import mpisppy.tests.examples.farmer as farmer
    options = {}
    options["asynchronousPH"] = False
    options["solver_name"] = solver_name
    options["PHIterLimit"] = 10
    options["defaultPHrho"] = 1
    options["convthresh"] = 0.001
    options["subsolvedirectives"] = None
    options["verbose"] = False
    options["display_timing"] = False
    options["display_progress"] = False
    if "cplex" in solver_name:
        options["iter0_solver_options"] = {"mip_tolerances_mipgap": 0.001}
        options["iterk_solver_options"] = {"mip_tolerances_mipgap": 0.00001}
    else:
        options["iter0_solver_options"] = {"mipgap": 0.001}
        options["iterk_solver_options"] = {"mipgap": 0.00001}

    
    options["PHIterLimit"] = 1
    ph = mpisppy.opt.ph.PH(
        options,
        farmer.scenario_names_creator(3),
        farmer.scenario_creator,
        farmer.scenario_denouement,
        scenario_creator_kwargs=farmer.kw_creator(options),
    )

    conv, obj, trivial_bound = ph.ph_main()
    wt = WTracker(ph)
    wt.track_Ws()
