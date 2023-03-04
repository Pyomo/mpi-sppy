# This software is distributed under the 3-clause BSD License.
# Started by DLW, March 2023

"""
- A utility to track W and compute interesting statistics and/or log the w values
- Uses time and memory so it is intended for diagnostics, not everyday use.
- If we do a gather, we are going to take a comm as input,
  so you can be careful if you use this with APH (you might want to call it from a lagrangian spoke)
- TBD as of March 2023: track oscillations, which are important for MIPs (could compare two moving_stddevs)
"""

import numpy as np
import mpisppy.opt.ph
import pyomo.environ as pyo

class WTracker():
    """
    Args:
        PHB (PHBase): the object that has all the scenarios and _PHIter
        comm (MPI comm): to maybe do a gather
    Notes:
    - A utility to track W and compute interesting statistics and/or log the w values
    - We are going to take a comm as input so we can do a gather if we want to,
    so be careful if you use this with APH (you might want to call it from a lagrangian spoke)
    
    """

    
    def __init__(self, PHB):
        self.local_Ws = dict()  #  [iteration][sname](list of W vals)
        self.PHB = PHB
        # get the nonant variable names, bound by index to W vals
        arbitrary_scen = PHB.local_scenarios[list(PHB.local_scenarios.keys())[0]]
        self.varnames = [var.name for node in arbitrary_scen._mpisppy_node_list
                         for (ix, var) in enumerate(node.nonant_vardata_list)]

        
    def grab_local_Ws(self):
        """ Get the W values from the PHB that we are following
            NOTE: we assume this is called only once per iteration
        """
        
        scenario_Ws = {sname: [pyo.value(scenario._mpisppy_model.W[node.name, ix])
                               for node in scenario._mpisppy_node_list
                               for (ix, var) in enumerate(node.nonant_vardata_list)]
                       for (sname, scenario) in self.PHB.local_scenarios.items()}

        self.local_Ws[self.PHB._PHIter] = scenario_Ws


    def compute_moving_stats(self, wlen, offsetback=0):
        """ Use self.local_Ws to compute moving mean and stdev
        ASSUMES grab_local_Ws is called before this
        Args:
            wlen (int): desired window length
            offsetback (int): how far back from the most recent observation to start
        Returns:
            window_stats (dict): xxxxx
        NOTE: we sort of treat iterations as one-based
        """
        cI = self.PHB._PHIter
        li = cI - offsetback
        fi = max(1, li - wlen)
        if li - fi < wlen:
            raise RuntimeError(f"Not enough iterations ({cI}) for window len {wlen} and"
                               f" offsetback {offsetback}")
        window_stats = dict()
        for idx, varname in enumerate(self.varnames):
            for sname in self.PHB.local_scenario_names:
                wlist = [self.local_Ws[i][sname][idx] for i in range(fi, li+1)]
                window_stats[(varname, sname)] = [np.mean(wlist), np.std(wlist)]
        return window_stats


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
    wt.grab_local_Ws()

    # wild hack to look for stack traces...
    wlen = 5
    for i in range(wlen):
        ph._PHIter += 1
        wt.grab_local_Ws()
    wstats = wt.compute_moving_stats(wlen)
    for idx, v in wstats.items():
        print(idx, v)
