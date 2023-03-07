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
import pandas as pd
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
            window_stats (dict): (varname, scenname): [mean, stdev]
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


    def report_by_moving_stats(self, wlen, reportlen=None, stdevthresh=None):
        """ Compute window_stats then sort by "badness" to print a report
        ASSUMES grab_local_Ws is called before this
        Args:
            wlen (int): desired window length
            reportlen (int): max rows in each report
            stdevthresh (float): threshold for a good enough std dev
        NOTE:
            For large problems, this will create a lot of garbage for the collector
        """
        print(f"==== Moving Stats W Report at iteration {self.PHB._PHIter}")
        print(f"    {len(self.varnames)} noants\n"
              f"    {len(self.PHB.local_scenario_names)} scenarios")
        total_traces = len(self.varnames) * len(self.PHB.local_scenario_names)
        
        self.compute_moving_stats(wlen)
        Wsdf = pd.DataFrame.from_dict(wstats, orient='index',
                                      columns=["mean", "stdev"])

        stt = stdevthresh if stdevthresh is not None else self.PHB.E1_tolerance
        
        # unscaled
        goodcnt = len(Wsdf[Wsdf["stdev"] <= stt])
        print(f" {goodcnt} of {total_traces} have windowed stdev (unscaled) below {stt}")
        total_stdev = Wsdf["stdev"].sum()
        print(f" sum of stdev={total_stdev}")

        print(f"Sorted by windowed stdev, row limit={reportlen}, window len={wlen}")
        by_stdev = Wsdf.sort_values(by="stdev", ascending=False)
        print(by_stdev[0:reportlen])

        # scaled
        ####scaleddf = Wsdf.assign(CV=lambda x: x.stdev/x.mean if x.mean > 0 else np.nan)
        scaleddf = np.where(Wsdf["main"] > 0 V=lambda x: x.stdev/x.mean #??? in-place?
        goodcnt = len(scaledf[scaleddf["CV"] <= stt])
        print(f" {goodcnt} of {total_traces} have windowed CV below {stt}")
        total_CV = scaleddf["CV"].sum()
        print(f" sum of CV={tota_CV}")

        print(f"Sorted by windowed CV, row limit={reportlen}, window len={wlen}")
        by_CV = scaleddf.sort_values(by="CV", ascending=False)
        print(by_CV[0:reportlen])
    
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

    wt.report_by_moving_stats(wlen, reportlen=6, stdevthresh=1e-16)