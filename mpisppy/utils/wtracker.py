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
import mpisppy.MPI as MPI

class WTracker():
    """
    Args:
        PHB (PHBase): the object that has all the scenarios and _PHIter
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

        scenario_Ws = {sname: [w._value for w in scenario._mpisppy_model.W.values()]
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
                                  OR returns a warning string
        NOTE: we sort of treat iterations as one-based
        """
        cI = self.PHB._PHIter
        li = cI - offsetback
        fi = max(1, li - wlen)
        if li - fi < wlen:
            return (f"WARNING: Not enough iterations ({cI}) for window len {wlen} and"
                   f" offsetback {offsetback}\n")
        else:
            window_stats = dict()
            for idx, varname in enumerate(self.varnames):
                for sname in self.PHB.local_scenario_names:
                    wlist = [self.local_Ws[i][sname][idx] for i in range(fi, li+1)]
                    window_stats[(varname, sname)] = (np.mean(wlist), np.std(wlist))
            return window_stats


    def report_by_moving_stats(self, wlen, reportlen=None, stdevthresh=None, file_prefix=''):
        """ Compute window_stats then sort by "badness" to write to three files
        ASSUMES grab_local_Ws is called before this
        Args:
            wlen (int): desired window length
            reportlen (int): max rows in each report
            stdevthresh (float): threshold for a good enough std dev
        NOTE:
            For large problems, this will create a lot of garbage for the collector
        """
        fname = f"{file_prefix}_summary_iter{self.PHB._PHIter}_rank{self.PHB.global_rank}.txt"
        stname = f"{file_prefix}_stdev_iter{self.PHB._PHIter}_rank{self.PHB.global_rank}.csv"
        cvname = f"{file_prefix}_cv_iter{self.PHB._PHIter}_rank{self.PHB.global_rank}.csv"
        if self.PHB.cylinder_rank == 0:
            print(f"Writing (a) W tracker report(s) to files with names like {fname}, {stname}, and {cvname}")
        with open(fname, "w") as fil:
            fil.write(f"Moving Stats W Report at iteration {self.PHB._PHIter}\n")
            fil.write(f"    {len(self.varnames)} nonants\n"
                      f"    {len(self.PHB.local_scenario_names)} scenarios\n")
            
        total_traces = len(self.varnames) * len(self.PHB.local_scenario_names)
        
        wstats = self.compute_moving_stats(wlen)
        if not isinstance(wstats, str):
            Wsdf = pd.DataFrame.from_dict(wstats, orient='index',
                                          columns=["mean", "stdev"])

            stt = stdevthresh if stdevthresh is not None else self.PHB.E1_tolerance

            # unscaled
            goodcnt = len(Wsdf[Wsdf["stdev"] <= stt])
            total_stdev = Wsdf["stdev"].sum()

            with open(fname, "a") as fil:
                fil.write(f" {goodcnt} of {total_traces} have windowed stdev (unscaled) below {stt}\n")
                fil.write(f" sum of stdev={total_stdev}\n")

                fil.write(f"Sorted by windowed stdev, row limit={reportlen}, window len={wlen} in {stname}\n")
            by_stdev = Wsdf.sort_values(by="stdev", ascending=False)
            by_stdev[0:reportlen].to_csv(path_or_buf=stname, header=True, index=True, index_label=None, mode='w')

            # scaled
            Wsdf["absCV"] = np.where(Wsdf["mean"] != 0, Wsdf["stdev"]/abs(Wsdf["mean"]), np.nan)
            goodcnt = len(Wsdf[Wsdf["absCV"] <= stt])
            mean_absCV = Wsdf["absCV"].mean()
            stdev_absCV = Wsdf["absCV"].std()
            zeroWcnt = len(Wsdf[Wsdf["mean"] == 0])
            with open(fname, "a") as fil:
                fil.write(f"  {goodcnt} of ({total_traces} less meanzero {zeroWcnt}) have windowed absCV below {stt}\n")
                fil.write(f"  mean absCV={mean_absCV}, stdev={stdev_absCV}\n")
                fil.write(f"  Sorted by windowed abs CV, row limit={reportlen}, window len={wlen} in {cvname}\n")
            by_absCV = Wsdf.sort_values(by="absCV", ascending=False)
            by_absCV[0:reportlen].to_csv(path_or_buf=cvname, header=True, index=True, index_label=None, mode='w')
        else:  # not enough data
            with open(fname, "a") as fil:
                fil.write(wstats)   # warning string


    def W_diff(self):
        """ Compute the norm of the difference between to consecutive Ws / num_scenarios.

        Returns:
           global_diff (float): difference between to consecutive Ws

        """
        cI = self.PHB._PHIter
        self.grab_local_Ws()
        self.local_Ws[-1] = self.local_Ws[0]
        global_diff = np.zeros(1)
        local_diff = np.zeros(1)
        varcount = 0
        local_diff[0] = 0
        for (sname, scenario) in self.PHB.local_scenarios.items():
            local_wdiffs = [w - w1
                            for w, w1 in zip(self.local_Ws[cI][sname], self.local_Ws[cI-1][sname])]
            for wdiff in local_wdiffs:
                local_diff[0] += abs(wdiff)
                varcount += 1
        local_diff[0] /= varcount
        self.PHB.comms["ROOT"].Allreduce(local_diff, global_diff, op=MPI.SUM)

        return global_diff[0] / self.PHB.n_proc


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
