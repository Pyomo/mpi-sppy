###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
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
        if hasattr(self.PHB, '_PHIter'):
            self.ph_iter = self.PHB._PHIter
        elif hasattr(self.PHB, 'A_iter'):
            self.ph_iter = self.PHB.A_iter
        else:
            raise RuntimeError("WTracker created before its PHBase object has _PHIter")

        
    def grab_local_Ws(self):
        """ Get the W values from the PHB that we are following
            NOTE: we assume this is called only once per iteration
        """
        if hasattr(self.PHB, '_PHIter'): #update the iter value
            self.ph_iter = self.PHB._PHIter
        elif hasattr(self.PHB, 'A_iter'):
            self.ph_iter = self.PHB.A_iter
        scenario_Ws = {sname: [w._value for w in scenario._mpisppy_model.W.values()]
                       for (sname, scenario) in self.PHB.local_scenarios.items()}
        self.local_Ws[self.ph_iter] = scenario_Ws

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
        cI = self.ph_iter
        li = cI - offsetback
        fi = max(1, li - wlen)
        if li - fi < wlen:
            return (f"WARNING: Not enough iterations ({cI}) for window len {wlen} and"
                   f" offsetback {offsetback}\n")
        else:
            window_stats = dict()
            wlist = dict()
            for idx, varname in enumerate(self.varnames):
                for sname in self.PHB.local_scenario_names:
                    wlist[(varname, sname)] = [self.local_Ws[i][sname][idx] for i in range(fi, li+1)]
                    window_stats[(varname, sname)] = (np.mean(wlist[(varname, sname)]), np.std(wlist[(varname, sname)]))
            return wlist, window_stats


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
        fname = f"{file_prefix}_summary_iter{self.ph_iter}_rank{self.PHB.global_rank}.txt"
        stname = f"{file_prefix}_stdev_iter{self.ph_iter}_rank{self.PHB.global_rank}.csv"
        cvname = f"{file_prefix}_cv_iter{self.ph_iter}_rank{self.PHB.global_rank}.csv"

        if self.PHB.cylinder_rank == 0:
            print(f"Writing (a) W tracker report(s) to files with names like {fname}, {stname}, and {cvname}")
        with open(fname, "w") as fil:
            fil.write(f"Moving Stats W Report at iteration {self.ph_iter}\n")
            fil.write(f"    {len(self.varnames)} nonants\n"
                      f"    {len(self.PHB.local_scenario_names)} scenarios\n")
            
        total_traces = len(self.varnames) * len(self.PHB.local_scenario_names)
        wstats = self.compute_moving_stats(wlen)[1]
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
        cI = self.ph_iter
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

    def check_cross_zero(self, wlen, offsetback=0):
        """NOTE: we assume this is called after grab_local_Ws
        """
        cI = self.ph_iter
        li = cI - offsetback
        fi = max(1, li - wlen)
        if li - fi < wlen:
            return (f"WTRACKER WARNING: Not enough iterations ({cI}) for window len {wlen} and"
                   f" offsetback {offsetback}\n")
        else:
            print(f"{np.shape(self.local_Ws)}")
            for i in range(fi+1, li+1):
                for sname, _ in self.PHB.local_scenarios.items():
                    sgn_curr_iter = np.sign(np.array(self.local_Ws[fi][sname]))
                    sgn_last_iter = np.sign(np.array(self.local_Ws[fi-1][sname]))
                    if np.all(sgn_curr_iter!=sgn_last_iter):
                        return (f"WTRACKER BAD: Ws crossed zero, sensed at iter {i}")
            return (f"WTRACKER GOOD: Ws did not cross zero over the past {wlen} iters")

    def check_w_stdev(self, wlen, stdevthresh, offsetback=0):
        _, window_stats = self.compute_moving_stats(wlen)
        cI = self.ph_iter
        li = cI - offsetback
        fi = max(1, li - wlen)
        if li - fi < wlen:
            return (f"WTRACKER WARNING: Not enough iterations ({cI}) for window len {wlen} and"
                    f" offsetback {offsetback}\n")
        else:
            for idx, varname in enumerate(self.varnames):
                for sname in self.PHB.local_scenario_names:
                    if np.abs(window_stats[(varname, sname)][1]) >= np.abs(stdevthresh * window_stats[(varname, sname)][0]): #stdev scaled by mean
                        return (f"WTRACKER BAD: at least one scaled stdev is bigger than {stdevthresh}")
            return (f"WTRACKER GOOD: all scaled stdev are smaller than {stdevthresh}")
    
    


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
    _, wstats = wt.compute_moving_stats(wlen)

    wt.report_by_moving_stats(wlen, reportlen=6, stdevthresh=1e-16)
