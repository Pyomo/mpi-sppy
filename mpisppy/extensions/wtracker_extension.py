###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################


import mpisppy.extensions.extension
import mpisppy.utils.w_utils.wtracker as wtracker
from mpisppy import global_toc


class Wtracker_extension(mpisppy.extensions.extension.Extension):
    """
        wrap the wtracker code as an extension
        
        Args:
            opt (PHBase (inherets from SPOpt) object): gives the problem that we bound

        Attributes:
          scenario_name_to_rank (dict of dict): nodes (i.e. comms) scen names
                keys are comms (i.e., tree nodes); values are dicts with keys
                that are scenario names and values that are ranks
    """

    def __init__(self, opt, comm=None):
        super().__init__(opt)
        self.cylinder_rank = self.opt.cylinder_rank
        self.verbose = self.opt.options["verbose"]
        self.wtracker = wtracker.WTracker(opt)
        self.options = opt.options["wtracker_options"]
        # TBD: more graceful death if options are bad
        self.wlen = self.options["wlen"]
        #: The wlen the checkpoint this run resumed from was written with,
        #: or None on a run that did not resume.
        self._carried_wlen = None

    def pre_iter0(self):
        pass

    def post_iter0(self):
        pass
        
    def miditer(self):
        pass

    def enditer(self):
        self.wtracker.grab_local_Ws()

    def checkpoint_state(self):
        """The W sets the end-of-run report reads.

        `report_by_moving_stats` reads `local_Ws` at the last `wlen + 1`
        iterations (`compute_moving_stats` goes from `li - wlen` through
        `li`). A resumed run holds only the sets it grabbed itself, so
        without these its report covers the resumed leg alone: a shorter
        window than the one asked for, or none at all, where an
        uninterrupted run of the same length would have reported the full
        one. The report is the whole point of --wtracker, so it is the
        thing that has to survive the stop.

        (A resumed leg shorter than the window used to die with a KeyError
        here. That is fixed in `compute_moving_stats` itself and is no
        longer what this carries state for -- the two are independent, and
        `wtracker.py`'s window still needs to hold for a run that asked for
        --wtracker only on the resumed leg.)

        Carry the last `wlen + 1` sets, not the whole history, which grows by
        scenarios times nonants of floats per iteration.
        """
        local_Ws = self.wtracker.local_Ws
        if not local_Ws:
            return None
        keep = sorted(local_Ws)[-(self.wlen + 1):]
        carried = {k: local_Ws[k] for k in keep}
        self._toc_what_this_costs(carried)
        # wlen rides along so a resume can tell "the study had not run long
        # enough yet" -- which the report already says for itself -- from
        # "the two legs were asked for different windows".
        return {"local_Ws": carried, "ph_iter": self.wtracker.ph_iter,
                "wlen": self.wlen}

    def _toc_what_this_costs(self, carried):
        """Say once what --wtracker adds to every checkpoint write.

        The window is the user's `wlen`, so this scales with an option they
        chose: (wlen+1) sets x the scenarios on this rank x nonants per
        scenario x 8 bytes, on every write. At the default it is a few
        hundred bytes; at a wlen of 1000 on a large instance it is gigabytes
        per rank. Reported rather than capped -- carrying less than was
        asked for would make the resumed report quietly disagree with an
        uninterrupted one -- and reported once, because it does not change.
        """
        if getattr(self, "_size_tocced", False):
            return
        self._size_tocced = True
        # This runs at the *first* write, where the window holds one set and
        # every write after it carries wlen+1. A set is the same size every
        # iteration -- this rank's scenarios by their nonants -- so the
        # window's cost is that one set multiplied out, which is arithmetic
        # rather than an estimate. Counting what happened to be carried at
        # this one write reported 1/(wlen+1) of what the option costs, which
        # is the opposite of the point: at the wlen of 1000 above it turned
        # gigabytes into megabytes.
        sets = max(len(carried), self.wlen + 1)
        per_set = sum(len(w) for w in next(iter(carried.values())).values())
        values = sets * per_set
        global_toc(
            f"Wtracker carries the last {sets} W set(s) (wlen {self.wlen}) "
            f"into every checkpoint: {values} values, about "
            f"{values * 8 / 1024:.1f} KiB per rank per write.",
            self.opt.cylinder_rank == 0)

    def restore_state(self, state):
        carried = state["local_Ws"]
        self.wtracker.local_Ws.update(carried)
        self.wtracker.ph_iter = state["ph_iter"]
        # Kept rather than judged here: whether a window wider than the
        # checkpoint's can be filled depends on how far this leg runs, and
        # this hook runs at the end of Iter0. See
        # _say_why_the_window_is_short.
        self._carried_wlen = state.get("wlen")
        return None

    def post_everything(self):
        reportlen = self.options.get("reportlen")
        stdevthresh = self.options.get("stdevthresh")
        # "" rather than None when the option is unset, matching
        # report_by_moving_stats's own default: unset, the three reports were
        # named after a stringified None ("None_summary_iter5_rank0.txt").
        file_prefix = self.options.get("file_prefix") or ""
        shortfall = self.wtracker.report_by_moving_stats(
            self.wlen,
            reportlen=reportlen,
            stdevthresh=stdevthresh,
            file_prefix=file_prefix)
        self._say_why_the_window_is_short(shortfall)

    def _say_why_the_window_is_short(self, shortfall):
        """Name the earlier leg's narrower window when that is what cost the
        report.

        Only the report knows whether the window was filled. A resumed leg
        that runs long enough fills a wider window on its own, whatever the
        leg before it was asked for, and its report is then exactly the one
        an uninterrupted run of the same length would have written. Asked at
        the restore instead -- which happens at the end of Iter0, before this
        leg has run an iteration -- this warned every such run that its
        report was short while the report it went on to write was complete.

        A window the *same* wlen could not fill stays silent: the report says
        "not enough iterations tracked" for itself, and that case is the
        study not having run long enough rather than the two legs having been
        asked different questions.
        """
        if shortfall is None or self._carried_wlen is None:
            return
        if self._carried_wlen >= self.wlen:
            return
        global_toc(
            f"WARNING: Wtracker_extension: no report for window len "
            f"{self.wlen}. The checkpoint was written with wlen "
            f"{self._carried_wlen}, so it carried that window rather than "
            f"this one, and this leg did not run long enough to fill the "
            f"wider one on its own.",
            self.opt.cylinder_rank == 0)

        
