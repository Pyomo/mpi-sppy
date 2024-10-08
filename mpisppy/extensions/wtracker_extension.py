###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################


import mpisppy.extensions.extension
import mpisppy.utils.wtracker as wtracker


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

    def pre_iter0(self):
        pass

    def post_iter0(self):
        pass
        
    def miditer(self):
        pass

    def enditer(self):
        self.wtracker.grab_local_Ws()

    def post_everything(self):
        reportlen = self.options.get("reportlen")
        stdevthresh = self.options.get("stdevthresh")
        file_prefix = self.options.get("file_prefix")
        self.wtracker.report_by_moving_stats(self.wlen,
                                             reportlen=reportlen,
                                             stdevthresh=stdevthresh,
                                             file_prefix=file_prefix)

        
