# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# An extension to compute and output avg, min, max for
# a component (e.g., first stage cost).
# DLW, Feb 2019
# This extension uses PHoptions["avgminmax_name"]

import mpisppy.extensions.xhatbase

class MinMaxAvg(mpisppy.extensions.xhatbase.XhatBase):
    """
    Args:
        ph (PH object): the calling object
        rank (int): mpi process rank of currently running process
    """
    def __init__(self, ph, rank, n_proc):
        super().__init__(ph, rank, n_proc)
        self.compstr = self.ph.PHoptions["avgminmax_name"]

    def pre_iter0(self):
        return

    def post_iter0(self):
        avgv, minv, maxv = self.ph.avg_min_max(self.compstr)
        if (self.rank == 0):
            print ("  ### ", self.compstr,": avg, min, max, max-min", avgv, minv, maxv, maxv-minv)
        
    def miditer(self, PHIter, conv):
        return

    def enditer(self, PHIter):
        avgv, minv, maxv = self.ph.avg_min_max(self.compstr)
        if (self.rank == 0):
            print ("  ### ", self.compstr,": avg, min, max, max-min", avgv, minv, maxv, maxv-minv)

    def post_everything(self, PHIter, conv):
        return


