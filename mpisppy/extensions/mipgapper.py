###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
""" Code for a mipgap schedule. This can be used
    as the only extension, but it could also be called from a "multi"
    extension.
"""

import mpisppy.extensions.extension

class Gapper(mpisppy.extensions.extension.Extension):

    def __init__(self, ph):
        self.ph = ph
        self.cylinder_rank = self.ph.cylinder_rank
        self.gapperoptions = self.ph.options["gapperoptions"] # required
        self.mipgapdict = self.gapperoptions["mipgapdict"]
        self.verbose = self.ph.options["verbose"] \
                       or self.gapperoptions["verbose"]
                       
    def _vb(self, str):
        if self.verbose and self.cylinder_rank == 0:
            print ("(rank0) mipgapper:" + str)

    def set_mipgap(self, mipgap):
        """ set the mipgap
        Args:
            float (mipgap): the gap to set
        """
        oldgap = None
        if "mipgap" in self.ph.current_solver_options:
            oldgap = self.ph.current_solver_options["mipgap"]
        self._vb("Changing mipgap from "+str(oldgap)+" to "+str(mipgap))
        self.ph.current_solver_options["mipgap"] = float(mipgap)
        
    def pre_iter0(self):
        if self.mipgapdict is None:
            return
        if 0 in self.mipgapdict:
            self.set_mipgap(self.mipgapdict[0])
                                        
    def post_iter0(self):
        return

    def miditer(self):
        if self.mipgapdict is None:
            return
        PHIter = self.ph._PHIter
        if PHIter in self.mipgapdict:
            self.set_mipgap(self.mipgapdict[PHIter])


    def enditer(self):
        return

    def post_everything(self):
        return
