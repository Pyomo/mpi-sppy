# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
""" Code for a mipgap schedule. This can be used
    as the only extension, but it could also be called from a "master"
    extension.
"""

import pyomo.environ as pyo
import mpisppy.extensions.extension

class Gapper(mpisppy.extensions.extension.Extension):

    def __init__(self, ph):
        self.ph = ph
        self.rank = self.ph.rank
        self.rank0 = self.ph.rank0
        self.gapperoptions = self.ph.PHoptions["gapperoptions"] # required
        self.mipgapdict = self.gapperoptions["mipgapdict"]
        self.verbose = self.ph.PHoptions["verbose"] \
                       or self.gapperoptions["verbose"]
                       
    def _vb(self, str):
        if self.verbose and self.rank == 0:
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
