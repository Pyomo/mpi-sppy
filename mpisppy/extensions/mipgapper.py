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

from mpisppy import global_toc
import mpisppy.extensions.extension

class Gapper(mpisppy.extensions.extension.Extension):

    def __init__(self, ph):
        self.ph = ph
        self.cylinder_rank = self.ph.cylinder_rank
        self.gapperoptions = self.ph.options["gapperoptions"] # required
        self.mipgapdict = self.gapperoptions.get("mipgapdict", None)
        self.starting_mipgap = self.gapperoptions.get("starting_mipgap", None)
        self.mipgap_ratio = self.gapperoptions.get("mipgap_ratio", None)
        self.verbose = self.ph.options["verbose"] \
                       or self.gapperoptions.get("verbose", True)
        self.verbose = True
        self._check_options()

    def _check_options(self):
        if self.mipgapdict is None and self.starting_mipgap is None:
            raise RuntimeError(f"{self.ph._get_cylinder_name()}: Need to either set a mipgapdict or a starting_mipgap for Gapper")
        if self.mipgapdict is not None and self.starting_mipgap is not None:
            raise RuntimeError(f"{self.ph._get_cylinder_name()} Gapper: Either use a mipgapdict or automatic mode, not both.")
        # exactly one is not None
        return

    def _vb(self, msg):
        if self.verbose:
            global_toc(f"{self.ph._get_cylinder_name()} {self.__class__.__name__}: {msg}", self.cylinder_rank == 0)

    def set_mipgap(self, mipgap):
        """ set the mipgap
        Args:
            float (mipgap): the gap to set
        """
        oldgap = None
        mipgap = float(mipgap)
        if "mipgap" in self.ph.current_solver_options:
            oldgap = self.ph.current_solver_options["mipgap"]
        if oldgap is None or oldgap != mipgap:
            oldgap_str = f"{oldgap}" if oldgap is None else f"{oldgap*100:.3f}%"
            self._vb(f"Changing mipgap from {oldgap_str} to {mipgap*100:.3f}%")
        # no harm in unconditionally setting this, and covers iteration 1
        self.ph.current_solver_options["mipgap"] = mipgap
        
    def pre_iter0(self):
        if self.mipgapdict is None:
            # spcomm not yet set in `__init__`, so check this here
            if self.ph.spcomm is None:
                raise RuntimeError("Automatic gapper can only be used with cylinders -- needs both an upper bound and lower bound cylinder")
            self.set_mipgap(self.starting_mipgap)
        elif 0 in self.mipgapdict:
            self.set_mipgap(self.mipgapdict[0])
                                        
    def post_iter0(self):
        return

    def _autoset_mipgap(self):
        self.ph.spcomm.receive_innerbounds()
        self.ph.spcomm.receive_outerbounds()
        abs_gap, rel_gap = self.ph.spcomm.compute_gaps()
        cylinder_gap = rel_gap * self.mipgap_ratio
        # global_toc(f"{self.ph._get_cylinder_name()}: {self.ph.spcomm.BestInnerBound=}, {self.ph.spcomm.BestOuterBound=}", self.ph.cylinder_rank == 0)
        if cylinder_gap < self.starting_mipgap:
            self.set_mipgap(cylinder_gap)
        # current_solver_options changes in iteration 1
        elif self.ph._PHIter == 1:
            self.set_mipgap(self.starting_mipgap)

    def miditer(self):
        if self.mipgapdict is None:
            self._autoset_mipgap()
            return
        PHIter = self.ph._PHIter
        if PHIter in self.mipgapdict:
            self.set_mipgap(self.mipgapdict[PHIter])


    def enditer(self):
        return

    def post_everything(self):
        return
