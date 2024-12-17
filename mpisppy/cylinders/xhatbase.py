###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import abc
import mpisppy.cylinders.spoke as spoke
from mpisppy.utils.xhat_eval import Xhat_Eval

class XhatInnerBoundBase(spoke.InnerBoundNonantSpoke):

    @abc.abstractmethod
    def xhat_extension(self):
        raise NotImplementedError
    

    def xhat_prep(self):
        if "bundles_per_rank" in self.opt.options\
           and self.opt.options["bundles_per_rank"] != 0:
            raise RuntimeError("xhat spokes cannot have bundles (yet)")

        ## for later
        self.verbose = self.opt.options["verbose"] # typing aid  

        if not isinstance(self.opt, Xhat_Eval):
            raise RuntimeError(f"{self.__class__.__name__} must be used with Xhat_Eval.")
            
        xhatter = self.xhat_extension()

        ### begin iter0 stuff
        xhatter.pre_iter0()
        if self.opt.extensions is not None:
            self.opt.extobject.pre_iter0()  # for an extension
        self.opt._save_original_nonants()

        self.opt._lazy_create_solvers()  # no iter0 loop, but we need the solvers

        self.opt._update_E1()
        if abs(1 - self.opt.E1) > self.opt.E1_tolerance:
            raise ValueError(f"Total probability of scenarios was {self.opt.E1} "+\
                                 f"(E1_tolerance is {self.opt.E1_tolerance})")
        ### end iter0 stuff (but note: no need for iter 0 solves in an xhatter)

        xhatter.post_iter0()
        if self.opt.extensions is not None:
            self.opt.extobject.post_iter0()  # for an extension
 
        self.opt._save_nonants() # make the cache

        return xhatter
