###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# dlw, Feb 2018. Code for closes scenario to xbar
# TODO Apr. 2020 Eliminate old references to companiondriver
import numpy as np
import mpisppy.MPI as mpi
import mpisppy.extensions.xhatbase
import pyomo.environ as pyo

class XhatClosest(mpisppy.extensions.xhatbase.XhatBase):
    """
    Args:
        rank (int): mpi process rank of currently running process
    """
    def __init__(self, ph):
        super().__init__(ph)
        self.options = ph.options["xhat_closest_options"]
        self.solver_options = self.options["xhat_solver_options"]
        self.keep_solution = True
        if ('keep_solution' in self.options) and (not self.options['keep_solution']):
            self.keep_solution = False

    def xhat_closest_to_xbar(self, verbose=False, restore_nonants=True):
        """ Get a truncated z score and look for the closest overall.

        Returns:
            obj (float or None): objective value, none if infeasible.
            snamedict (str): closest scenarios
        """
        def _vb(msg):
            if verbose and self.cylinder_rank == 0:
                print ("(rank0) xhat_looper: " + msg)

        localmindist = np.zeros(1, dtype='d')
        globalmindist = np.zeros(1, dtype='d')
        localwinnername = None
        for k, s in self.opt.local_scenarios.items():
            dist = 0
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                diff = pyo.value(xvar) - pyo.value(s._mpisppy_model.xbars[ndn_i])
                variance = pyo.value(s._mpisppy_model.xsqbars[ndn_i]) \
                  - pyo.value(s._mpisppy_model.xbars[ndn_i])*pyo.value(s._mpisppy_model.xbars[ndn_i])
                if variance > 0:
                    stdev = np.sqrt(variance)
                    dist += min(3, abs(diff)/stdev)
            if localwinnername is None:
                localmindist[0] = dist
                localwinnername = k
            elif dist < localmindist[0]:
                localmindist[0] = dist
                localwinnername = k

        self.comms["ROOT"].Allreduce([localmindist, mpi.DOUBLE],
                                     [globalmindist, mpi.DOUBLE],
                                     op=mpi.MIN)
        # ties are possible, so break the tie
        localwinrank = np.zeros(1, dtype='d')  # could use a python variable.
        globalwinrank = np.zeros(1, dtype='d')
        if globalmindist[0] < localmindist[0]:
            localwinrank[0] = -1  # we lost
        else:
            localwinrank[0] = self.cylinder_rank
        self.comms["ROOT"].Allreduce([localwinrank, mpi.DOUBLE],
                                     [globalwinrank, mpi.DOUBLE],
                                     op=mpi.MAX)

        # We only used the rank to break a possible tie.
        if self.cylinder_rank == int(globalwinrank[0]):
            globalwinnername = localwinnername
        else:
            globalwinnername = None

        sroot = globalwinrank[0]

        sname = self.comms["ROOT"].bcast(globalwinnername, root=sroot)
        _vb("Trying scenario "+sname)
        _vb("   Solver options="+str(self.solver_options))
        # xxx TBD mult-stage
        snamedict = {"ROOT": sname}
        obj = self._try_one(snamedict,
                            solver_options=self.solver_options,
                            verbose=False,
                            restore_nonants=restore_nonants)
        if obj is None:
            _vb("    Infeasible")
        else:
            _vb("    Feasible, returning " + str(obj))
            
        return obj, snamedict
        
    def pre_iter0(self):
        if self.opt.multistage:
            raise RuntimeError("xhatclosest not done for multi-stage")

    def post_iter0(self):
        # a little bit silly
        self.comms = self.opt.comms
        
    def miditer(self):
        pass

    def enditer(self):
        pass

    def post_everything(self):
        # if we're keeping the solution, we *do not* restore the nonants
        restore_nonants = not self.keep_solution
        self.opt.disable_W_and_prox()
        obj, srcsname = self.xhat_closest_to_xbar(verbose=self.verbose, restore_nonants=restore_nonants)
        self.opt.reenable_W_and_prox()
        self.xhat_common_post_everything("closest to xbar", obj, srcsname, restore_nonants)
        self._final_xhat_closest_obj = obj
