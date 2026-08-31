###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import time
import pyomo.environ as pyo
import mpisppy.extensions.extension
from mpisppy.utils.sputils import is_persistent
from mpisppy import global_toc

class IntegerRelaxThenEnforce(mpisppy.extensions.extension.Extension):
    """ Class for relaxing integer variables, running PH, and then
        enforcing the integality constraints after some condition.
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.integer_relaxer = pyo.TransformationFactory('core.relax_integer_vars')
        options = opt.options.get("integer_relax_then_enforce_options", {})
        # fraction of iterations or time to spend in relaxed mode
        self.ratio = options.get("ratio", 0.5)


    def pre_iter0(self):
        global_toc(f"{self.__class__.__name__}: relaxing integrality constraints", self.opt.cylinder_rank == 0)
        for s in self.opt.local_scenarios.values():
            self.integer_relaxer.apply_to(s) 
        self._integers_relaxed = True

    def _unrelax_integers(self):
        for sub in self.opt.local_scenarios.values():
            subproblem_solver = sub._solver_plugin
            vlist = None
            if is_persistent(subproblem_solver):
                vlist = list(v for v,d in sub._relaxed_integer_vars[None].values())
            self.integer_relaxer.apply_to(sub, options={"undo":True})
            if is_persistent(subproblem_solver):
                for v in vlist:
                    subproblem_solver.update_var(v)
        self._integers_relaxed = False

    def miditer(self):
        # Each branch below returns after unrelaxing: the conditions are
        # independent and more than one can be true in the same pass -- a run
        # that is past its time fraction and its iteration fraction at once --
        # and the second undo raises, because Pyomo's first one deletes
        # _relaxed_integer_vars.
        if not self._integers_relaxed:
            return
        # time is running out
        #
        # Each rank has its own clock, so without allreduce_or the ranks stop
        # relaxing at different iterations: some would solve MIPs while the
        # others were still solving LPs, and Compute_Xbar would average the
        # two. The other conditions below read _PHIter and conv, which are the
        # same on every rank.
        time_limit = self.opt.options["time_limit"]
        out_of_time = time_limit not in (None, float("inf")) and self.opt.allreduce_or(
            (time.perf_counter() - self.opt.start_time) > (time_limit * self.ratio))
        if out_of_time:
            global_toc(f"{self.__class__.__name__}: enforcing integrality constraints, ran so far for more than {self.opt.options['time_limit']*self.ratio} seconds", self.opt.cylinder_rank == 0)
            self._unrelax_integers()
            return
        # iterations are running out. Both sides are measured from where this
        # run started, because _PHIter counts the study: on a resume it is
        # already past any fraction of this run's iteration budget, and
        # comparing the two directly would enforce integrality immediately.
        start = getattr(self.opt, "_resume_iteration", 0)
        stop = getattr(self.opt, "_stop_iteration", None)
        if stop is None:
            stop = start + int(self.opt.options["PHIterLimit"])
        if (self.opt._PHIter - start) > (stop - start) * self.ratio:
            global_toc(f"{self.__class__.__name__}: enforcing integrality constraints, ran so far for {self.opt._PHIter - 1} iterations", self.opt.cylinder_rank == 0)
            self._unrelax_integers()
            return
        # nearly converged
        if self.opt.conv < (self.opt.options["convthresh"] * 1.1):
            global_toc(f"{self.__class__.__name__}: Enforcing integrality constraints, PH is nearly converged", self.opt.cylinder_rank == 0)
            self._unrelax_integers()
