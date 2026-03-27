###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# An extension to output diagnostic information at each iteration
# to a file for each scenario.
# DLW, March 2019
# This extension uses options["diagnostics_outdir"]
# This could be used as a starting point for writing your own diangostics;
#   for "canned" diagnostics, use the baseparser PH option with-display-convergence-detail

import datetime as dt
import os
import pyomo.environ as pyo
import mpisppy.extensions.xhatbase

class Diagnoser(mpisppy.extensions.xhatbase.XhatBase):
    """
    Args:
        ph (PH object): the calling object
    """
    def __init__(self, ph):
        dirname = ph.options["diagnoser_options"]["diagnoser_outdir"]
        if os.path.exists(dirname):
            if ph.cylinder_rank == 0:
                print ("Shutting down because Diagnostic directory exists:",
                       dirname)
            quit()
        if ph.cylinder_rank == 0:
            os.mkdir(dirname) # just let it crash

        super().__init__(ph)
        self.options = self.opt.options["diagnoser_options"]
        self.dirname = self.options["diagnoser_outdir"]

    def write_loop(self):
        for sname, s in self.opt.local_scenarios.items():
            fname = self.dirname+os.sep+sname+".dag"
            with open(fname, "a") as f:
                f.write(str(self.opt._PHIter)+",")
                objfct = self.opt.saved_objectives[sname]
                f.write(str(pyo.value(objfct)))
                f.write("\n")

    def pre_iter0(self):
        return

    def post_iter0(self):
        for sname, s in self.opt.local_scenarios.items():
            fname = self.dirname+os.sep+sname+".dag"
            with open(fname, "w") as f:
                f.write(str(dt.datetime.now())+", diagnoser\n")

        self.write_loop()
        
    def miditer(self):
        return

    def enditer(self):
        self.write_loop()

    def post_everything(self):
        return
