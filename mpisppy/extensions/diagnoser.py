# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
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
from mpisppy.utils.sputils import find_active_objective

class Diagnoser(mpisppy.extensions.xhatbase.XhatBase):
    """
    Args:
        ph (PH object): the calling object
        rank (int): mpi process rank of currently running process
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
        self.options = self.ph.options["diagnoser_options"]
        self.dirname = self.options["diagnoser_outdir"]

    def write_loop(self):
        """ Bundles are special. Also: this code needs help
        from the ph object to be more efficient...
        """
        for sname, s in self.ph.local_scenarios.items():
            bundling = self.ph.bundling
            fname = self.dirname+os.sep+sname+".dag"
            with open(fname, "a") as f:
                f.write(str(self.ph._PHIter)+",")
                if not bundling:
                    objfct = find_active_objective(s)
                    f.write(str(pyo.value(objfct)))
                else:
                    f.write("Bundling"+",")
                    f.write(str(pyo.value(self.ph.saved_objs[sname])))
                f.write("\n")

    def pre_iter0(self):
        return

    def post_iter0(self):
        for sname, s in self.ph.local_scenarios.items():
            fname = self.dirname+os.sep+sname+".dag"
            with open(fname, "w") as f:
                f.write(str(dt.datetime.now())+", diagnoser\n")

        self.write_loop()
        
    def miditer(self, PHIter, conv):
        return

    def enditer(self, PHIter):
        self.write_loop()

    def post_everything(self, PHIter, conv):
        return
