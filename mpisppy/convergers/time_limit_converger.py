#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:58:40 2024

@author: glista1
"""

import mpisppy.convergers.converger
from mpisppy import global_toc

class TimeLimitConverger(mpisppy.convergers.converger.Converger):

    def __init__(self, ph):

        self.ph = ph

    def is_converged(self):
        """ check for convergence
        Args:
            self (object): create by prep

        Returns:
           converged?: True if converged, False otherwise
        """
         
        if 'time_limit' not in self.ph.options:
            raise RuntimeError("TimeLimitConverger can only be used if time_limit is in PH options")
        
        converged = False
        if self.ph.elapsed_time >= self.ph.options['time_limit']:
            converged = True
            global_toc("Elapsed time=%f reached user-supplied time limit=%f" \
                       % (self.ph.elapsed_time, self.ph.options['time_limit']), self.ph.cylinder_rank == 0)
        
        elif self.ph.conv is not None:
            if self.ph.conv < self.ph.options["convthresh"]:
                converged = True
                global_toc("Convergence metric=%f dropped below user-supplied threshold=%f" \
                           % (self.ph.conv, self.ph.options["convthresh"]), self.ph.cylinder_rank == 0)
        
        return converged

