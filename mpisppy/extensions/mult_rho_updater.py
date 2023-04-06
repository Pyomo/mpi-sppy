# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

# DLW March 2023 A simple rho updater. Hold rho as a constant multple of the convergence metric
# but only update when convergence improves
# Preference given to user-supplied converger

import math
import mpisppy.extensions.extension

import numpy as np
import mpisppy.MPI as MPI

# for ph.options['mult_rho_options']:
_mult_rho_defaults = { 'convergence_tolerance' : 1e-4,
                           'rho_update_stop_iteration' : None,
                           'rho_update_start_iteration' : None,
                           'verbose' : False,
}

_attr_to_option_name_map = {
    '_tol': 'convergence_tolerance',
    '_stop_iter' : 'rho_update_stop_iteration',
    '_start_iter' : 'rho_update_start_iteration',
    '_verbose' : 'verbose',
}


class MultRhoUpdater(mpisppy.extensions.extension.Extension):

    def __init__(self, ph):

        self.ph = ph
        self.mult_rho_options = \
            ph.options['mult_rho_options'] if 'mult_rho_options' in ph.options else dict()

        self._set_options()
        self._first_rho = None
        self.best_conv = float("inf")

        
    def _conv(self):
        if self.ph.convobject is not None:
            return self.ph.convobject.conv
        else:
            return self.ph.conv 


    def _set_options(self):
        options = self.mult_rho_options
        for attr_name, opt_name in _attr_to_option_name_map.items():
            setattr(self, attr_name, options[opt_name] if opt_name in options else _mult_rho_defaults[opt_name])

            
    def _attach_rho_ratio_data(self, ph, conv):
        if conv == None or conv == self._tol:
            return
        self.first_c = conv
        if not self.ph.multistage:
            # two stage
            for s in ph.local_scenarios.values():
                 break # arbitrary scenario
            self._first_rho = {ndn_i: rho._value for ndn_i, rho in s._mpisppy_model.rho.items()}
        else:
            # loop over all scenarios to get all nodes when multi-stage (wastes time...)
            self._first_rho = dict()
            for k, s in ph.local_scenarios.items():
                for ndn_i, rho  in s._mpisppy_model.rho.items():
                    self._first_rho[ndn_i] = rho._value


    def pre_iter0(self):
        pass

    def post_iter0(self):
        pass

    def miditer(self):

        ph = self.ph
        ph_iter = ph._PHIter
        if (self._stop_iter is not None and \
            ph_iter > self._stop_iter) \
            or \
            (self._start_iter is not None and \
             ph_iter < self._start_iter):
            return
        conv =  self._conv()
        if conv < self.best_conv:
            self.best_conv = conv
        else:
            return   # only do something if we have a new best
        if self._first_rho is None:
            self._attach_rho_ratio_data(ph, conv)  # rho / conv
        elif conv != 0:
            for s in ph.local_scenarios.values():
                for ndn_i, rho in s._mpisppy_model.rho.items():
                    rho._value = self._first_rho[ndn_i] * self.first_c / conv
            if ph.cylinder_rank == 0:
                print(f"MultRhoUpdater iter={ph_iter}; {ndn_i} now has value {rho._value}")

    def enditer(self):
        pass

    def post_everything(self):
        pass
