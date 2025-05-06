###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2025, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
'''
This class is implemented as an extension to be used in mpi-sppy to add a callback to a persistent
solver to implement a time-dependent target MIP gap.
For now, the only solver supported is GurobiPersistent.
'''

import mpisppy.extensions.extension
import mpisppy.utils.sputils as sputils
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent

from gurobipy import GRB

class TimedMIPGapCB(mpisppy.extensions.extension.Extension):
    '''
    This extension adds a solver callback function that implements a time-dependent target MIP gap.
    The curve is defined by a sequence of (time,gap) pairs, monotonically increasing in both dimensions.
    For each (t,g) pair, when the solver reaches time t, it relaxes the target MIP gap to g.

    This class requires the following options:
    'timed_mipgap':
        'timecurve': string of gap:time pairs separated by spaces. Example: "0.02:100 0.05:200"

    Attributes
    ----------
    timecurve: dict of {gap:time} with sequence of (time,gap) pairs.

    Reference
    ---------
    Based on post found at:
    https://support.gurobi.com/hc/en-us/articles/360047717291-How-do-I-use-callbacks-to-terminate-the-solver-
    (Oct 2023)
    '''

    def __init__(self, ph):

        self.ph = ph

        self._set_options()

    def _set_options(self):
        ph = self.ph
        if 'timed_mipgap' not in ph.options:
            raise RuntimeError('Did not find "timed_mipgap" options')
        timecurve_str = ph.options['timed_mipgap']['timecurve']
        self.timecurve = {float(ent.split(':')[0]):float(ent.split(':')[1]) for ent in timecurve_str.split(' ')}

    def iter0_post_solver_creation(self):
        ph = self.ph
        for sname, s in ph.local_subproblems.items():
            if not hasattr(s, '_solver_plugin'):
                raise RuntimeError('Solver must be created before calling callback extension')
            if not (sputils.is_persistent(s._solver_plugin)):
                raise RuntimeError('Solvers must be persistent for callback definition')
            if not s._solver_plugin.has_instance():
                raise RuntimeError('Solver must be instantiated before calling callback extension')
            if not isinstance(s._solver_plugin,GurobiPersistent):
                raise RuntimeError('Currently, only GurobiPersistent solver is supported.')
            
            def cb_fun(cb_m,cb_opt,cb_wh):
                '''
                callback function
                '''
                return self._timecurve_cb(cb_m,cb_opt,cb_wh,self.timecurve)
            s._solver_plugin.set_callback(cb_fun)

    @staticmethod
    def _timecurve_cb(cb_model, cb_opt, cb_where,
                     timecurve_dict):
        '''
        Inputs
        ------
        cb_model: Pyomo ConcreteModel
        cb_opt: SolverFactory model of gurobi_persistent type
        cb_where: argument that indicates where in the algorith the callback is being called from

        timecurve_dict: dict of {gap:time} with:
            time: time in sec
            gap: target MIP gap (in p.u.)
        Note that (time,gap) only define a meaningful timecurve if there is monotonicity, so monotonicity
        is assumed. However, it is not verified.

        '''

        if cb_where == GRB.Callback.MIP:
            grb_m = cb_opt._solver_model # gurobipy model
            runtime = grb_m.cbGet(GRB.Callback.RUNTIME)
            objbst = grb_m.cbGet(GRB.Callback.MIP_OBJBST)
            objbnd = grb_m.cbGet(GRB.Callback.MIP_OBJBND)
            gap = abs((objbst - objbnd) / objbst)

            for tc_gap,tc_t in timecurve_dict.items():
                if runtime > tc_t and gap < tc_gap:
                    grb_m.terminate()
    
