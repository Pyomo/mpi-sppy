###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import pyomo.environ as pyo
import numpy as np
from mpisppy.cylinders.spcommunicator import communicator_array
from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
from mpisppy.utils.sputils import is_persistent 
from mpisppy import MPI

class ReducedCostsSpoke(LagrangianOuterBound):

    converger_spoke_char = 'R'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bound_tol = self.opt.options['rc_bound_tol']
        self.consensus_threshold = np.sqrt(self.bound_tol)

    def make_windows(self):
        if not hasattr(self.opt, "local_scenarios"):
            raise RuntimeError("Provided SPBase object does not have local_scenarios attribute")

        if len(self.opt.local_scenarios) == 0:
            raise RuntimeError("Rank has zero local_scenarios")

        rbuflen = 2
        for s in self.opt.local_scenarios.values():
            rbuflen += len(s._mpisppy_data.nonant_indices)

        self.nonant_length = self.opt.nonant_length

        self._modeler_fixed_nonants = {}

        for k,s in self.opt.local_scenarios.items():
            self._modeler_fixed_nonants[s] = set()
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.fixed:
                    self._modeler_fixed_nonants[s].add(ndn_i)

        scenario_buffer_len = 0
        for s in self.opt.local_scenarios.values():
            scenario_buffer_len += len(s._mpisppy_data.nonant_indices)
        self._scenario_rc_buffer = np.zeros(scenario_buffer_len)
        # over load the _bound attribute here
        # so the rest of the class works as expected
        # first float will be the bound we're sending
        # indices 1:1+self.nonant_length will be the
        # expected reduced costs,
        # 1+self.nonant_length:1+self.nonant_length+|S|*self.nonant_length
        # will be the scenario reduced costs, and
        # the last index will be the serial number
        sbuflen = 1 + self.nonant_length + scenario_buffer_len

        self._make_windows(sbuflen, rbuflen)
        self._locals = communicator_array(rbuflen)
        self._bound = communicator_array(sbuflen)
        # print(f"nonant_length: {self.nonant_length}, integer_nonant_length: {self.integer_nonant_length}")

    @property
    def rc_global(self):
        return self._bound[1:1+self.nonant_length]

    @rc_global.setter
    def rc_global(self, vals):
        self._bound[1:1+self.nonant_length] = vals

    @property
    def rc_scenario(self):
        return self._bound[1+self.nonant_length:1+self.nonant_length+len(self._scenario_rc_buffer)]

    @rc_scenario.setter
    def rc_scenario(self, vals):
        self._bound[1+self.nonant_length:1+self.nonant_length+len(self._scenario_rc_buffer)] = vals

    def lagrangian_prep(self):
        """
        same as base class, but relax the integer variables and
        attach the reduced cost suffix
        """
        # Split up PH_Prep? Prox option is important for APH.
        # Seems like we shouldn't need the Lagrangian stuff, so attach_prox=False
        # Scenarios are created here
        self.opt.PH_Prep(attach_prox=False)
        self.opt._reenable_W()

        relax_integer_vars = pyo.TransformationFactory("core.relax_integer_vars")
        for s in self.opt.local_subproblems.values():
            relax_integer_vars.apply_to(s)
            s.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        self.opt._create_solvers(presolve=False)

    def lagrangian(self, need_solution=True):
        if not need_solution:
            raise RuntimeError("ReducedCostsSpoke always needs a solution to work")
        bound = super().lagrangian(need_solution=need_solution)
        if bound is not None:
            self.extract_and_store_reduced_costs()
        return bound

    def extract_and_store_reduced_costs(self):
        self.opt.Compute_Xbar()
        # NaN will signal that the x values do not agree in
        # every scenario, we can't extract an expected reduced
        # cost
        # Note: might be good ta have a rc, even if scenarios are not
        # in complete agreement, e.g. for more aggressive fixing
        # would probably need additional info about where scenarios disagree
        rc = np.zeros(self.nonant_length)

        for sub in self.opt.local_subproblems.values():
            if is_persistent(sub._solver_plugin):
                # Note: what happens with non-persistent solvers? 
                # - if rc is accepted as a model suffix by the solver (e.g. gurobi shell), it is loaded in postsolve
                # - if not, the solver should throw an error
                # - direct solvers seem to behave the same as persistent solvers
                # GurobiDirect needs vars_to_load argument
                # XpressDirect loads for all vars by default - TODO: should notify someone of this inconsistency
                vars_to_load = [x for sn in sub.scen_list for _, x in self.opt.local_scenarios[sn]._mpisppy_data.nonant_indices.items()]
                sub._solver_plugin.load_rc(vars_to_load=vars_to_load)

            for sn in sub.scen_list:
                s = self.opt.local_scenarios[sn]
                for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
                    # fixed by modeler
                    if ndn_i in self._modeler_fixed_nonants[s]:
                        rc[ci] = np.nan
                        continue
                    xb = s._mpisppy_model.xbars[ndn_i].value
                    # check variance of xb to determine if consensus achieved
                    var_xb = pyo.value(s._mpisppy_model.xsqbars[ndn_i]) - xb * xb
            
                    if var_xb  > self.consensus_threshold * self.consensus_threshold:
                        rc[ci] = np.nan
                        continue

                    # solver takes care of sign of rc, based on lb, ub and max,min
                    # rc can be of wrong sign if numerically 0 - accepted here, checked in extension
                    if (xvar.lb is not None and xb - xvar.lb <= self.bound_tol) or (xvar.ub is not None and xvar.ub - xb <= self.bound_tol):
                        rc[ci] += sub._mpisppy_probability * sub.rc[xvar]
                    # not close to either bound -> rc = nan
                    else:
                        rc[ci] = np.nan

        self._scenario_rc_buffer.fill(0)
        ci = 0 # buffer index
        for sub in self.opt.local_subproblems.values():
            for sn in sub.scen_list:
                s = self.opt.local_scenarios[sn]
                for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                    # fixed by modeler
                    if ndn_i in self._modeler_fixed_nonants[s]:
                        self._scenario_rc_buffer[ci] = np.nan
                    else:
                        self._scenario_rc_buffer[ci] = sub.rc[xvar]
                    ci += 1
        self.rc_scenario = self._scenario_rc_buffer
        # print(f"In ReducedCostsSpoke; {self.rc_scenario=}")

        rcg = np.zeros(self.nonant_length)
        self.cylinder_comm.Allreduce(rc, rcg, op=MPI.SUM)
        self.rc_global = rcg

    def main(self):
        # need the solution for ReducedCostsSpoke
        super().main(need_solution=True)
