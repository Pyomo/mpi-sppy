###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import math
import pyomo.environ as pyo
import numpy as np
from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
from mpisppy.cylinders.spwindow import Field
from mpisppy.utils.sputils import is_persistent
from mpisppy import MPI

class ReducedCostsSpoke(LagrangianOuterBound):

    send_fields = (*LagrangianOuterBound.send_fields, Field.EXPECTED_REDUCED_COST, Field.SCENARIO_REDUCED_COST,
                   Field.NONANT_LOWER_BOUNDS, Field.NONANT_UPPER_BOUNDS,)
    receive_fields = (*LagrangianOuterBound.receive_fields,)

    converger_spoke_char = 'R'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bound_tol = self.opt.options['rc_bound_tol']
        self.consensus_threshold = np.sqrt(self.bound_tol)

    def register_send_fields(self) -> None:

        super().register_send_fields()

        if not hasattr(self.opt, "local_scenarios"):
            raise RuntimeError("Provided SPBase object does not have local_scenarios attribute")

        if len(self.opt.local_scenarios) == 0:
            raise RuntimeError("Rank has zero local_scenarios")

        rbuflen = 0
        for s in self.opt.local_scenarios.values():
            rbuflen += len(s._mpisppy_data.nonant_indices)

        self.nonant_length = self.opt.nonant_length

        self._modeler_fixed_nonants = {}

        for k,s in self.opt.local_scenarios.items():
            self._modeler_fixed_nonants[s] = set()
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.fixed:
                    self._modeler_fixed_nonants[s].add(ndn_i)
                ## End if
            ## End for
        ## End for

        scenario_buffer_len = 0
        for s in self.opt.local_scenarios.values():
            scenario_buffer_len += len(s._mpisppy_data.nonant_indices)
        self._scenario_rc_buffer = np.zeros(scenario_buffer_len)

        self.initialize_bound_fields()
        self.create_integer_variable_where()

        return

    def initialize_bound_fields(self):
        self._nonant_lower_bounds = self.send_buffers[Field.NONANT_LOWER_BOUNDS].value_array()
        self._nonant_upper_bounds = self.send_buffers[Field.NONANT_UPPER_BOUNDS].value_array()

        self._nonant_lower_bounds[:] = -np.inf
        self._nonant_upper_bounds[:] = np.inf

        for s in self.opt.local_scenarios.values():
            scenario_lower_bounds = np.fromiter(
                    _lb_generator(s._mpisppy_data.nonant_indices.values()),
                    dtype=float,
                    count=len(s._mpisppy_data.nonant_indices),
                )
            np.maximum(self._nonant_lower_bounds, scenario_lower_bounds, out=self._nonant_lower_bounds)

            scenario_upper_bounds = np.fromiter(
                    _ub_generator(s._mpisppy_data.nonant_indices.values()),
                    dtype=float,
                    count=len(s._mpisppy_data.nonant_indices),
                )
            np.minimum(self._nonant_upper_bounds, scenario_upper_bounds, out=self._nonant_upper_bounds)

    def create_integer_variable_where(self):
        self._integer_variable_where = np.full(len(self._nonant_lower_bounds), False)
        for s in self.opt.local_scenarios.values():
            for idx, xvar in enumerate(s._mpisppy_data.nonant_indices.values()):
                if xvar.is_integer():
                    self._integer_variable_where[idx] = True

    @property
    def rc_global(self):
        return self.send_buffers[Field.EXPECTED_REDUCED_COST].value_array()

    @rc_global.setter
    def rc_global(self, vals):
        arr = self.send_buffers[Field.EXPECTED_REDUCED_COST].value_array()
        arr[:] = vals
        return

    @property
    def rc_scenario(self):
        return self.send_buffers[Field.SCENARIO_REDUCED_COST].value_array()

    @rc_scenario.setter
    def rc_scenario(self, vals):
        arr = self.send_buffers[Field.SCENARIO_REDUCED_COST].value_array()
        arr[:] = vals
        return

    def lagrangian_prep(self):
        """
        same as base class, but relax the integer variables and
        attach the reduced cost suffix
        """
        relax_integer_vars = pyo.TransformationFactory("core.relax_integer_vars")
        for s in self.opt.local_subproblems.values():
            relax_integer_vars.apply_to(s)
            s.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        super().lagrangian_prep()

    def lagrangian(self, need_solution=True):
        if not need_solution:
            raise RuntimeError("ReducedCostsSpoke always needs a solution to work")
        bound = super().lagrangian(need_solution=need_solution)
        if bound is not None:
            self.extract_and_store_reduced_costs()
            self.extract_and_store_updated_nonant_bounds(bound)
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

        self.put_send_buffer(
            self.send_buffers[Field.EXPECTED_REDUCED_COST],
            Field.EXPECTED_REDUCED_COST,
        )
        self.put_send_buffer(
            self.send_buffers[Field.SCENARIO_REDUCED_COST],
            Field.SCENARIO_REDUCED_COST,
        )

    def extract_and_store_updated_nonant_bounds(self, lr_outer_bound):
        # update the best bound from the hub
        # as a side effect, calls update_receive_buffers
        # TODO: fix this side effect 
        if self.got_kill_signal():
            return

        self.update_innerbounds()        

        if math.isinf(self.BestInnerBound):
            # can do anything with no bound
            return
        
        nonzero_rc = np.where(self.rc_global==0, np.nan, self.rc_global)
        bound_tightening = np.divide(self.BestInnerBound - lr_outer_bound, nonzero_rc)

        tighten_upper = np.where(bound_tightening>0, bound_tightening, np.nan)
        tighten_lower = np.where(bound_tightening<0, bound_tightening, np.nan)

        tighten_upper += self._nonant_lower_bounds 
        tighten_lower += self._nonant_upper_bounds

        # max of existing lower and new lower
        np.fmax(tighten_lower, self._nonant_lower_bounds, out=self._nonant_lower_bounds)

        # min of existing upper and new upper
        np.fmin(tighten_upper, self._nonant_upper_bounds, out=self._nonant_upper_bounds)

        # ceiling of lower bounds for integer variables
        np.ceil(self._nonant_lower_bounds, out=self._nonant_lower_bounds, where=self._integer_variable_where)
        # floor of upper bounds for integer variables
        np.floor(self._nonant_upper_bounds, out=self._nonant_upper_bounds, where=self._integer_variable_where)

        self.put_send_buffer(
            self.send_buffers[Field.NONANT_LOWER_BOUNDS],
            Field.NONANT_LOWER_BOUNDS,
        )
        self.put_send_buffer(
            self.send_buffers[Field.NONANT_UPPER_BOUNDS],
            Field.NONANT_UPPER_BOUNDS,
        )

    def main(self):
        # need the solution for ReducedCostsSpoke
        super().main(need_solution=True)


def _lb_generator(var_iterable):
    for v in var_iterable:
        lb = v.lb
        if lb is None:
            yield -np.inf
        yield lb


def _ub_generator(var_iterable):
    for v in var_iterable:
        ub = v.ub
        if ub is None:
            yield np.inf
        yield ub
