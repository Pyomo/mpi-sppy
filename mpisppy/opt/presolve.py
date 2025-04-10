###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

# This module defines a presolver, which is used like an extension to an
# SPBase object. However, it may be useful for this presolver extension
# to be called by *other* extensions (like a fixer extension); hence it
# is defined separately.

# For now, this mainly serves as a wrapper around Pyomo's feasibility-based
# bounds tightening capability. However, if more advanced presolving capabilities
# are added it would make sense to include them as part of this module.

import abc
import weakref

import numpy as np

from pyomo.common.errors import InfeasibleConstraintException, DeferredImportError
from pyomo.contrib.appsi.fbbt import IntervalTightener

from mpisppy import MPI

_INF = 1e100


class _SPPresolver(abc.ABC):
    """Defines a presolver for distributed stochastic optimization problems

    Args:
        spbase (SPBase): an SPBase object
    """

    def __init__(self, spbase, verbose=False):
        self._opt = None
        self.opt = spbase
        self.verbose = verbose

    @abc.abstractmethod
    def presolve(self) -> bool:
        """should return `True` if modifications are made, `False` otherwise"""

    @property
    def opt(self):
        if self._opt is None:
            return None
        return self._opt()

    @opt.setter
    def opt(self, value):
        if self._opt is None:
            self._opt = weakref.ref(value)
        else:
            raise RuntimeError("SPPresolve.opt should only be set once")


class SPIntervalTightener(_SPPresolver):
    """Interval Tightener (feasibility-based bounds tightening)
    TODO: enable options
    """

    def __init__(self, spbase, verbose=False):
        super().__init__(spbase, verbose)

        self.subproblem_tighteners = {}
        for k, s in self.opt.local_subproblems.items():
            try:
                self.subproblem_tighteners[k] = it = IntervalTightener()
            except DeferredImportError as e:
                # User may not have extension built
                raise ImportError(f"presolve needs the APPSI extensions for pyomo: {e}")

            # ideally, we'd be able to share the `_cmodel`
            # here between interfaces, etc.
            try:
                it.set_instance(s)
            except Exception as e:
                # TODO: IntervalTightener won't handle
                # every Pyomo model smoothly, see:
                # https://github.com/Pyomo/pyomo/issues/3002
                # https://github.com/Pyomo/pyomo/issues/3184
                # https://github.com/Pyomo/pyomo/issues/1864#issuecomment-1989164335
                raise Exception(
                    f"Issue with IntervalTightener; cannot apply presolve to this problem. Error: {e}"
                )

        self._lower_bound_cache = {}
        self._upper_bound_cache = {}

    def presolve(self):
        """Run the interval tightener (FBBT):
        1. FBBT on each subproblem
        2. Narrow bounds on the nonants across all subproblems
        3. If the bounds are updated, go to (1)
        """

        printed_warning = False
        update = False

        same_nonant_bounds, global_lower_bounds, global_upper_bounds = (
            self._compare_nonant_bounds()
        )

        while True:
            if not same_nonant_bounds:
                update = True

                # update the bounds if changed
                for sub_n, _, k, s in self.opt.subproblem_scenario_generator():
                    feas_tol = self.subproblem_tighteners[sub_n].config.feasibility_tol
                    for node in s._mpisppy_node_list:
                        node_comm = self.opt.comms[node.name]
                        for var, lb, ub in zip(
                            node.nonant_vardata_list,
                            global_lower_bounds[node.name],
                            global_upper_bounds[node.name],
                            # strict=True, # TODO: this is a Python 3.10+ feature
                        ):
                            if ub - lb <= -feas_tol:
                                msg = f"Nonant {var.name} has lower bound greater than upper bound; lb: {lb}, ub: {ub}"
                                raise InfeasibleConstraintException(msg)
                            if (
                                (not printed_warning or self.verbose)
                                and (lb, ub) != var.bounds
                                and (node_comm.Get_rank() == 0)
                            ):
                                if not printed_warning:
                                    msg = "WARNING: SPIntervalTightener found different bounds on nonanticipative variables from different scenarios."
                                    if self.verbose:
                                        print(msg + " See below.")
                                    else:
                                        print(msg + " Use verbose=True to see details.")
                                    printed_warning = True
                                if self.verbose:
                                    print(
                                        f"Tightening bounds on nonant {var.name} in scenario {k} from {var.bounds} to {(lb, ub)} based on global bound information."
                                    )
                            if np.isnan(lb):
                                lb = None
                            if np.isnan(ub):
                                ub = None
                            var.bounds = (lb, ub)

            # Now do FBBT
            big_iters = 0.0
            for k, it in self.subproblem_tighteners.items():
                n_iters = it.perform_fbbt(self.opt.local_subproblems[k])
                # get the number of constraints after we do
                # FBBT so we get any updates on the subproblem
                big_iters = max(big_iters, n_iters / len(it._cmodel.constraints))

            update_this_pass = big_iters > 1.0
            update_this_pass = self.opt.allreduce_or(update_this_pass)

            if not update_this_pass:
                break

            update = True

            same_nonant_bounds, global_lower_bounds, global_upper_bounds = (
                self._compare_nonant_bounds()
            )

            if same_nonant_bounds:
                # The nonant bounds did not change, so it is unlikely
                # that further rounds of FBBT will do any good.
                break

        if update:
            self._print_bound_movement()

        return update

    def _compare_nonant_bounds(self):
        # keyed by nodename
        local_lower_bounds = {}
        local_upper_bounds = {}

        global_lower_bounds = {}
        global_upper_bounds = {}

        same_nonant_bounds = True
        for k, s in self.opt.local_scenarios.items():
            for node in s._mpisppy_node_list:
                ndn = node.name
                nlen = s._mpisppy_data.nlens[ndn]

                # gather lower bounds
                scenario_node_lower_bounds = np.fromiter(
                    _lb_generator(node.nonant_vardata_list),
                    dtype=float,
                    count=nlen,
                )
                if (k, ndn) not in self._lower_bound_cache:
                    self._lower_bound_cache[k, ndn] = scenario_node_lower_bounds
                if ndn in local_lower_bounds:
                    np.maximum(
                        local_lower_bounds[ndn],
                        scenario_node_lower_bounds,
                        out=local_lower_bounds[ndn],
                    )
                    if same_nonant_bounds:
                        same_nonant_bounds = np.allclose(
                            local_lower_bounds[ndn], scenario_node_lower_bounds
                        )
                else:
                    local_lower_bounds[ndn] = scenario_node_lower_bounds
                    global_lower_bounds[ndn] = np.zeros(nlen, dtype=float)

                # gather upper bounds
                scenario_node_upper_bounds = np.fromiter(
                    _ub_generator(node.nonant_vardata_list),
                    dtype=float,
                    count=nlen,
                )
                if (k, ndn) not in self._upper_bound_cache:
                    self._upper_bound_cache[k, ndn] = scenario_node_upper_bounds
                if ndn in local_upper_bounds:
                    np.minimum(
                        local_upper_bounds[ndn],
                        scenario_node_upper_bounds,
                        out=local_upper_bounds[ndn],
                    )
                    if same_nonant_bounds:
                        same_nonant_bounds = np.allclose(
                            local_upper_bounds[ndn], scenario_node_upper_bounds
                        )
                else:
                    local_upper_bounds[ndn] = scenario_node_upper_bounds
                    global_upper_bounds[ndn] = np.zeros(nlen, dtype=float)

        # reduce lower bounds
        for ndn, local_bounds in local_lower_bounds.items():
            self.opt.comms[ndn].Allreduce(
                [local_bounds, MPI.DOUBLE],
                [global_lower_bounds[ndn], MPI.DOUBLE],
                op=MPI.MAX,
            )
            if same_nonant_bounds:
                same_nonant_bounds = np.allclose(global_lower_bounds[ndn], local_bounds)

        # reduce upper bounds
        for ndn, local_bounds in local_upper_bounds.items():
            self.opt.comms[ndn].Allreduce(
                [local_bounds, MPI.DOUBLE],
                [global_upper_bounds[ndn], MPI.DOUBLE],
                op=MPI.MIN,
            )
            if same_nonant_bounds:
                same_nonant_bounds = np.allclose(global_upper_bounds[ndn], local_bounds)

        # Once we've done one pass of FBBT, we can quit here
        # if the bounds are the same in every scenario
        # Reduce here for safety in case of numerical gremlins
        same_nonant_bounds = not self.opt.allreduce_or(not same_nonant_bounds)

        return same_nonant_bounds, global_lower_bounds, global_upper_bounds

    def _print_bound_movement(self):
        lower_bound_movement = {}
        upper_bound_movement = {}

        global_lower_bound_movement = {}
        global_upper_bound_movement = {}

        for k, s in self.opt.local_scenarios.items():
            for node in s._mpisppy_node_list:
                ndn = node.name
                nlen = s._mpisppy_data.nlens[ndn]

                original_lower_bounds = self._lower_bound_cache[k, ndn]

                current_lower_bounds = np.fromiter(
                    _lb_generator(node.nonant_vardata_list),
                    dtype=float,
                    count=nlen,
                )

                scenario_lower_bound_movement = (
                    current_lower_bounds - original_lower_bounds
                )
                if ndn in lower_bound_movement:
                    np.maximum(
                        lower_bound_movement[ndn],
                        scenario_lower_bound_movement,
                        out=lower_bound_movement[ndn],
                    )
                else:
                    lower_bound_movement[ndn] = scenario_lower_bound_movement
                    global_lower_bound_movement[ndn] = np.zeros(nlen, dtype=float)

                original_upper_bounds = self._upper_bound_cache[k, ndn]

                current_upper_bounds = np.fromiter(
                    _ub_generator(node.nonant_vardata_list),
                    dtype=float,
                    count=nlen,
                )

                scenario_upper_bound_movement = (
                    original_upper_bounds - current_upper_bounds
                )
                if ndn in upper_bound_movement:
                    np.maximum(
                        upper_bound_movement[ndn],
                        scenario_upper_bound_movement,
                        out=upper_bound_movement[ndn],
                    )
                else:
                    upper_bound_movement[ndn] = scenario_upper_bound_movement
                    global_upper_bound_movement[ndn] = np.zeros(nlen, dtype=float)

        # reduce lower bounds
        for ndn, local_bounds in lower_bound_movement.items():
            self.opt.comms[ndn].Allreduce(
                [local_bounds, MPI.DOUBLE],
                [global_lower_bound_movement[ndn], MPI.DOUBLE],
                op=MPI.MAX,
            )

        # reduce upper bounds
        for ndn, local_bounds in upper_bound_movement.items():
            self.opt.comms[ndn].Allreduce(
                [local_bounds, MPI.DOUBLE],
                [global_upper_bound_movement[ndn], MPI.DOUBLE],
                op=MPI.MAX,
            )

        printed_nodes = set()
        bounds_tightened = 0
        for k, s in self.opt.local_scenarios.items():
            for node in s._mpisppy_node_list:
                ndn = node.name
                if ndn in printed_nodes:
                    continue
                node_comm = self.opt.comms[ndn]
                for original_lower_bound, lower_bound_move, var in zip(
                    self._lower_bound_cache[k, ndn],
                    global_lower_bound_movement[ndn],
                    node.nonant_vardata_list,
                ):
                    if (
                        (lower_bound_move > 1e-6)
                        and (node_comm.Get_rank() == 0)
                    ):
                        bounds_tightened += 1
                        if self.verbose:
                            print(
                                f"Tightened lower bound for {var.name} from {original_lower_bound} to {var.lb}"
                            )
                for original_upper_bound, upper_bound_move, var in zip(
                    self._upper_bound_cache[k, ndn],
                    global_upper_bound_movement[ndn],
                    node.nonant_vardata_list,
                ):
                    if (
                        (upper_bound_move > 1e-6)
                        and (node_comm.Get_rank() == 0)
                    ):
                        bounds_tightened += 1
                        if self.verbose:
                            print(
                                f"Tightened upper bound for {var.name} from {original_upper_bound} to {var.ub}"
                            )

                printed_nodes.add(ndn)

        if (
            bounds_tightened > 0
            and self.opt.cylinder_rank == 0
        ):
            print(f"SPIntervalTightener tightend {bounds_tightened} bounds.")


def _lb_generator(var_iterable):
    for v in var_iterable:
        lb = v.lb
        if lb is None:
            yield -_INF
        yield lb


def _ub_generator(var_iterable):
    for v in var_iterable:
        ub = v.ub
        if ub is None:
            yield _INF
        yield ub


class SPPresolve(_SPPresolver):
    """Default a presolver for distributed stochastic optimization problems

    Args:
        spbase (SPBase): an SPBase object
    """

    def __init__(self, spbase, verbose=False):
        super().__init__(spbase, verbose)

        self.interval_tightener = SPIntervalTightener(spbase, verbose)

    def presolve(self):
        return self.interval_tightener.presolve()
