# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

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
from mpisppy import global_toc


class _SPPresolver(abc.ABC):
    """Defines a presolver for distributed stochastic optimization problems

    Args:
        spbase (SPBase): an SPBase object
    """

    def __init__(self, spbase):
        self._opt = None
        self.opt = spbase

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

    def __init__(self, spbase):
        super().__init__(spbase)

        self.subproblem_tighteners = {}
        for k, s in self.opt.local_subproblems.items():
            try:
                self.subproblem_tighteners[k] = it = IntervalTightener()
            except DeferredImportError:
                # User may not have extension built
                # TODO: we should print a message --
                #       especially if it needs to be
                #       specifically enabled
                return
            # ideally, we'd be able to share the `_cmodel`
            # here between interfaces, etc.
            try:
                it.set_instance(s)
            except KeyError:
                # TODO: IntervalTightener won't handle
                # every Pyomo model smoothly, see:
                # https://github.com/Pyomo/pyomo/issues/3002
                # https://github.com/Pyomo/pyomo/issues/3184
                # https://github.com/Pyomo/pyomo/issues/1864#issuecomment-1989164335
                return

    def presolve(self):
        """Run the interval tightener (FBBT):
        1. FBBT on each subproblem
        2. Narrow bounds on the nonants across all subproblems
        3. If the bounds are updated, go to (1)
        """

        update = False

        while True:
            big_iters = 0.0
            for k, it in self.subproblem_tighteners.items():
                # if the `set_instance` failed in __init__, bail out
                if it is None:
                    return False
                n_iters = it.perform_fbbt(self.opt.local_subproblems[k])
                # get the number of constraints after we do
                # FBBT so we get any updates on the subproblem
                big_iters = max(big_iters, n_iters / len(it._cmodel.constraints))

            update_this_pass = big_iters > 1.0
            update_this_pass = self.opt.allreduce_or(update_this_pass)

            if not update_this_pass:
                break

            # if we got here, there's an update on the model,
            # but maybe not on the nonants, however
            update = True

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
                    same_nonant_bounds = np.allclose(
                        global_lower_bounds[ndn], local_bounds
                    )

            # reduce upper bounds
            for ndn, local_bounds in local_upper_bounds.items():
                self.opt.comms[ndn].Allreduce(
                    [local_bounds, MPI.DOUBLE],
                    [global_upper_bounds[ndn], MPI.DOUBLE],
                    op=MPI.MIN,
                )
                if same_nonant_bounds:
                    same_nonant_bounds = np.allclose(
                        global_upper_bounds[ndn], local_bounds
                    )

            # At this point, we've either proved that
            # there are tighter bounds or not.
            # If not, we can quit
            # Reduce here for safety in case of numerical gremlins
            same_nonant_bounds = not self.opt.allreduce_or(not same_nonant_bounds)
            if same_nonant_bounds:
                break

            # otherwise, update the bounds and go to the top
            for sub_n, _, _, s in self.opt.subproblem_scenario_generator():
                feas_tol = self.subproblem_tighteners[sub_n].config.feasibility_tol
                for node in s._mpisppy_node_list:
                    for var, lb, ub in zip(
                        node.nonant_vardata_list,
                        global_lower_bounds[node.name],
                        global_upper_bounds[node.name],
                        strict=True,
                    ):
                        if ub - lb <= -feas_tol:
                            msg = f"Nonant {var.name} has lower bound greater than upper bound; lb: {lb}, ub: {ub}"
                            raise InfeasibleConstraintException(msg)
                        if (lb, ub) != var.bounds:
                            global_toc(
                                f"Tightening bounds on nonant {var.name} from {var.bounds} to {(lb, ub)}"
                            )
                        var.bounds = (lb, ub)

        return update


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


class SPPresolve(_SPPresolver):
    """Default a presolver for distributed stochastic optimization problems

    Args:
        spbase (SPBase): an SPBase object
    """

    def __init__(self, spbase):
        super().__init__(spbase)

        self.interval_tightener = SPIntervalTightener(spbase)

    def presolve(self):
        return self.interval_tightener.presolve()
