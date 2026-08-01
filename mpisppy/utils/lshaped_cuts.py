###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Cut generators used by the mpi-sppy L-shaped method."""

from pyomo.core.base.block import declare_custom_block
from pyomo.core import Constraint, Var
from pyomo.core.base.componentuid import ComponentUID
from pyomo.repn.standard_repn import generate_standard_repn
import pyomo.environ as pe
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
import pyomo.contrib.benders.benders_cuts as bc

from mpisppy import MPI
from mpisppy.spopt import set_instance_retry
from mpisppy.utils.sputils import find_active_objective

import logging
import numpy as np

logger = logging.getLogger(__name__)


solver_dual_sign_convention = dict()
solver_dual_sign_convention['ipopt'] = -1
solver_dual_sign_convention['gurobi'] = -1
solver_dual_sign_convention['gurobi_direct'] = -1
solver_dual_sign_convention['gurobi_persistent'] = -1
solver_dual_sign_convention['cplex'] = -1
solver_dual_sign_convention['cplex_direct'] = -1
solver_dual_sign_convention['cplexdirect'] = -1
solver_dual_sign_convention['cplex_persistent'] = -1
solver_dual_sign_convention['glpk'] = -1
solver_dual_sign_convention['cbc'] = -1
solver_dual_sign_convention['xpress_direct'] = -1
solver_dual_sign_convention['xpress_persistent'] = -1


class StandardLPL1CutGenerator:
    """Standard LP Benders cuts with L1-normalized feasibility cuts.

    This generator is intentionally opt-in and assumes that each subproblem is
    an LP. For a fixed incumbent first-stage solution ``xbar``, it first solves
    the ordinary scenario recourse LP. In compact notation this is

    ``Q_s(xbar) = min_y { q_s^T y : W_s y >= h_s - T_s xbar }``.

    The implementation obtains the same dual information by keeping scenario
    copies of the complicating variables and adding temporary fixing rows:

    ``min_{x_s,y} q_s^T y``
    ``s.t. W_s y + T_s x_s >= h_s``
    ``     x_s = xbar``.

    When this LP is optimal, the dual multipliers on the temporary fixing
    constraints define an optimality cut. After applying the Xpress dual sign
    multiplier used by this implementation, the generated cut has the form

    ``Q_s(xbar) - pi_s^T (x - xbar) <= eta_s``.

    If the ordinary recourse LP is infeasible, the generator clones the
    subproblem and solves an L1 feasibility LP. The original objective is
    deactivated, a nonnegative scalar ``z`` is minimized, and every active
    original row is relaxed so that its violation is bounded by ``z``:

    ``min_{x_s,y,z} z``

    ``s.t. body_i(x_s, y) - ub_i <= z`` for upper-bounded rows,

    ``     lb_i - body_i(x_s, y) <= z`` for lower-bounded rows,

    ``     body_i(x_s, y) - rhs_i <= z`` for equality rows,

    ``     rhs_i - body_i(x_s, y) <= z`` for equality rows,

    ``     z >= 0``,

    ``     x_s = xbar``.

    The dual multipliers on the fixing constraints of this L1 problem define
    the feasibility cut

    ``z_s(xbar) - rho_s^T (x - xbar) <= 0``.

    This class does not add an objective-vs-eta row to the L1 feasibility LP.
    That is the main formulation difference from Pyomo's generic feasibility
    subproblem transformation used by the default cut generator.
    """

    _solver_dual_sign_convention = {
        "xpress": -1,
        "xpress_direct": -1,
        "xpress_persistent": -1,
    }

    _optimal_tc = {pe.TerminationCondition.optimal}
    _infeasible_tc = {
        pe.TerminationCondition.infeasible,
        pe.TerminationCondition.infeasibleOrUnbounded,
    }

    def __init__(self):
        self.root_vars = None
        self.root_etas = None
        self._root_eta_index = None
        self.tol = None
        self.comm = None
        self.ls = None
        self.cuts = pe.ConstraintList()
        self.subproblems = []
        self.complicating_vars_maps = []
        self.subproblem_solvers = []
        self.subproblem_solver_names = []
        self.subproblem_solver_options = []
        self.subproblem_indices = []

    def set_input(self, root_vars, tol=1e-6, comm=None):
        """Store root variables, cut tolerance, and MPI communicator.

        Args:
            root_vars (iterable): Root/master variables used in Benders cuts.
            tol (float): Violation tolerance for deciding whether to add a cut.
            comm (MPI communicator, optional): Communicator used to reduce cuts
                across ranks. Defaults to ``MPI.COMM_WORLD``.
        """
        self.root_vars = list(root_vars)
        self.tol = tol
        self.comm = comm if comm is not None else MPI.COMM_WORLD

    def set_ls(self, ls):
        """Attach the owning L-shaped object and install the cut container.

        Args:
            ls (LShapedMethod): Owning L-shaped method instance. The root model
                on this object receives the generated cut ``ConstraintList``.
        """
        self.ls = ls
        if not hasattr(ls.root, "_standard_lshaped_l1_cuts"):
            ls.root.add_component("_standard_lshaped_l1_cuts", self.cuts)
        self.root_etas = list(ls.root.eta.values())
        self._root_eta_index = {s: i for i, s in enumerate(ls.root.eta.keys())}

    def global_num_subproblems(self):
        """Return the global number of scenario subproblems.

        Returns:
            int: Number of scenario eta variables, equivalently the number of
            global scenario subproblems represented by the root problem.
        """
        return len(self.root_etas)

    def add_subproblem(self, subproblem_fn, subproblem_fn_kwargs, root_eta,
                       subproblem_solver="xpress_persistent",
                       relax_subproblem_cons=False,
                       subproblem_solver_options=None):
        """Create and register the local scenario subproblem handled here.

        Args:
            subproblem_fn (callable): Function that returns a scenario model and
                a map from root variables to scenario copies.
            subproblem_fn_kwargs (dict): Keyword arguments for
                ``subproblem_fn``. Must include ``scenario_name``.
            root_eta (VarData): Root eta variable for this scenario. This
                argument is accepted for API compatibility; eta lookup is based
                on the scenario name.
            subproblem_solver (str or Solver): Solver name or solver object used
                for the recourse LP.
            relax_subproblem_cons (bool): Accepted for API compatibility with
                the default cut generator. It is not used by this generator.
            subproblem_solver_options (dict, optional): Solver options applied
                to the recourse and L1 feasibility solvers.

        Raises:
            ValueError: If the scenario subproblem is not linear.
            RuntimeError: If a solver object name cannot be determined.
        """
        scenario_name = subproblem_fn_kwargs["scenario_name"]
        if scenario_name not in self._root_eta_index:
            return

        subproblem, complicating_vars_map = subproblem_fn(**subproblem_fn_kwargs)
        self._validate_linear_subproblem(subproblem)
        if not hasattr(subproblem, "dual"):
            subproblem.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

        if not hasattr(subproblem, "_mpisppy_lshaped_bound_cons"):
            subproblem._mpisppy_lshaped_bound_cons = pe.ConstraintList()
        self._convert_recourse_bounds_to_constraints(subproblem, complicating_vars_map)

        solver, solver_name = self._make_solver(subproblem_solver, subproblem_solver_options)
        if isinstance(solver, PersistentSolver):
            set_instance_retry(subproblem, solver, scenario_name)

        self.subproblems.append(subproblem)
        self.complicating_vars_maps.append(complicating_vars_map)
        self.subproblem_solvers.append(solver)
        self.subproblem_solver_names.append(solver_name)
        self.subproblem_solver_options.append(dict(subproblem_solver_options or {}))
        self.subproblem_indices.append(self._root_eta_index[scenario_name])

    def generate_cut(self):
        """Solve local subproblems, reduce coefficients, and add violated cuts.

        Returns:
            list: Pyomo cut constraints added on this call.
        """
        nsubs = self.global_num_subproblems()
        nroot = len(self.root_vars)
        constants = np.zeros(nsubs, dtype="d")
        coefficients = np.zeros(nsubs * nroot, dtype="d")
        needs_cut = np.zeros(nsubs, dtype="d")
        infeasible = np.zeros(nsubs, dtype="d")

        for local_ndx, subproblem in enumerate(self.subproblems):
            global_ndx = self.subproblem_indices[local_ndx]
            root_eta = self.root_etas[global_ndx]
            result = self._solve_recourse_or_l1(local_ndx, root_eta)
            constants[global_ndx] = result["constant"]
            needs_cut[global_ndx] = float(result["needs_cut"])
            infeasible[global_ndx] = float(result["infeasible"])
            offset = global_ndx * nroot
            for i, coeff in enumerate(result["coefficients"]):
                coefficients[offset + i] = coeff

        global_constants = np.zeros(nsubs, dtype="d")
        global_coefficients = np.zeros(nsubs * nroot, dtype="d")
        global_needs_cut = np.zeros(nsubs, dtype="d")
        global_infeasible = np.zeros(nsubs, dtype="d")
        self.comm.Allreduce([constants, MPI.DOUBLE], [global_constants, MPI.DOUBLE])
        self.comm.Allreduce([coefficients, MPI.DOUBLE], [global_coefficients, MPI.DOUBLE])
        self.comm.Allreduce([needs_cut, MPI.DOUBLE], [global_needs_cut, MPI.DOUBLE])
        self.comm.Allreduce([infeasible, MPI.DOUBLE], [global_infeasible, MPI.DOUBLE])

        cuts_added = []
        for global_ndx in range(nsubs):
            if global_needs_cut[global_ndx] <= 0.5:
                continue
            offset = global_ndx * nroot
            cut_lhs = global_constants[global_ndx]
            for i, root_var in enumerate(self.root_vars):
                cut_lhs -= global_coefficients[offset + i] * (root_var - root_var.value)
            if global_infeasible[global_ndx] > 0.5:
                new_cut = self.cuts.add(cut_lhs <= 0)
            else:
                new_cut = self.cuts.add(cut_lhs <= self.root_etas[global_ndx])
            cuts_added.append(new_cut)
        return cuts_added

    def _solve_recourse_or_l1(self, local_ndx, root_eta):
        """Solve ordinary recourse, falling back to the L1 feasibility LP.

        Args:
            local_ndx (int): Local subproblem index in this rank's registered
                subproblem lists.
            root_eta (VarData): Root eta variable associated with the scenario.

        Returns:
            dict: Cut data with ``constant``, ``coefficients``, ``needs_cut``,
            and ``infeasible`` entries.

        Raises:
            RuntimeError: If the recourse solve returns an unsupported
            termination condition.
            NotImplementedError: If the solver dual sign convention is unknown.
        """
        subproblem = self.subproblems[local_ndx]
        solver = self.subproblem_solvers[local_ndx]
        solver_name = self.subproblem_solver_names[local_ndx]
        sign = self._solver_sign(solver_name)
        fix_cons = self._add_fixing_constraints(subproblem, self.complicating_vars_maps[local_ndx])
        try:
            res = self._solve_model(subproblem, solver, solver_name, fix_cons.values())
            tc = res.solver.termination_condition
            if tc in self._optimal_tc:
                obj = pe.value(find_active_objective(subproblem).expr)
                coeffs = self._fixing_dual_coefficients(subproblem, fix_cons, sign)
                return {
                    "constant": obj,
                    "coefficients": coeffs,
                    "needs_cut": obj - pe.value(root_eta) > self.tol,
                    "infeasible": False,
                }
            if tc in self._infeasible_tc:
                return self._solve_l1_feasibility(local_ndx, solver_name)
            raise RuntimeError(f"Unexpected subproblem termination condition: {tc}")
        finally:
            self._remove_fixing_constraints(subproblem, solver, fix_cons.values())

    def _solve_l1_feasibility(self, local_ndx, solver_name):
        """Build and solve the cloned L1 feasibility model for one scenario.

        Args:
            local_ndx (int): Local subproblem index in this rank's registered
                subproblem lists.
            solver_name (str): Solver interface name used for the L1 model.

        Returns:
            dict: Feasibility cut data with ``constant``, ``coefficients``,
            ``needs_cut``, and ``infeasible`` entries.

        Raises:
            RuntimeError: If the L1 feasibility LP does not solve to optimality.
            NotImplementedError: If the solver dual sign convention is unknown.
        """
        base = self.subproblems[local_ndx]
        cmap = self.complicating_vars_maps[local_ndx]
        subproblem = base.clone()
        if not hasattr(subproblem, "dual"):
            subproblem.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
        clone_cmap = pe.ComponentMap(
            (root_var, ComponentUID(sub_var).find_component_on(subproblem))
            for root_var, sub_var in cmap.items()
        )
        self._build_l1_model(subproblem)
        fix_cons = self._add_fixing_constraints(subproblem, clone_cmap)
        solver = pe.SolverFactory(solver_name)
        for k, v in self.subproblem_solver_options[local_ndx].items():
            solver.options[k] = v
        if isinstance(solver, PersistentSolver):
            set_instance_retry(subproblem, solver, "lshaped_l1")
        res = self._solve_model(subproblem, solver, solver_name, [])
        tc = res.solver.termination_condition
        if tc not in self._optimal_tc:
            raise RuntimeError(f"L1 feasibility subproblem did not solve to optimality: {tc}")
        zval = pe.value(subproblem._mpisppy_l1_z)
        coeffs = self._fixing_dual_coefficients(
            subproblem, fix_cons, self._solver_sign(solver_name)
        )
        return {
            "constant": zval,
            "coefficients": coeffs,
            "needs_cut": zval > self.tol,
            "infeasible": True,
        }

    def _build_l1_model(self, subproblem):
        """Replace the active LP rows by row-violation constraints using ``z``.

        Args:
            subproblem (ConcreteModel): Cloned scenario model to transform in
                place into the L1 feasibility LP.
        """
        obj = find_active_objective(subproblem)
        obj.deactivate()
        subproblem._mpisppy_l1_z = pe.Var(bounds=(0, None))
        subproblem._mpisppy_l1_obj = pe.Objective(expr=subproblem._mpisppy_l1_z)
        subproblem._mpisppy_l1_cons = pe.ConstraintList()
        z = subproblem._mpisppy_l1_z
        original_cons = [
            c for c in subproblem.component_data_objects(Constraint, active=True, descend_into=True)
        ]
        for c in original_cons:
            body = c.body
            if c.equality:
                rhs = pe.value(c.lower)
                subproblem._mpisppy_l1_cons.add(body - rhs <= z)
                subproblem._mpisppy_l1_cons.add(rhs - body <= z)
            else:
                if c.upper is not None:
                    subproblem._mpisppy_l1_cons.add(body - pe.value(c.upper) <= z)
                if c.lower is not None:
                    subproblem._mpisppy_l1_cons.add(pe.value(c.lower) - body <= z)
            c.deactivate()

    def _add_fixing_constraints(self, subproblem, complicating_vars_map):
        """Temporarily fix scenario copies of first-stage variables to root values.

        Args:
            subproblem (ConcreteModel): Scenario model receiving temporary
                equality constraints.
            complicating_vars_map (ComponentMap): Map from root variables to
                their scenario-copy variables on ``subproblem``.

        Returns:
            ComponentMap: Map from root variables to the temporary fixing
            constraints added for them.
        """
        subproblem._mpisppy_lshaped_fix_cons = pe.ConstraintList()
        fix_cons = pe.ComponentMap()
        for root_var in self.root_vars:
            if root_var not in complicating_vars_map:
                continue
            sub_var = complicating_vars_map[root_var]
            sub_var.set_value(root_var.value, skip_validation=True)
            con = subproblem._mpisppy_lshaped_fix_cons.add(sub_var - root_var.value == 0)
            fix_cons[root_var] = con
        return fix_cons

    def _remove_fixing_constraints(self, subproblem, solver, constraints):
        """Remove temporary fixing constraints from a subproblem and solver.

        Args:
            subproblem (ConcreteModel): Scenario model containing the temporary
                fixing ``ConstraintList``.
            solver (Solver): Solver object used for the subproblem. Persistent
                solvers also need the rows removed from the solver instance.
            constraints (iterable): Constraint data objects to remove from a
                persistent solver before deleting the component.
        """
        if isinstance(solver, PersistentSolver):
            for con in constraints:
                solver.remove_constraint(con)
        if hasattr(subproblem, "_mpisppy_lshaped_fix_cons"):
            subproblem.del_component(subproblem._mpisppy_lshaped_fix_cons)

    def _fixing_dual_coefficients(self, subproblem, fix_cons, sign):
        """Read fixing-constraint duals as root-variable cut coefficients.

        Args:
            subproblem (ConcreteModel): Solved scenario or L1 feasibility model
                with imported duals.
            fix_cons (ComponentMap): Map from root variables to fixing
                constraints.
            sign (int): Solver-specific multiplier applied to Pyomo dual values.

        Returns:
            numpy.ndarray: Coefficients ordered to match ``self.root_vars``.
        """
        coeffs = np.zeros(len(self.root_vars), dtype="d")
        for i, root_var in enumerate(self.root_vars):
            if root_var in fix_cons:
                coeffs[i] = sign * pe.value(subproblem.dual[fix_cons[root_var]])
        return coeffs

    def _solve_model(self, model, solver, solver_name, added_constraints):
        """Solve a model and load primal and dual values when optimal.

        Args:
            model (ConcreteModel): Pyomo model to solve for non-persistent
                solvers.
            solver (Solver): Pyomo solver object.
            solver_name (str): Solver interface name. Accepted for symmetry with
                related helpers; currently not inspected here.
            added_constraints (iterable): Constraints added after persistent
                solver instance setup and needing explicit solver registration.

        Returns:
            SolverResults: Pyomo solver results object.
        """
        allowed = self._optimal_tc | self._infeasible_tc
        if isinstance(solver, PersistentSolver):
            for con in added_constraints:
                solver.add_constraint(con)
            res = solver.solve(tee=False, load_solutions=False, save_results=False)
            if res.solver.termination_condition not in allowed:
                return res
            if res.solver.termination_condition in self._optimal_tc:
                solver.load_vars()
                solver.load_duals()
        else:
            res = solver.solve(model, tee=False, load_solutions=False)
            if res.solver.termination_condition in self._optimal_tc:
                model.solutions.load_from(res)
        return res

    def _make_solver(self, subproblem_solver, solver_options):
        """Create or normalize a solver object and apply solver options.

        Args:
            subproblem_solver (str or Solver): Solver interface name or existing
                Pyomo solver object.
            solver_options (dict, optional): Options assigned to the solver
                object's ``options`` mapping.

        Returns:
            tuple: ``(solver, solver_name)``.

        Raises:
            RuntimeError: If ``solver_name`` cannot be inferred.
        """
        if isinstance(subproblem_solver, str):
            solver_name = subproblem_solver
            solver = pe.SolverFactory(subproblem_solver)
        else:
            solver = subproblem_solver
            solver_name = getattr(solver, "name", None)
        if solver_name is None:
            raise RuntimeError("Could not determine subproblem solver name")
        if solver_options:
            for k, v in solver_options.items():
                solver.options[k] = v
        return solver, solver_name

    def _solver_sign(self, solver_name):
        """Return the dual sign multiplier for the supported Xpress interfaces.

        Args:
            solver_name (str): Pyomo solver interface name.

        Returns:
            int: Multiplier applied to imported fixing-constraint duals.

        Raises:
            NotImplementedError: If ``solver_name`` is not a supported Xpress
            interface.
        """
        if solver_name not in self._solver_dual_sign_convention:
            raise NotImplementedError(
                "standard_lp_l1 currently supports xpress, xpress_direct, "
                f"and xpress_persistent; got {solver_name}"
            )
        return self._solver_dual_sign_convention[solver_name]

    def _validate_linear_subproblem(self, subproblem):
        """Raise if the active objective or constraints are not linear.

        Args:
            subproblem (ConcreteModel): Scenario model to validate.

        Raises:
            ValueError: If the active objective or any active constraint body is
            quadratic or nonlinear.
        """
        obj = find_active_objective(subproblem)
        repn = generate_standard_repn(obj.expr, quadratic=True)
        if repn.nonlinear_vars or repn.quadratic_vars:
            raise ValueError("standard_lp_l1 requires a linear subproblem objective")
        for con in subproblem.component_data_objects(Constraint, active=True, descend_into=True):
            repn = generate_standard_repn(con.body, quadratic=True)
            if repn.nonlinear_vars or repn.quadratic_vars:
                raise ValueError("standard_lp_l1 requires linear subproblem constraints")

    def _convert_recourse_bounds_to_constraints(self, subproblem, complicating_vars_map):
        """Move recourse variable bounds into explicit rows for L1 relaxation.

        Args:
            subproblem (ConcreteModel): Scenario model whose recourse variable
                bounds should be converted in place.
            complicating_vars_map (ComponentMap): Map from root variables to
                their scenario-copy variables. Bounds on these complicating
                variables are left untouched.
        """
        complicating_vars = {id(v) for v in complicating_vars_map.values()}
        cons = subproblem._mpisppy_lshaped_bound_cons
        for var in subproblem.component_data_objects(Var, active=True, descend_into=True):
            if id(var) in complicating_vars:
                continue
            lb, ub = var.bounds
            if lb is not None:
                cons.add(var >= lb)
            if ub is not None:
                cons.add(var <= ub)
            if lb is not None or ub is not None:
                var.setlb(None)
                var.setub(None)


@declare_custom_block(name='LShapedCutGenerator')
class LShapedCutGeneratorData(bc.BendersCutGeneratorData):
    def __init__(self, component):
        super().__init__(component)
        # self.local_subproblem_count = 0
        # self.global_subproblem_count = 0

    def set_ls(self, ls):
        self.ls = ls
        self.global_subproblem_count = len(self.ls.all_scenario_names)
        self._subproblem_ndx_map = dict.fromkeys(range(len(self.ls.local_scenario_names)))
        for s in self._subproblem_ndx_map.keys():
            self._subproblem_ndx_map[s] = self.ls.all_scenario_names.index(self.ls.local_scenario_names[s])
        # print(self._subproblem_ndx_map)
        self.all_root_etas = list(self.ls.root.eta.values())

    def global_num_subproblems(self):
        return self.global_subproblem_count

    def add_subproblem(self, subproblem_fn, subproblem_fn_kwargs, root_eta, subproblem_solver='gurobi_persistent',
                       relax_subproblem_cons=False, subproblem_solver_options=None):
        # print(self._subproblem_ndx_map)
        # self.all_root_etas.append(root_eta)
        # self.global_subproblem_count += 1
        if subproblem_fn_kwargs['scenario_name'] in self.ls.local_scenario_names:
            # self.local_subproblem_count += 1
            self.root_etas.append(root_eta)
            subproblem, complicating_vars_map = subproblem_fn(**subproblem_fn_kwargs)
            self.subproblems.append(subproblem)
            self.complicating_vars_maps.append(complicating_vars_map)
            bc._setup_subproblem(subproblem, root_vars=[complicating_vars_map[i] for i in self.root_vars if
                                                       i in complicating_vars_map],
                              relax_subproblem_cons=relax_subproblem_cons)

            # self._subproblem_ndx_map[self.local_subproblem_count - 1] = self.global_subproblem_count - 1

            if isinstance(subproblem_solver, str):
                subproblem_solver = pe.SolverFactory(subproblem_solver)
            self.subproblem_solvers.append(subproblem_solver)
            if isinstance(subproblem_solver, PersistentSolver):
                set_instance_retry(subproblem, subproblem_solver, subproblem_fn_kwargs['scenario_name'])
            if subproblem_solver_options:
                for k,v in subproblem_solver_options.items():
                    subproblem_solver.options[k] = v
