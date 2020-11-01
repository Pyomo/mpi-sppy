# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' Implementation of the Frank-Wolfe Progressive Hedging (FW-PH) algorithm
    described in the paper:

    N. Boland et al., "Combining Progressive Hedging with a Frank-Wolfe method
    to compute Lagrangian dual bounds in stochastic mixed-integer programming".
    SIAM J. Optim. 28(2):1312--1336, 2018.

    Current implementation supports parallelism and bundling.

    Does not support:
         1. The use of PH_extensions
         2. The solution of models with more than two stages
         3. Simultaneous use of bundling and user-specified initial points

    The FWPH algorithm allows the user to specify an initial set of points
    (notated V^0_s in Boland) to approximate the convex hull of each scenario
    subproblem. Points are specified via a callable which takes as input a
    scenario name (str) and outputs a list of dicts. Each dict corresponds to a
    point, and is formatted

        point_dict[variable_name] = variable_value

    (e.g. point_dict['Allocation[2,5]'] = 1.7). The order of this list
    matters--the first point in the list is used to compute the initial dual
    weights and xBar value (in Boland notation, the first point in the list is
    the point (x0_s,y0_s) in Algorithm 3). The callable is passed to FWPH as an
    option in the FW_options dict, with the key "point_creator". The
    point_creator callable may also take an optional argument,
    "point_creator_data", which may be any data type, and contain any
    information needed to create the initial point set. The point_creator_data
    is passed to FWPH as an option in the FW_options dict, with the key
    "point_creator_data".

    See fwph_sslp.py for an example of user-specified point creation.
'''

import mpisppy.phbase
import mpisppy.utils.sputils as sputils
import numpy as np
import pyomo.environ as pyo
import time
import re # For manipulating scenario names

from mpi4py import MPI
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.pysp.phutils import find_active_objective
from pyomo.core.expr.visitor import replace_expressions
from pyomo.core.expr.numeric_expr import LinearExpression

class FWPH(mpisppy.phbase.PHBase):
    
    def __init__(
        self,
        PH_options,
        FW_options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement=None,
        all_nodenames=None,
        mpicomm=None,
        rank0=0,
        cb_data=None,
        PH_converger=None,
        rho_setter=None,
    ):
        super().__init__(
            PH_options, 
            all_scenario_names,
            scenario_creator,
            scenario_denouement,
            all_nodenames=all_nodenames,
            mpicomm=mpicomm,
            rank0=rank0,
            cb_data=cb_data,
            PH_extensions=None,
            PH_extension_kwargs=None,
            PH_converger=PH_converger,
            rho_setter=rho_setter,
        )

        self._init(FW_options)

    def _init(self, FW_options):
        self.FW_options = FW_options
        self._options_checks_fw()
        self.vb = True
        if ('FW_verbose' in self.FW_options):
            self.vb = self.FW_options['FW_verbose']

    def fw_prep(self):
        self.PH_Prep(attach_duals=True, attach_prox=False)
        self._check_for_multistage()
        self.subproblem_creation(self.PHoptions['verbose'])
        self._output_header()

        if ('point_creator' in self.FW_options):
            # The user cannot both specify and point_creator and use bundles.
            # At this point, we have already checked for that possibility, so
            # we can safely use the point_creator without further checks for
            # bundles.
            self._create_initial_points()
            check = True if 'check_initial_points' not in self.FW_options \
                         else self.FW_options['check_initial_points']
            if (check):
                self._check_initial_points()
            self._create_solvers()
            self._use_rho_setter(verbose and self.rank==self.rank0)
            self._initialize_MIP_var_values()
            best_bound = -np.inf if self.is_minimizing else np.inf
        else:
            trivial_bound = self.Iter0()
            secs = time.time() - self.t0
            self._output(0, trivial_bound, trivial_bound, np.nan, secs)
            best_bound = trivial_bound

        if ('mip_solver_options' in self.FW_options):
            self._set_MIP_solver_options()

        # Lines 2 and 3 of Algorithm 3 in Boland
        self.Compute_Xbar(self.PHoptions['verbose'], None)
        self.Update_W(self.PHoptions['verbose'])

        # Necessary pre-processing steps
        # We disable_W so they don't appear
        # in the MIP objective when _set_QP_objective
        # snarfs it for the QP
        self._disable_W()
        self._attach_MIP_vars()
        self._initialize_QP_subproblems()
        self._attach_indices()
        self._attach_MIP_QP_maps()
        self._set_QP_objective()
        self._initialize_QP_var_values()
        self._swap_nonant_vars()
        self._reenable_W()

        if (self.PH_converger):
            self.convobject = self.PH_converger(self, self.rank, self.n_proc)

        return best_bound

    def fwph_main(self):
        self.t0 = time.time()
        best_bound = self.fw_prep()

        # FWPH takes some time to initialize
        # If run as a spoke, check for convergence here
        if self.spcomm and self.spcomm.is_converged():
            return None, None, None

        # The body of the algorithm
        for itr in range(self.PHoptions['PHIterLimit']):
            self._PHIter = itr
            self._local_bound = 0
            for name in self.local_subproblems:
                dual_bound = self.SDM(name)
                self._local_bound += self.local_subproblems[name].PySP_prob * \
                                     dual_bound
            self._compute_dual_bound()
            if (self.is_minimizing):
                best_bound = np.maximum(best_bound, self._local_bound)
            else:
                best_bound = np.minimum(best_bound, self._local_bound)

            ## Hubs/spokes take precedence over convergers
            if self.spcomm:
                if self.spcomm.is_converged():
                    secs = time.time() - self.t0
                    self._output(itr+1, self._local_bound, 
                                 best_bound, np.nan, secs)
                    if (self.rank == self.rank0 and self.vb):
                        print('FWPH converged to user-specified criteria')
                    break
                self.spcomm.sync()
            if (self.PH_converger):
                self.Compute_Xbar(self.PHoptions['verbose'], None)
                diff = self.convobject.convergence_value()
                if (self.convobject.is_converged()):
                    secs = time.time() - self.t0
                    self._output(itr+1, self._local_bound, 
                                 best_bound, diff, secs)
                    if (self.rank == self.rank0 and self.vb):
                        print('FWPH converged to user-specified criteria')
                    break
            else: # Convergence check from Boland
                diff = self._conv_diff()
                self.Compute_Xbar(self.PHoptions['verbose'], None)
                if (diff < self.PHoptions['convthresh']):
                    secs = time.time() - self.t0
                    self._output(itr+1, self._local_bound, 
                                 best_bound, diff, secs)
                    if (self.rank == self.rank0 and self.vb):
                        print('PH converged based on standard criteria')
                    break

            secs = time.time() - self.t0
            self._output(itr+1, self._local_bound, best_bound, diff, secs)
            self.Update_W(self.PHoptions['verbose'])
            timed_out = self._is_timed_out()
            if (self._is_timed_out()):
                if (self.rank == self.rank0 and self.vb):
                    print('Timeout.')
                break

        self._swap_nonant_vars_back()
        weight_dict = self._gather_weight_dict() # None if rank != 0
        xbars_dict  = self._get_xbars() # None if rank != 0
        return itr+1, weight_dict, xbars_dict

    def SDM(self, model_name):
        '''  Algorithm 2 in Boland et al. (with small tweaks)
        '''
        mip = self.local_subproblems[model_name]
        qp  = self.local_QP_subproblems[model_name]
    
        # Set the QP dual weights to the correct values If we are bundling, we
        # initialize the QP dual weights to be the dual weights associated with
        # the first scenario in the bundle (this should be okay, because each
        # scenario in the bundle has the same dual weights, analytically--maybe
        # a numerical problem).
        arb_scen_mip = self.local_scenarios[mip.scen_list[0]] \
                       if self.bundling else mip
        for (node_name, ix) in arb_scen_mip._nonant_indexes:
            qp._Ws[node_name, ix]._value = \
                arb_scen_mip._Ws[node_name, ix].value

        alpha = self.FW_options['FW_weight']
        # Algorithm 3 line 6
        xt = {ndn_i:
            (1 - alpha) * pyo.value(arb_scen_mip._xbars[ndn_i])
            + alpha * pyo.value(xvar)
            for ndn_i, xvar in arb_scen_mip._nonant_indexes.items()
            }

        for itr in range(self.FW_options['FW_iter_limit']):
            # Algorithm 2 line 4
            mip_source = mip.scen_list if self.bundling else [model_name]
            for scenario_name in mip_source:
                scen_mip = self.local_scenarios[scenario_name]
                for ndn_i, nonant in scen_mip._nonant_indexes.items():
                    x_source = xt[ndn_i] if itr==0 \
                               else nonant._value
                    scen_mip._Ws[ndn_i]._value = (
                        qp._Ws[ndn_i]._value
                        + scen_mip._PHrho[ndn_i]._value
                        * (x_source
                        -  scen_mip._xbars[ndn_i]._value))

            # Algorithm 2 line 5
            if (sputils.is_persistent(mip._solver_plugin)):
                mip_obj = find_active_objective(mip, True)
                mip._solver_plugin.set_objective(mip_obj)
            mip_results = mip._solver_plugin.solve(mip)
            self._check_solve(mip_results, model_name + ' (MIP)')

            # Algorithm 2 lines 6--8
            obj = find_active_objective(mip, True)
            if (itr == 0):
                if (self.is_minimizing):
                    dual_bound = mip_results.Problem[0].Lower_bound
                else:
                    dual_bound = mip_results.Problem[0].Upper_bound

            # Algorithm 2 line 9 (compute \Gamma^t)
            val0 = pyo.value(obj)
            new  = replace_expressions(obj.expr, mip.mip_to_qp)
            val1 = pyo.value(new)
            obj.expr = replace_expressions(new, qp.qp_to_mip)
            if abs(val0) > 1e-9:
                stop_check = (val1 - val0) / abs(val0) # \Gamma^t in Boland, but normalized
            else:
                stop_check = val1 - val0 # \Gamma^t in Boland
            stop_check_tol = self.FW_options["stop_check_tol"]\
                             if "stop_check_tol" in self.FW_options else 1e-4
            if (self.is_minimizing and stop_check < -stop_check_tol):
                print('Warning (fwph): convergence quantity Gamma^t = '
                     '{sc:.2e} (should be non-negative)'.format(sc=stop_check))
                print('Try decreasing the MIP gap tolerance and re-solving')
            elif (not self.is_minimizing and stop_check > stop_check_tol):
                print('Warning (fwph): convergence quantity Gamma^t = '
                     '{sc:.2e} (should be non-positive)'.format(sc=stop_check))
                print('Try decreasing the MIP gap tolerance and re-solving')

            self._add_QP_column(model_name)
            if (sputils.is_persistent(qp._QP_solver_plugin)):
                qp_obj = find_active_objective(qp, True)
                qp._QP_solver_plugin.set_objective(qp_obj)
            qp_results = qp._QP_solver_plugin.solve(qp)
            self._check_solve(qp_results, model_name + ' (QP)')

            if (stop_check < self.FW_options['FW_conv_thresh']):
                break

        # Re-set the mip._Ws so that the QP objective 
        # is correct in the next major iteration
        mip_source = mip.scen_list if self.bundling else [model_name]
        for scenario_name in mip_source:
            scen_mip = self.local_scenarios[scenario_name]
            for (node_name, ix) in scen_mip._nonant_indexes:
                scen_mip._Ws[node_name, ix]._value = \
                    qp._Ws[node_name, ix]._value

        return dual_bound

    def _add_QP_column(self, model_name):
        ''' Add a column to the QP, with values taken from the most recent MIP
            solve.
        '''
        mip = self.local_subproblems[model_name]
        qp  = self.local_QP_subproblems[model_name]
        solver = qp._QP_solver_plugin
        persistent = sputils.is_persistent(solver)

        if hasattr(solver, 'add_column'):
            new_var = qp.a.add()
            coef_list = [1.]
            constr_list = [qp.sum_one]
            target = mip.ref_vars if self.bundling else mip.nonant_vars
            for (node, ix) in qp.eqx.index_set():
                coef_list.append(target[node, ix].value)
                constr_list.append(qp.eqx[node, ix])
            for key in mip.y_indices:
                coef_list.append(mip.leaf_vars[key].value)
                constr_list.append(qp.eqy[key])
            solver.add_column(qp, new_var, 0, constr_list, coef_list)
            return

        # Add new variable and update \sum a_i = 1 constraint
        new_var = qp.a.add() # Add the new convex comb. variable
        if (persistent):
            solver.add_var(new_var)
            solver.remove_constraint(qp.sum_one)
            qp.sum_one._body += new_var
            solver.add_constraint(qp.sum_one)
        else:
            qp.sum_one._body += new_var

        target = mip.ref_vars if self.bundling else mip.nonant_vars
        for (node, ix) in qp.eqx.index_set():
            if (persistent):
                solver.remove_constraint(qp.eqx[node, ix])
                qp.eqx[node, ix]._body += new_var * target[node, ix].value
                solver.add_constraint(qp.eqx[node, ix])
            else:
                qp.eqx[node, ix]._body += new_var * target[node,ix].value
        for key in mip.y_indices:
            if (persistent):
                solver.remove_constraint(qp.eqy[key])
                qp.eqy[key]._body += new_var * pyo.value(mip.leaf_vars[key])
                solver.add_constraint(qp.eqy[key])
            else:
                qp.eqy[key]._body += new_var * pyo.value(mip.leaf_vars[key])

    def _attach_indices(self):
        ''' Attach the fields x_indices and y_indices to the model objects in
            self.local_subproblems (not self.local_scenarios, nor
            self.local_QP_subproblems).

            x_indices is a list of tuples of the form...
                (scenario name, node name, variable index) <-- bundling
                (node name, variable index)                <-- no bundling
            y_indices is a list of tuples of the form...
                (scenario_name, "LEAF", variable index)    <-- bundling
                ("LEAF", variable index)                   <-- no bundling
            
            Must be called after the subproblems (MIPs AND QPs) are created.
        '''
        for name in self.local_subproblems.keys():
            mip = self.local_subproblems[name]
            qp  = self.local_QP_subproblems[name]
            if (self.bundling):
                x_indices = [(scenario_name, node_name, ix)
                    for scenario_name in mip.scen_list
                    for (node_name, ix) in 
                        self.local_scenarios[scenario_name]._nonant_indexes]
                y_indices = [(scenario_name, 'LEAF', ix)
                    for scenario_name in mip.scen_list
                    for ix in range(mip.num_leaf_vars[scenario_name])]
            else:
                x_indices = mip._nonant_indexes.keys()
                y_indices = [('LEAF', ix) for ix in range(len(qp.y))]

            y_indices = pyo.Set(initialize=y_indices)
            y_indices.construct()
            x_indices = pyo.Set(initialize=x_indices)
            x_indices.construct()
            mip.x_indices = x_indices
            mip.y_indices = y_indices

    def _attach_MIP_QP_maps(self):
        ''' Create dictionaries that map MIP variable ids to their QP
            counterparts, and vice versa.
        '''
        for name in self.local_subproblems.keys():
            mip = self.local_subproblems[name]
            qp  = self.local_QP_subproblems[name]

            mip.mip_to_qp = {id(mip.nonant_vars[key]): qp.x[key]
                                for key in mip.x_indices}
            mip.mip_to_qp.update({id(mip.leaf_vars[key]): qp.y[key]
                                for key in mip.y_indices})
            qp.qp_to_mip = {id(qp.x[key]): mip.nonant_vars[key]
                                for key in mip.x_indices}
            qp.qp_to_mip.update({id(qp.y[key]): mip.leaf_vars[key]
                                for key in mip.y_indices})

    def _attach_MIP_vars(self):
        ''' Create a list indexed (node_name, ix) for all the MIP
            non-anticipative and leaf variables, so that they can be easily
            accessed when adding columns to the QP.
        '''
        if (self.bundling):
            for (bundle_name, EF) in self.local_subproblems.items():
                EF.nonant_vars = dict()
                EF.leaf_vars   = dict()
                EF.num_leaf_vars = dict() # Keys are scenario names
                for scenario_name in EF.scen_list:
                    mip = self.local_scenarios[scenario_name]
                    # Non-anticipative variables
                    nonant_dict = {(scenario_name, ndn, ix): nonant
                        for (ndn,ix), nonant in mip._nonant_indexes.items()}
                    EF.nonant_vars.update(nonant_dict)
                    # Leaf variables
                    leaf_vars = self._get_leaf_vars(mip)
                    leaf_var_dict = {(scenario_name, 'LEAF', ix): 
                        leaf_vars[ix] for ix in range(len(leaf_vars))}
                    EF.leaf_vars.update(leaf_var_dict)
                    EF.num_leaf_vars[scenario_name] = len(leaf_vars)
                    # Reference variables are already attached: EF.ref_vars
                    # indexed by (node_name, index)
        else:
            for (name, mip) in self.local_scenarios.items():
                mip.nonant_vars = mip._nonant_indexes
                leaf_vars = self._get_leaf_vars(mip)
                mip.leaf_vars = { ('LEAF', ix): 
                    leaf_vars[ix] for ix in range(len(leaf_vars))
                }

    def _check_for_multistage(self):
        if self.multistage:
            raise RuntimeError('The FWPH algorithm only supports '
                               'two-stage models at this time.')

    def _check_initial_points(self):
        ''' If t_max (i.e. the inner iteration limit) is set to 1, then the
            initial point set must satisfy the additional condition (17) in
            Boland et al. This function verifies that condition (17) is
            satisfied by solving a linear program (similar to the Phase I
            auxiliary LP in two-phase simplex).

            This function is only called by a single rank, which MUST be rank
            0. The rank 0 check happens before this function is called.
        '''
        # Need to get the first-stage variable names (as supplied by the user)
        # by picking them off of any random scenario that's laying around.
        arb_scenario = list(self.local_scenarios.keys())[0]
        arb_mip = self.local_scenarios[arb_scenario]
        root = arb_mip._PySPnode_list[0]
        stage_one_var_names = [var.name for var in root.nonant_vardata_list]

        init_pts = self.comms['ROOT'].gather(self.local_initial_points, root=0)
        if (self.rank != self.rank0):
            return

        print('Checking initial points...', end='', flush=True)
        
        points = {key: value for block in init_pts 
                             for (key, value) in block.items()}
        scenario_names = points.keys()
        num_scenarios = len(points)

        # Some index sets we will need..
        conv_ix = [(scenario_name, var_name) 
                    for scenario_name in scenario_names
                    for var_name in stage_one_var_names]
        conv_coeff = [(scenario_name, ix) 
                    for scenario_name in scenario_names
                    for ix in range(len(points[scenario_name]))]

        aux = pyo.ConcreteModel()
        aux.x = pyo.Var(stage_one_var_names, within=pyo.Reals)
        aux.slack_plus = pyo.Var(conv_ix, within=pyo.NonNegativeReals)
        aux.slack_minus = pyo.Var(conv_ix, within=pyo.NonNegativeReals)
        aux.conv = pyo.Var(conv_coeff, within=pyo.NonNegativeReals)

        def sum_one_rule(model, scenario_name):
            return pyo.quicksum(model.conv[scenario_name,ix] \
                for ix in range(len(points[scenario_name]))) == 1
        aux.sum_one = pyo.Constraint(scenario_names, rule=sum_one_rule)

        def conv_rule(model, scenario_name, var_name):
            return model.x[var_name] \
                + model.slack_plus[scenario_name, var_name] \
                - model.slack_minus[scenario_name, var_name] \
                == pyo.quicksum(model.conv[scenario_name, ix] *
                    points[scenario_name][ix][var_name] 
                    for ix in range(len(points[scenario_name])))
        aux.comb = pyo.Constraint(conv_ix, rule=conv_rule)

        obj_expr = pyo.quicksum(aux.slack_plus.values()) \
                   + pyo.quicksum(aux.slack_minus.values())
        aux.obj = pyo.Objective(expr=obj_expr, sense=pyo.minimize)

        solver = pyo.SolverFactory(self.FW_options['solvername'])
        results = solver.solve(aux)
        self._check_solve(results, 'Auxiliary LP')

        check_tol = self.FW_options['check_tol'] \
                        if 'check_tol' in self.FW_options.keys() else 1e-4
        if (pyo.value(obj_expr) > check_tol):
            print('error.')
            raise ValueError('The specified initial points do not satisfy the '
                'critera necessary for convergence. Please specify different '
                'initial points, or increase FW_iter_limit')
        print('done.')

    def _check_solve(self, results, model_name):
        ''' Verify that the solver solved to optimality '''
        if (results.solver.status != pyo.SolverStatus.ok) or \
            (results.solver.termination_condition != pyo.TerminationCondition.optimal):
            print('Solve failed on model', model_name)
            print('Solver status:', results.solver.status)
            print('Termination conditions:', results.solver.termination_condition)
            raise RuntimeError()

    def _compute_dual_bound(self):
        ''' Compute the FWPH dual bound using self._local_bound from each rank
        '''
        send = np.array(self._local_bound)
        recv = np.array(0.)
        self.comms['ROOT'].Allreduce(
            [send, MPI.DOUBLE], [recv, MPI.DOUBLE], op=MPI.SUM)
        self._local_bound = recv

    def _conv_diff(self):
        ''' Perform the convergence check of Algorithm 3 in Boland et al. '''
        diff = 0.
        for name in self.local_subproblems.keys():
            mip = self.local_subproblems[name]
            arb_mip = self.local_scenarios[mip.scen_list[0]] \
                        if self.bundling else mip
            qp  = self.local_QP_subproblems[name]
            diff_s = 0.
            for (node_name, ix) in arb_mip._nonant_indexes:
                qpx = qp.xr if self.bundling else qp.x
                diff_s += np.power(pyo.value(qpx[node_name,ix]) - 
                        pyo.value(arb_mip._xbars[node_name,ix]), 2)
            diff_s *= mip.PySP_prob
            diff += diff_s
        diff = np.array(diff)
        recv = np.array(0.)
        self.comms['ROOT'].Allreduce(
            [diff, MPI.DOUBLE], [recv, MPI.DOUBLE], op=MPI.SUM)
        return recv

    def _create_initial_points(self):
        pc = self.FW_options['point_creator']
        if ('point_creator_data' in self.FW_options.keys()):
            pd = self.FW_options['point_creator_data']
        else:
            pd = None
        self.local_initial_points = dict()
        for scenario_name in self.local_scenario_names:
            pts = pc(scenario_name, point_creator_data=pd)
            self.local_initial_points[scenario_name] = pts

    def _extract_objective(self, mip):
        ''' Extract the original part of the provided MIP's objective function
            (no dual or prox terms), and create a copy containing the QP
            variables in place of the MIP variables.

            Args:
                mip (Pyomo ConcreteModel): MIP model for a scenario or bundle.

            Returns:
                obj (Pyomo Objective): objective function extracted
                    from the MIP
                new (Pyomo Expression): expression from the MIP model
                    objective with the MIP variables replaced by QP variables.
                    Does not inculde dual or prox terms.

            Notes:
                Acts on either a single-scenario model or a bundle
        '''
        mip_to_qp = mip.mip_to_qp
        obj = find_active_objective(mip, True)
        repn = generate_standard_repn(obj.expr, quadratic=True)
        if len(repn.nonlinear_vars) > 0:
            raise ValueError("FWPH does not support models with nonlinear objective functions")
        linear_vars = [mip_to_qp[id(var)] for var in repn.linear_vars]
        new = LinearExpression(
            constant=repn.constant, linear_coefs=repn.linear_coefs, linear_vars=linear_vars
        )
        if repn.quadratic_vars:
            quadratic_vars = (
                (mip_to_qp[id(x)], mip_to_qp[id(y)]) for x,y in repn.quadratic_vars
            )
            new += pyo.quicksum(
                (coef*x*y for coef,(x,y) in zip(repn.quadratic_coefs, quadratic_vars))
            )
        return obj, new

    def _gather_weight_dict(self, strip_bundle_names=False):
        ''' Compute a double nested dictionary of the form

                weights[scenario name][variable name] = weight value

            for FWPH to return to the user.
        
            Notes:
                Must be called after the variables are swapped back.
        '''
        local_weights = dict()
        for (name, scenario) in self.local_scenarios.items():
            if (self.bundling and strip_bundle_names):
                scenario_weights = dict()
                for ndn_ix, var in scenario._nonant_indexes.items():
                    rexp = '^' + scenario.name + '\.'
                    var_name = re.sub(rexp, '', var.name)
                    scenario_weights[var_name] = \
                                        scenario._Ws[ndn_ix].value
            else:
                scenario_weights = {nonant.name: scenario._Ws[ndn_ix].value
                        for ndn_ix, nonant in scenario._nonant_indexes.items()}
            local_weights[name] = scenario_weights

        weights = self.comms['ROOT'].gather(local_weights, root=0)
        return weights

    def _get_leaf_vars(self, scenario):
        ''' This method simply needs to take an input scenario
            (pyo.ConcreteModel) and return a list of variable objects
            corresponding to the leaf node variables for that scenario.

            Functions by returning the complement of the set of
            non-anticipative variables.
        '''
        nonant_var_ids = [id(var) for node in scenario._PySPnode_list
                                  for var  in node.nonant_vardata_list]
        return [var for var in scenario.component_data_objects(pyo.Var)
                         if id(var) not in nonant_var_ids]

    def _get_xbars(self, strip_bundle_names=False):
        ''' Return the xbar vector if rank = 0 and None, otherwise
            (Consistent with _gather_weight_dict).

            Args:
                TODO
                
            Notes:
                Paralellism is not necessary since each rank already has an
                identical copy of xbar, provided by Compute_Xbar().

                Returned dictionary is indexed by variable name 
                (as provided by the user).

                Must be called after variables are swapped back (I think).
        '''
        if (self.rank != self.rank0):
            return None
        else:
            random_scenario_name = list(self.local_scenarios.keys())[0]
            scenario = self.local_scenarios[random_scenario_name]
            xbar_dict = {}
            for node in scenario._PySPnode_list:
                for (ix, var) in enumerate(node.nonant_vardata_list):
                    var_name = var.name
                    if (self.bundling and strip_bundle_names):
                        rexp = '^' + random_scenario_name + '\.'
                        var_name = re.sub(rexp, '', var_name)
                    xbar_dict[var_name] = scenario._xbars[node.name, ix].value
            return xbar_dict

    def _initialize_MIP_var_values(self):
        ''' Initialize the MIP variable values to the user-specified point.

            For now, arbitrarily choose the first point in each list.
        '''
        points = self.local_initial_points
        for (name, mip) in self.local_subproblems.items():
            pt = points[name][0] # Select the first point arbitrarily
            mip_vars = list(mip.component_data_objects(pyo.Var))
            for var in mip_vars:
                try:
                    var.set_value(pt[var.name])
                except KeyError as e:
                    raise KeyError('Found variable named' + var.name +
                        ' in model ' + name + ' not contained in the '
                        'specified initial point dictionary') from e

    def _initialize_QP_subproblems(self):
        ''' Instantiates the (convex) QP subproblems (eqn. (13) in the Boland
            paper) for each scenario. Does not create/attach an objective.

            Attachs a local_QP_subproblems dict to self. Keys are scenario
            names (or bundle names), values are Pyomo ConcreteModel objects
            corresponding to the QP subproblems. 

            QP subproblems are in their original form, without the x and y
            variables eliminated. Rationale: pre-solve will get this, easier
            bookkeeping (objective does not need to be changed at each inner
            iteration this way).
        '''
        self.local_QP_subproblems = dict()
        has_init_pts = hasattr(self, 'local_initial_points')
        for (name, model) in self.local_subproblems.items():
            if (self.bundling):
                xr_indices = model.ref_vars.keys()
                nonant_indices = model.nonant_vars.keys()
                leaf_indices = model.leaf_vars.keys()
                if (has_init_pts):
                    raise RuntimeError('Cannot currently specify '
                        'initial points while using bundles')
            else:
                nonant_indices = model._nonant_indexes.keys()
                leaf_indices = model.leaf_vars.keys()

            ''' Convex comb. coefficients '''
            QP = pyo.ConcreteModel()
            QP.a = pyo.VarList(domain=pyo.NonNegativeReals)
            if (has_init_pts):
                for _ in range(len(self.local_initial_points[name])):
                    QP.a.add()
            else:
                QP.a.add() # Just one variable (1-based index!) to start

            ''' Other variables '''
            QP.x = pyo.Var(nonant_indices, within=pyo.Reals)
            QP.y = pyo.Var(leaf_indices, within=pyo.Reals)
            if (self.bundling):
                QP.xr = pyo.Var(xr_indices, within=pyo.Reals)

            ''' Non-anticipativity constraint '''
            if (self.bundling):
                def nonant_rule(m, scenario_name, node_name, ix):
                    return m.x[scenario_name, node_name, ix] == \
                            m.xr[node_name, ix]
                QP.na = pyo.Constraint(nonant_indices, rule=nonant_rule)
            
            ''' (x,y) constraints '''
            if (self.bundling):
                def x_rule(m, node_name, ix):
                    return -m.xr[node_name, ix] + m.a[1] * \
                            model.ref_vars[node_name, ix].value == 0
                def y_rule(m, scenario_name, node_name, ix):
                    return -m.y[scenario_name, node_name, ix] + m.a[1]\
                        * model.leaf_vars[scenario_name,node_name,ix].value == 0 
                QP.eqx = pyo.Constraint(xr_indices, rule=x_rule)
            else:
                if (has_init_pts):
                    pts = self.local_initial_points[name]
                    def x_rule(m, node_name, ix):
                        nm = model.nonant_vars[node_name, ix].name
                        return -m.x[node_name, ix] + \
                            pyo.quicksum(m.a[i+1] * pts[i][nm] 
                                for i in range(len(pts))) == 0
                    def y_rule(m, node_name, ix):
                        nm = model.leaf_vars[node_name, ix].name
                        return -m.y[node_name,ix] + \
                            pyo.quicksum(m.a[i+1] * pts[i][nm] 
                                for i in range(len(pts))) == 0
                else:
                    def x_rule(m, node_name, ix):
                        return -m.x[node_name, ix] + m.a[1] * \
                                model.nonant_vars[node_name, ix].value == 0
                    def y_rule(m, node_name, ix):
                        return -m.y[node_name,ix] + m.a[1] * \
                                model.leaf_vars['LEAF', ix].value == 0
                QP.eqx = pyo.Constraint(nonant_indices, rule=x_rule)

            QP.eqy = pyo.Constraint(leaf_indices, rule=y_rule)
            QP.sum_one = pyo.Constraint(expr=pyo.quicksum(QP.a.values())==1)

            self.local_QP_subproblems[name] = QP
                
    def _initialize_QP_var_values(self):
        ''' Set the value of the QP variables to be equal to the values of the
            corresponding MIP variables.

            Notes:
                Must be called before _swap_nonant_vars()

                Must be called after _initialize_MIP_var_values(), if the user
                specifies initial sets of points. Otherwise, it must be called
                after Iter0().
        '''
        for name in self.local_subproblems.keys():
            mip = self.local_subproblems[name]
            qp  = self.local_QP_subproblems[name]

            for key in mip.x_indices:
                qp.x[key].set_value(mip.nonant_vars[key].value)
            for key in mip.y_indices:
                qp.y[key].set_value(mip.leaf_vars[key].value)

            # Set the non-anticipative reference variables if we're bundling
            if (self.bundling):
                arb_scenario = mip.scen_list[0]
                naix = self.local_scenarios[arb_scenario]._nonant_indexes
                for (node_name, ix) in naix:
                    # Check that non-anticipativity is satisfied
                    # within the bundle (for debugging)
                    vals = [mip.nonant_vars[scenario_name, node_name, ix].value
                            for scenario_name in mip.scen_list]
                    assert(max(vals) - min(vals) < 1e-7)
                    qp.xr[node_name, ix].set_value(
                        mip.nonant_vars[arb_scenario, node_name, ix].value)

    def _is_timed_out(self):
        if (self.rank == self.rank0):
            time_elapsed = time.time() - self.t0
            status = 1 if (time_elapsed > self.FW_options['time_limit']) \
                       else 0
        else:
            status = None
        status = self.comms['ROOT'].bcast(status, root=0)
        return status != 0

    def _options_checks_fw(self):
        ''' Name                Boland notation (Algorithm 2)
            -------------------------------------------------
            FW_iter_limit       t_max
            FW_weight           alpha
            FW_conv_thresh      tau
        '''
        # 1. Check for required options
        reqd_options = ['FW_iter_limit', 'FW_weight', 'FW_conv_thresh',
                        'solvername']
        losers = [opt for opt in reqd_options if opt not in self.FW_options]
        if (len(losers) > 0):
            msg = "FW_options is missing the following key(s): " + \
                  ", ".join(losers)
            raise RuntimeError(msg)

        # 2. Check that bundles, pre-specified points and t_max play nice. This
        #    is only checked on rank 0, because that is where the initial
        #    points are supposed to be specified.
        use_bundles = ('bundles_per_rank' in self.PHoptions 
                        and self.PHoptions['bundles_per_rank'] > 0)
        t_max = self.FW_options['FW_iter_limit']
        specd_init_pts = 'point_creator' in self.FW_options.keys() and \
                         self.FW_options['point_creator'] is not None

        if (use_bundles and specd_init_pts):
            if (t_max == 1):
                raise RuntimeError('Cannot use bundles and specify initial '
                    'points with t_max=1 at the same time.')
            else:
                if (self.rank == self.rank0):
                    print('WARNING: Cannot specify initial points and use '
                        'bundles at the same time. Ignoring specified initial '
                        'points')
                # Remove specified initial points
                self.FW_options.pop('point_creator', None)

        if (t_max == 1 and not specd_init_pts):
            raise RuntimeError('FW_iter_limit set to 1. To ensure '
                'convergence, provide initial points, or increase '
                'FW_iter_limit')

        # 3. Check that the user did not specify the linearization of binary
        #    proximal terms (no binary variables allowed in FWPH QPs)
        if ('linearize_binary_proximal_terms' in self.PHoptions
            and self.PHoptions['linearize_binary_proximal_terms']):
            print('Warning: linearize_binary_proximal_terms cannot be used '
                  'with the FWPH algorithm. Ignoring...')
            self.PHoptions['linearize_binary_proximal_terms'] = False

        # 4. Provide a time limit of inf if the user did not specify
        if ('time_limit' not in self.FW_options.keys()):
            self.FW_options['time_limit'] = np.inf

    def _output(self, itr, bound, best_bound, diff, secs):
        if (self.rank == self.rank0 and self.vb):
            print('{itr:3d} {bound:12.4f} {best_bound:12.4f} {diff:12.4e} {secs:11.1f}s'.format(
                    itr=itr, bound=bound, best_bound=best_bound, 
                    diff=diff, secs=secs))
        if (self.rank == self.rank0 and 'save_file' in self.FW_options.keys()):
            fname = self.FW_options['save_file']
            with open(fname, 'a') as f:
                f.write('{itr:d},{bound:.16f},{best_bound:.16f},{diff:.16f},{secs:.16f}\n'.format(
                    itr=itr, bound=bound, best_bound=best_bound,
                    diff=diff, secs=secs))

    def _output_header(self):
        if (self.rank == self.rank0 and self.vb):
            print('itr {bound:>12s} {bb:>12s} {cd:>12s} {tm:>12s}'.format(
                    bound="bound", bb="best bound", cd="conv diff", tm="time"))
        if (self.rank == self.rank0 and 'save_file' in self.FW_options.keys()):
            fname = self.FW_options['save_file']
            with open(fname, 'a') as f:
                f.write('{itr:s},{bound:s},{bb:s},{diff:s},{secs:s}\n'.format(
                    itr="Iteration", bound="Bound", bb="Best bound",
                    diff="Error", secs="Time(s)"))

    def save_weights(self, fname):
        ''' Save the computed weights to the specified file.

            Notes:
                Handles parallelism--only writes one copy of the file.

                Rather "fast-and-loose", in that it doesn't enforce _when_ this
                function can be called.
        '''
        weights = self._gather_weight_dict(strip_bundle_names=self.bundling) # None if rank != 0
        if (self.rank != self.rank0):
            return
        with open(fname, 'w') as f:
            for block in weights:
                for (scenario_name, wts) in block.items():
                    for (var_name, weight_val) in wts.items():
                        row = '{sn},{vn},{wv:.16f}\n'.format(
                            sn=scenario_name, vn=var_name, wv=weight_val)
                        f.write(row)

    def save_xbars(self, fname):
        ''' Save the computed xbar to the specified file.

            Notes:
                Handles parallelism--only writes one copy of the file.

                Rather "fast-and-loose", in that it doesn't enforce _when_ this
                function can be called.
        '''
        if (self.rank != self.rank0):
            return
        xbars = self._get_xbars(strip_bundle_names=self.bundling) # None if rank != 0
        with open(fname, 'w') as f:
            for (var_name, xbs) in xbars.items():
                row = '{vn},{vv:.16f}\n'.format(vn=var_name, vv=xbs)
                f.write(row)

    def _set_MIP_solver_options(self):
        mip_opts = self.FW_options['mip_solver_options']
        if (len(mip_opts) > 0):
            for model in self.local_subproblems.values():
                for (key, option) in mip_opts.items():
                    model._solver_plugin.options[key] = option

    def _set_QP_objective(self):
        ''' Attach dual weights, objective function and solver to each QP.
        
            QP dual weights are initialized to the MIP dual weights.
        '''

        for name, mip in self.local_subproblems.items():
            QP = self.local_QP_subproblems[name]

            obj, new = self._extract_objective(mip)

            ## Finish setting up objective for QP
            if self.bundling:
                m_source = self.local_scenarios[mip.scen_list[0]]
                x_source = QP.xr
            else:
                m_source = mip
                x_source = QP.x

            QP._Ws = pyo.Param(
                m_source._nonant_indexes.keys(), mutable=True, initialize=m_source._Ws
            )
            # rhos are attached to each scenario, not each bundle (should they be?)
            ph_term = pyo.quicksum((
                QP._Ws[nni] * x_source[nni] +
                (m_source._PHrho[nni] / 2.) * (x_source[nni] - m_source._xbars[nni]) * (x_source[nni] - m_source._xbars[nni])
                for nni in m_source._nonant_indexes
            ))

            if obj.is_minimizing():
                QP.obj = pyo.Objective(expr=new+ph_term, sense=pyo.minimize)
            else:
                QP.obj = pyo.Objective(expr=-new+ph_term, sense=pyo.minimize)

            ''' Attach a solver with various options '''
            solver = pyo.SolverFactory(self.FW_options['solvername'])
            if sputils.is_persistent(solver):
                solver.set_instance(QP)
            if 'qp_solver_options' in self.FW_options:
                qp_opts = self.FW_options['qp_solver_options']
                if qp_opts:
                    for (key, option) in qp_opts.items():
                        solver.options[key] = option

            self.local_QP_subproblems[name]._QP_solver_plugin = solver

    def _swap_nonant_vars(self):
        ''' Change the pointers in
            scenario._PySPnode_list[i].nonant_vardata_list
            to point to the QP variables, rather than the MIP variables.

            Notes:
                When computing xBar and updating the weights in the outer
                iteration, the values of the x variables are pulled from
                scenario._PySPnode_list[i].nonant_vardata_list. In the FWPH
                algorithm, xBar should be computed using the QP values, not the
                MIP values (like in normal PH).

                Reruns SPBase.attach_nonant_indexes so that the scenario 
                _nonant_indexes dictionary has the correct variable pointers
                
                Updates nonant_vardata_list but NOT nonant_list.
        '''
        for (name, model) in self.local_subproblems.items():
            scens = model.scen_list if self.bundling else [name]
            for scenario_name in scens:
                scenario = self.local_scenarios[scenario_name]
                num_nonant_vars = scenario._PySP_nlens
                node_list = scenario._PySPnode_list
                for node in node_list:
                    node.nonant_vardata_list = [
                        self.local_QP_subproblems[name].xr[node.name,i]
                        if self.bundling else
                        self.local_QP_subproblems[name].x[node.name,i]
                        for i in range(num_nonant_vars[node.name])]
        self.attach_nonant_indexes()

    def _swap_nonant_vars_back(self):
        ''' Swap variables back, in case they're needed somewhere else.
        '''
        for (name, model) in self.local_subproblems.items():
            if (self.bundling):
                EF = self.local_subproblems[name]
                for scenario_name in EF.scen_list:
                    scenario = self.local_scenarios[scenario_name]
                    num_nonant_vars = scenario._PySP_nlens
                    for node in scenario._PySPnode_list:
                        node.nonant_vardata_list = [
                            EF.nonant_vars[scenario_name,node.name,ix]
                            for ix in range(num_nonant_vars[node.name])]
            else:
                scenario = self.local_scenarios[name]
                num_nonant_vars = scenario._PySP_nlens
                for node in scenario._PySPnode_list:
                    node.nonant_vardata_list = [
                        scenario.nonant_vars[node.name,ix]
                        for ix in range(num_nonant_vars[node.name])]
        self.attach_nonant_indexes()

if __name__=='__main__':
    print('fwph.py has no main()')
