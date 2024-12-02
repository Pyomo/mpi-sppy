###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
###############################################################################
# This file was originally part of parapint, available:
# https://github.com/sandialabs/parapint/ 
# See LICENSE.md in this directory for full copyright and license information.
###############################################################################

from pyomo.contrib.pynumero.interfaces import pyomo_nlp
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
import numpy as np
import scipy.sparse
from pyomo.common.timing import HierarchicalTimer

class InteriorPointInterface:
    def __init__(self, pyomo_model):
        self._nlp = pyomo_nlp.PyomoNLP(pyomo_model, nl_file_options={'skip_trivial_constraints': True})

        self.bounds_relaxation_factor = 0

        self._slacks = self.init_slacks()

        # set the init_duals_primals_lb/ub from ipopt_zL_out, ipopt_zU_out if available
        # need to compress them as well and initialize the duals_primals_lb/ub 
        self._init_duals_primals_lb, self._init_duals_primals_ub =\
            self._get_full_duals_primals_bounds()
        self._init_duals_primals_lb[np.isneginf(self.primals_lb())] = 0
        self._init_duals_primals_ub[np.isinf(self.primals_ub())] = 0
        self._duals_primals_lb = self._init_duals_primals_lb.copy()
        self._duals_primals_ub = self._init_duals_primals_ub.copy()

        # set the init_duals_slacks_lb/ub from the init_duals_ineq
        # need to be compressed and set according to their sign
        # (-) value indicates it the upper is active, while (+) indicates
        # that lower is active
        self._init_duals_slacks_lb = self._nlp.init_duals_ineq().copy()
        self._init_duals_slacks_lb[self._init_duals_slacks_lb < 0] = 0
        self._init_duals_slacks_ub = self._nlp.init_duals_ineq().copy()
        self._init_duals_slacks_ub[self._init_duals_slacks_ub > 0] = 0
        self._init_duals_slacks_ub *= -1.0

        self._duals_slacks_lb = self._init_duals_slacks_lb.copy()
        self._duals_slacks_ub = self._init_duals_slacks_ub.copy()

        self._delta_primals = None
        self._delta_slacks = None
        self._delta_duals_eq = None
        self._delta_duals_ineq = None
        self._barrier = None

    def get_bounds_relaxation_factor(self) -> float:
        return self.bounds_relaxation_factor

    def set_bounds_relaxation_factor(self, val: float):
        self.bounds_relaxation_factor = val

    def n_primals(self):
        return self._nlp.n_primals()

    def nnz_hessian_lag(self):
        return self._nlp.nnz_hessian_lag()

    def set_obj_factor(self, obj_factor):
        self._nlp.set_obj_factor(obj_factor)

    def get_obj_factor(self):
        return self._nlp.get_obj_factor()

    def n_eq_constraints(self):
        return self._nlp.n_eq_constraints()

    def n_ineq_constraints(self):
        return self._nlp.n_ineq_constraints()

    def nnz_jacobian_eq(self):
        return self._nlp.nnz_jacobian_eq()

    def nnz_jacobian_ineq(self):
        return self._nlp.nnz_jacobian_ineq()

    def init_primals(self):
        primals = self._nlp.init_primals()
        return primals

    def init_slacks(self):
        slacks = self._nlp.evaluate_ineq_constraints()
        return slacks

    def init_duals_eq(self):
        return self._nlp.init_duals_eq()

    def init_duals_ineq(self):
        return self._nlp.init_duals_ineq()

    def init_duals_primals_lb(self):
        return self._init_duals_primals_lb

    def init_duals_primals_ub(self):
        return self._init_duals_primals_ub

    def init_duals_slacks_lb(self):
        return self._init_duals_slacks_lb

    def init_duals_slacks_ub(self):
        return self._init_duals_slacks_ub

    def set_primals(self, primals):
        self._nlp.set_primals(primals)

    def set_slacks(self, slacks):
        self._slacks = slacks

    def set_duals_eq(self, duals):
        self._nlp.set_duals_eq(duals)

    def set_duals_ineq(self, duals):
        self._nlp.set_duals_ineq(duals)

    def set_duals_primals_lb(self, duals):
        self._duals_primals_lb = duals

    def set_duals_primals_ub(self, duals):
        self._duals_primals_ub = duals

    def set_duals_slacks_lb(self, duals):
        self._duals_slacks_lb = duals

    def set_duals_slacks_ub(self, duals):
        self._duals_slacks_ub = duals

    def get_primals(self):
        return self._nlp.get_primals()

    def get_slacks(self):
        return self._slacks

    def get_duals_eq(self):
        return self._nlp.get_duals_eq()

    def get_duals_ineq(self):
        return self._nlp.get_duals_ineq()

    def get_duals_primals_lb(self):
        return self._duals_primals_lb

    def get_duals_primals_ub(self):
        return self._duals_primals_ub

    def get_duals_slacks_lb(self):
        return self._duals_slacks_lb

    def get_duals_slacks_ub(self):
        return self._duals_slacks_ub

    def primals_lb(self):
        lbs = self._nlp.primals_lb()
        if self.bounds_relaxation_factor == 0:
            return lbs
        eye = np.ones(lbs.size)
        lbs_mod = lbs - self.bounds_relaxation_factor * np.max(np.array([eye, np.abs(lbs)]), axis=0)
        return lbs_mod

    def primals_ub(self):
        ubs = self._nlp.primals_ub()
        if self.bounds_relaxation_factor == 0:
            return ubs
        eye = np.ones(ubs.size)
        ubs_mod = ubs + self.bounds_relaxation_factor * np.max(np.array([eye, np.abs(ubs)]), axis=0)
        return ubs_mod

    def ineq_lb(self):
        lbs = self._nlp.ineq_lb()
        if self.bounds_relaxation_factor == 0:
            return lbs
        eye = np.ones(lbs.size)
        lbs_mod = lbs - self.bounds_relaxation_factor * np.max(np.array([eye, np.abs(lbs)]), axis=0)
        return lbs_mod

    def ineq_ub(self):
        ubs = self._nlp.ineq_ub()
        if self.bounds_relaxation_factor == 0:
            return ubs
        eye = np.ones(ubs.size)
        ubs_mod = ubs + self.bounds_relaxation_factor * np.max(np.array([eye, np.abs(ubs)]), axis=0)
        return ubs_mod

    def set_barrier_parameter(self, barrier):
        self._barrier = barrier

    def pyomo_nlp(self):
        return self._nlp

    def evaluate_primal_dual_kkt_matrix(self, timer=None):
        if timer is None:
            timer = HierarchicalTimer()
        timer.start('eval hess')
        hess_block = self._nlp.evaluate_hessian_lag()
        timer.stop('eval hess')
        timer.start('eval jac')
        jac_eq = self._nlp.evaluate_jacobian_eq()
        jac_ineq = self._nlp.evaluate_jacobian_ineq()
        timer.stop('eval jac')

        duals_primals_lb = self._duals_primals_lb
        duals_primals_ub = self._duals_primals_ub
        duals_slacks_lb = self._duals_slacks_lb
        duals_slacks_ub = self._duals_slacks_ub
        primals = self._nlp.get_primals()

        timer.start('hess block')
        data = (duals_primals_lb/(primals - self.primals_lb()) +
                duals_primals_ub/(self.primals_ub() - primals))
        n = self._nlp.n_primals()
        indices = np.arange(n)
        hess_block.row = np.concatenate([hess_block.row, indices])
        hess_block.col = np.concatenate([hess_block.col, indices])
        hess_block.data = np.concatenate([hess_block.data, data])
        timer.stop('hess block')

        timer.start('slack block')
        data = (duals_slacks_lb/(self._slacks - self.ineq_lb()) +
                duals_slacks_ub/(self.ineq_ub() - self._slacks))
        n = self._nlp.n_ineq_constraints()
        indices = np.arange(n)
        slack_block = scipy.sparse.coo_matrix((data, (indices, indices)), shape=(n, n))
        timer.stop('slack block')

        timer.start('regularization block')
        eq_reg_blk = scipy.sparse.identity(self._nlp.n_eq_constraints(), format='coo')
        eq_reg_blk.data.fill(0)
        ineq_reg_blk = scipy.sparse.identity(self._nlp.n_ineq_constraints(), format='coo')
        ineq_reg_blk.data.fill(0)
        timer.stop('regularization block')

        timer.start('set block')
        kkt = BlockMatrix(4, 4)
        kkt.set_block(0, 0, hess_block)
        kkt.set_block(1, 1, slack_block)
        kkt.set_block(2, 0, jac_eq)
        kkt.set_block(0, 2, jac_eq.transpose())
        kkt.set_block(3, 0, jac_ineq)
        kkt.set_block(0, 3, jac_ineq.transpose())
        kkt.set_block(3, 1, -scipy.sparse.identity(
                                            self._nlp.n_ineq_constraints(),
                                            format='coo'))
        kkt.set_block(1, 3, -scipy.sparse.identity(
                                            self._nlp.n_ineq_constraints(),
                                            format='coo'))
        kkt.set_block(2, 2, eq_reg_blk)
        kkt.set_block(3, 3, ineq_reg_blk)
        timer.stop('set block')
        return kkt

    def evaluate_primal_dual_kkt_rhs(self, timer=None):
        if timer is None:
            timer = HierarchicalTimer()
        timer.start('eval grad obj')
        grad_obj = self.get_obj_factor() * self.evaluate_grad_objective()
        timer.stop('eval grad obj')
        timer.start('eval jac')
        jac_eq = self._nlp.evaluate_jacobian_eq()
        jac_ineq = self._nlp.evaluate_jacobian_ineq()
        timer.stop('eval jac')
        timer.start('eval cons')
        eq_resid = self._nlp.evaluate_eq_constraints()
        ineq_resid = self._nlp.evaluate_ineq_constraints() - self._slacks
        timer.stop('eval cons')

        timer.start('grad_lag_primals')
        grad_lag_primals = (grad_obj +
                            jac_eq.transpose() * self._nlp.get_duals_eq() +
                            jac_ineq.transpose() * self._nlp.get_duals_ineq() -
                            self._barrier / (self._nlp.get_primals() - self.primals_lb()) +
                            self._barrier / (self.primals_ub() - self._nlp.get_primals()))
        timer.stop('grad_lag_primals')

        timer.start('grad_lag_slacks')
        grad_lag_slacks = (-self._nlp.get_duals_ineq() -
                           self._barrier / (self._slacks - self.ineq_lb()) +
                           self._barrier / (self.ineq_ub() - self._slacks))
        timer.stop('grad_lag_slacks')

        rhs = BlockVector(4)
        rhs.set_block(0, grad_lag_primals)
        rhs.set_block(1, grad_lag_slacks)
        rhs.set_block(2, eq_resid)
        rhs.set_block(3, ineq_resid)
        rhs = -rhs
        return rhs

    def set_primal_dual_kkt_solution(self, sol):
        self._delta_primals = sol.get_block(0)
        self._delta_slacks = sol.get_block(1)
        self._delta_duals_eq = sol.get_block(2)
        self._delta_duals_ineq = sol.get_block(3)

    def get_delta_primals(self):
        return self._delta_primals

    def get_delta_slacks(self):
        return self._delta_slacks

    def get_delta_duals_eq(self):
        return self._delta_duals_eq

    def get_delta_duals_ineq(self):
        return self._delta_duals_ineq

    def get_delta_duals_primals_lb(self):
        res = (((self._barrier - self._duals_primals_lb * self._delta_primals) /
                (self._nlp.get_primals() - self.primals_lb())) -
               self._duals_primals_lb)
        return res

    def get_delta_duals_primals_ub(self):
        res = (((self._barrier + self._duals_primals_ub * self._delta_primals) /
                (self.primals_ub() - self._nlp.get_primals())) -
               self._duals_primals_ub)
        return res

    def get_delta_duals_slacks_lb(self):
        res = (((self._barrier - self._duals_slacks_lb * self._delta_slacks) /
                (self._slacks - self.ineq_lb())) -
               self._duals_slacks_lb)
        return res

    def get_delta_duals_slacks_ub(self):
        res = (((self._barrier + self._duals_slacks_ub * self._delta_slacks) /
                (self.ineq_ub() - self._slacks)) -
               self._duals_slacks_ub)
        return res

    def evaluate_objective(self):
        return self._nlp.evaluate_objective()

    def evaluate_eq_constraints(self):
        return self._nlp.evaluate_eq_constraints()

    def evaluate_ineq_constraints(self):
        return self._nlp.evaluate_ineq_constraints()

    def evaluate_grad_objective(self):
        return self._nlp.evaluate_grad_objective()

    def evaluate_jacobian_eq(self):
        return self._nlp.evaluate_jacobian_eq()

    def evaluate_jacobian_ineq(self):
        return self._nlp.evaluate_jacobian_ineq()

    def regularize_equality_gradient(self, kkt, coef, copy_kkt=True):
        # Not technically regularizing the equality gradient ...
        # Replace this with a regularize_diagonal_block function?
        # Then call with kkt matrix and the value of the perturbation?

        # Use a constant perturbation to regularize the equality constraint
        # gradient
        if copy_kkt:
            kkt = kkt.copy()
        reg_coef = coef
        eq_ptb = (reg_coef *
                  scipy.sparse.identity(self._nlp.n_eq_constraints(),
                                        format='coo'))
        ineq_ptb = (reg_coef *
                    scipy.sparse.identity(self._nlp.n_ineq_constraints(),
                                          format='coo'))

        kkt.set_block(2, 2, eq_ptb)
        kkt.set_block(3, 3, ineq_ptb)
        return kkt

    def regularize_hessian(self, kkt, coef, copy_kkt=True):
        if copy_kkt:
            kkt = kkt.copy()

        hess = kkt.get_block(0, 0)
        ptb = coef * scipy.sparse.identity(self._nlp.n_primals(), format='coo')
        hess += ptb
        kkt.set_block(0, 0, hess)
        return kkt

    def _get_full_duals_primals_bounds(self):
        full_duals_primals_lb = None
        full_duals_primals_ub = None
        # Check in case _nlp was constructed as an AmplNLP (from an nl file)
        if (hasattr(self._nlp, 'pyomo_model') and 
            hasattr(self._nlp, 'get_pyomo_variables')):
            pyomo_model = self._nlp.pyomo_model()
            pyomo_variables = self._nlp.get_pyomo_variables()
            if hasattr(pyomo_model,'ipopt_zL_out'):
                zL_suffix = pyomo_model.ipopt_zL_out 
                full_duals_primals_lb = np.empty(self._nlp.n_primals())
                for i,v in enumerate(pyomo_variables):
                    if v in zL_suffix:
                        full_duals_primals_lb[i] = zL_suffix[v]

            if hasattr(pyomo_model,'ipopt_zU_out'):
                zU_suffix = pyomo_model.ipopt_zU_out 
                full_duals_primals_ub = np.empty(self._nlp.n_primals())
                for i,v in enumerate(pyomo_variables):
                    if v in zU_suffix:
                        full_duals_primals_ub[i] = zU_suffix[v]

        if full_duals_primals_lb is None:
            full_duals_primals_lb = np.ones(self._nlp.n_primals())

        if full_duals_primals_ub is None:
            full_duals_primals_ub = np.ones(self._nlp.n_primals())

        return full_duals_primals_lb, full_duals_primals_ub

    def load_primals_into_pyomo_model(self):
        if not isinstance(self._nlp, pyomo_nlp.PyomoNLP):
            raise RuntimeError('Can only load primals into a pyomo model if a pyomo model was used in the constructor.')

        pyomo_variables = self._nlp.get_pyomo_variables()
        primals = self._nlp.get_primals()
        for i, v in enumerate(pyomo_variables):
            v.value = primals[i]

    def pyomo_model(self):
        return self._nlp.pyomo_model()

    def get_pyomo_variables(self):
        return self._nlp.get_pyomo_variables()

    def get_pyomo_constraints(self):
        return self._nlp.get_pyomo_constraints()

    def variable_names(self):
        return self._nlp.variable_names()

    def constraint_names(self):
        return self._nlp.constraint_names()

    def get_primal_indices(self, pyomo_variables):
        return self._nlp.get_primal_indices(pyomo_variables)

    def get_constraint_indices(self, pyomo_constraints):
        return self._nlp.get_constraint_indices(pyomo_constraints)
