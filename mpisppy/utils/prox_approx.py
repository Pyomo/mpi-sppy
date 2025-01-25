###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from math import isclose
from array import array
from bisect import bisect
from scipy.optimize import least_squares
from pyomo.core.expr.numeric_expr import LinearExpression

# helpers for distance from y = x**2
def _f_ls(val, x_pnt, y_pnt):
    return ( val[0] - x_pnt ,  val[0]**2 - y_pnt )
def _df_ls(val, x_pnt, y_pnt):
    return ( (val[0],), (2*val[0],) )

class ProxApproxManager:
    __slots__ = ()

    def __new__(cls, mpisppy_model, xvar, ndn_i):
        if xvar.is_integer():
            return ProxApproxManagerDiscrete(mpisppy_model, xvar, ndn_i)
        else:
            return ProxApproxManagerContinuous(mpisppy_model, xvar, ndn_i)

class _ProxApproxManager:
    '''
    A helper class to manage proximal approximations
    '''
    __slots__ = ()

    def __init__(self, scenario, xvar, ndn_i):
        self.xvar = xvar
        self.xvarsqrd = scenario._mpisppy_model.xsqvar[ndn_i]
        self.cuts = scenario._mpisppy_model.xsqvar_cuts
        self.xbar = scenario._mpisppy_model.xbars[ndn_i]
        self.rho = scenario._mpisppy_model.rho[ndn_i]
        self.W = scenario._mpisppy_model.W[ndn_i]
        self.c = scenario._mpisppy_data.nonant_cost_coeffs[ndn_i]
        self.var_index = ndn_i
        self.cut_index = 0
        self.cut_values = array("d")
        self.cut_values.append(0.0)
        self._store_bounds()

    def _store_bounds(self):
        if self.xvar.lb is None:
            self.lb = -float("inf")
        else:
            self.lb = self.xvar.lb
        if self.xvar.ub is None:
            self.ub = float("inf")
        else:
            self.ub = self.xvar.ub

    def add_cut(self, val, tolerance, persistent_solver):
        '''
        create a cut at val
        '''
        pass

    def check_and_add_value(self, val, tolerance):
        idx = bisect(self.cut_values, val)
        # self.cut_values is empty, has one element
        # or we're appending to the end
        if idx == len(self.cut_values):
            if val - self.cut_values[idx-1] < tolerance:
                return False
            else:
                self.cut_values.insert(idx, val)
                return True
        # here we're at the beginning
        if idx == 0:
            if self.cut_values[idx] - val < tolerance:
                return False
            else:
                self.cut_values.insert(idx, val)
                return True
        # in the middle
        if self.cut_values[idx] - val < tolerance:
            return False
        if val - self.cut_values[idx-1] < tolerance:
            return False
        self.cut_values.insert(idx, val)
        return True

    def add_initial_cuts_var(self, x_val, other_bound, sign, tolerance, persistent_solver):
        if other_bound is not None:
            min_approx_error_pnt = sign * (2.0 / 3.0) * (other_bound - x_val)
        else:
            min_approx_error_pnt = None
        if self.c > 0:
            cost_pnt = self.c / self.rho.value
        elif self.c < 0:
            cost_pnt = -self.c / self.rho.value
        else: # c == 0
            cost_pnt = None
        num_cuts = 0
        if cost_pnt is None and min_approx_error_pnt is None:
            # take a wild guess
            step = max(1, tolerance)
            num_cuts += self.add_cut(x_val + sign*step, tolerance, persistent_solver)
            num_cuts += self.add_cut(x_val + sign*10*step, tolerance, persistent_solver)
            return num_cuts
        # no upper bound, or weak upper bound compared to cost_pnt
        if (min_approx_error_pnt is None) or (cost_pnt is not None and min_approx_error_pnt > 10*cost_pnt):
            step = max(cost_pnt, tolerance)
            num_cuts += self.add_cut(x_val + sign*step, tolerance, persistent_solver)
            num_cuts += self.add_cut(x_val + sign*10*step, tolerance, persistent_solver)
            return num_cuts
        # no objective, or the objective is "large" compared to min_approx_error_pnt
        if (cost_pnt is None) or (cost_pnt >= min_approx_error_pnt / 2.0):
            step = max(min_approx_error_pnt, tolerance)
            num_cuts += self.add_cut(x_val + step, tolerance, persistent_solver)
            # guard against horrible bounds
            if min_approx_error_pnt / 2.0 > max(1, tolerance):
                step = max(1, tolerance)
                num_cuts += self.add_cut(x_val + sign*step, tolerance, persistent_solver)
            else:
                step = max(min_approx_error_pnt/2.0, tolerance)
                num_cuts += self.add_cut(x_val + sign*step, tolerance, persistent_solver)
            return num_cuts
        # cost_pnt and min_approx_error_pnt are *not* None
        # and (cost_pnt < min_approx_error_pnt / 2.0 <= min_approx_error_pnt)
        step = max(cost_pnt, tolerance)
        num_cuts += self.add_cut(x_val + sign*step, tolerance, persistent_solver)
        step = max(min_approx_error_pnt, tolerance)
        num_cuts += self.add_cut(x_val + sign*step, tolerance, persistent_solver)
        return num_cuts


    def add_initial_cuts_var_right(self, x_val, bounds, tolerance, persistent_solver):
        return self.add_initial_cuts_var(x_val, bounds[1], 1, tolerance, persistent_solver)

    def add_initial_cuts_var_left(self, x_val, bounds, tolerance, persistent_solver):
        return self.add_initial_cuts_var(x_val, bounds[0], -1, tolerance, persistent_solver)

    def add_initial_cuts(self, x_val, tolerance, persistent_solver):
        num_cuts = 0
        bounds = self.xvar.bounds
        # no lower bound **or** sufficiently inside lower bound
        if bounds[0] is None or (x_val > bounds[0] + 3*tolerance):
            num_cuts += self.add_initial_cuts_var_left(x_val, bounds, tolerance, persistent_solver)
        # no upper bound **or** sufficiently inside upper bound
        if bounds[1] is None or (x_val < bounds[1] - 3*tolerance):
            num_cuts += self.add_initial_cuts_var_right(x_val, bounds, tolerance, persistent_solver)
        return num_cuts

    def add_cuts(self, x_val, tolerance, persistent_solver):
        x_bar = self.xbar.value
        rho = self.rho.value
        W = self.W.value

        num_cuts = self.add_cut(x_val, tolerance, persistent_solver)
        # rotate x_val around x_bar, the minimizer of (\rho / 2)(x - x_bar)^2
        # to create a vertex at this point
        rotated_x_val_x_bar = 2*x_bar - x_val
        if not isclose(x_val, rotated_x_val_x_bar, abs_tol=tolerance):
            num_cuts += self.add_cut(rotated_x_val_x_bar, tolerance, persistent_solver)
        # aug_lagrange_point, is the minimizer of w\cdot x + (\rho / 2)(x - x_bar)^2
        # to create a vertex at this point
        aug_lagrange_point = -W / rho + x_bar
        if not isclose(x_val, aug_lagrange_point, abs_tol=tolerance):
            num_cuts += self.add_cut(2*aug_lagrange_point - x_val, tolerance, persistent_solver)
        # finally, create another vertex at the aug_lagrange_point by rotating
        # rotated_x_val_x_bar around the aug_lagrange_point
        if not isclose(rotated_x_val_x_bar, aug_lagrange_point, abs_tol=tolerance):
            num_cuts += self.add_cut(2*aug_lagrange_point - rotated_x_val_x_bar, tolerance, persistent_solver)
        # If we only added 0 or 1 cuts initially, add up to two more
        # to capture something of the proximal term. This can happen
        # when x_bar == x_val and W == 0.
        if self.cut_index <= 1:
            num_cuts += self.add_initial_cuts(x_val, tolerance, persistent_solver)
        # print(f"{x_val=}, {x_bar=}, {W=}")
        # print(f"{self.cut_values=}")
        # print(f"{self.cut_index=}")
        return num_cuts

    def check_tol_add_cut(self, tolerance, persistent_solver=None):
        '''
        add a cut if the tolerance is not satified
        '''
        if self.xvar.fixed:
            # don't do anything for fixed variables
            return 0
        x_pnt = self.xvar.value
        y_pnt = self.xvarsqrd.value
        # f_val = x_pnt**2

        # print(f"{x_pnt=}, {y_pnt=}, {f_val=}")
        # print(f"y-distance: {actual_val - measured_val})")
        if y_pnt is None:
            y_pnt = 0.0
        # We project the point x_pnt, y_pnt onto
        # the curve y = x**2 by finding the minimum distance
        # between y = x**2 and x_pnt, y_pnt.
        res = least_squares(_f_ls, x_pnt, args=(x_pnt, y_pnt), jac=_df_ls, method="lm", x_scale="jac")
        if not res.success:
            raise RuntimeError(f"Error in projecting {(x_pnt, y_pnt)} onto parabola for "
                               f"proximal approximation. Message: {res.message}")
        return self.add_cuts(res.x[0], tolerance,  persistent_solver)

class ProxApproxManagerContinuous(_ProxApproxManager):

    def add_cut(self, val, tolerance, persistent_solver):
        '''
        create a cut at val using a taylor approximation
        '''
        lb, ub = self.xvar.bounds
        if lb is not None and val < lb:
            val = lb
        if ub is not None and val > ub:
            val = ub
        if not self.check_and_add_value(val, tolerance):
            return 0
        # f'(a) = 2*val
        # f(a) - f'(a)a = val*val - 2*val*val
        f_p_a = 2*val
        const = -(val*val)

        ## f(x) >= f(a) + f'(a)(x - a)
        ## f(x) >= f'(a) x + (f(a) - f'(a)a)
        ## (0 , f(x) - f'(a) x - (f(a) - f'(a)a) , None)
        expr = LinearExpression( linear_coefs=[1, -f_p_a],
                                 linear_vars=[self.xvarsqrd, self.xvar],
                                 constant=-const )
        self.cuts[self.var_index, self.cut_index] = (0, expr, None)
        if persistent_solver is not None:
            persistent_solver.add_constraint(self.cuts[self.var_index, self.cut_index])
        self.cut_index += 1
        #print(f"added continuous cut for {self.xvar.name} at {val}, lb: {self.xvar.lb}, ub: {self.xvar.ub}")

        return 1

def _compute_mb(val):
    ## [(n+1)^2 - n^2] = 2n+1
    ## [(n+1) - n] = 1
    ## -> m = 2n+1
    m = 2*val+1

    ## b = n^2 - (2n+1)*n
    ## = -n^2 - n
    ## = -n (n+1)
    b = -val*(val+1)
    return m,b

class ProxApproxManagerDiscrete(_ProxApproxManager):

    def add_cut(self, val, tolerance, persistent_solver):
        '''
        create up to two cuts at val, exploiting integrality
        '''
        val = int(round(val))
        if tolerance > 1:
            # TODO: We should consider how to handle this, maybe.
            #       Tolerances less than or equal 1 won't affect
            #       discrete cuts
            pass

        ## cuts are indexed by the x-value to the right
        ## e.g., the cut for (2,3) is indexed by 3
        ##       the cut for (-2,-1) is indexed by -1
        cuts_added = 0

        ## So, a cut to the RIGHT of the point 3 is the cut for (3,4),
        ## which is indexed by 4
        if (*self.var_index, val+1) not in self.cuts and val < self.ub:
            m,b = _compute_mb(val)
            expr = LinearExpression( linear_coefs=[1, -m],
                                     linear_vars=[self.xvarsqrd, self.xvar],
                                     constant=-b )
            #print(f"adding cut for {(val, val+1)}")
            self.cuts[self.var_index, val+1] = (0, expr, None)
            if persistent_solver is not None:
                persistent_solver.add_constraint(self.cuts[self.var_index, val+1])
            cuts_added += 1

        ## Similarly, a cut to the LEFT of the point 3 is the cut for (2,3),
        ## which is indexed by 3
        if (*self.var_index, val) not in self.cuts and val > self.lb:
            m,b = _compute_mb(val-1)
            expr = LinearExpression( linear_coefs=[1, -m],
                                     linear_vars=[self.xvarsqrd, self.xvar],
                                     constant=-b )
            #print(f"adding cut for {(val-1, val)}")
            self.cuts[self.var_index, val] = (0, expr, None)
            if persistent_solver is not None:
                persistent_solver.add_constraint(self.cuts[self.var_index, val])
            cuts_added += 1
        #print(f"added {cuts_added} integer cut(s) for {self.xvar.name} at {val}, lb: {self.xvar.lb}, ub: {self.xvar.ub}")

        return cuts_added

if __name__ == '__main__':
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    bounds = (-100, 100)
    m.x = pyo.Var(bounds = bounds)
    #m.x = pyo.Var(within=pyo.Integers, bounds = bounds)
    m.xsqvar = pyo.Var(within=pyo.NonNegativeReals)

    m.xbars = pyo.Param(initialize=-73.2, mutable=True)
    m.W = pyo.Param(initialize=0, mutable=True)
    m.rho = pyo.Param(initialize=1, mutable=True)
    ## ( x - zero )^2 = x^2 - 2 x zero + zero^2
    m.obj = pyo.Objective( expr =m.rho*(m.xsqvar - 2*m.xbars*m.x + m.xbars**2 ))

    m.xsqvar_cuts = pyo.Constraint(pyo.Any, pyo.Integers)

    s = pyo.SolverFactory('xpress_persistent')
    prox_manager = ProxApproxManager(m, m.x, None)
    s.set_instance(m)
    m.pprint()
    new_cuts = True
    iter_cnt = 0
    while new_cuts:
        s.solve(m,tee=False)
        print(f"x: {pyo.value(m.x):.2e}, obj: {pyo.value(m.obj):.2e}")
        new_cuts = prox_manager.check_tol_add_cut(1e-1, persistent_solver=s)
        #m.pprint()
        iter_cnt += 1

    print(f"cuts: {len(m.xsqvar_cuts)}, iters: {iter_cnt}")
