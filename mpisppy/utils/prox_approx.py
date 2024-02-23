# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

from math import isclose
from pyomo.environ import value
from pyomo.core.expr.numeric_expr import LinearExpression

# helpers for distance from y = x**2
def _f(val, x_pnt, y_pnt):
    return (( val - x_pnt )**2 + ( val**2 - y_pnt )**2)/2.
def _df(val, x_pnt, y_pnt):
    #return 2*(val - x_pnt) + 4*(val**2 - y_pnt)*val
    return val*(1 - 2*y_pnt + 2*val*val) - x_pnt
def _d2f(val, x_pnt, y_pnt):
    return 1 + 6*val*val - 2*y_pnt

def _newton_step(val, x_pnt, y_pnt):
    return val - _df(val, x_pnt, y_pnt) / _d2f(val, x_pnt, y_pnt)

class ProxApproxManager:
    __slots__ = ()

    def __new__(cls, xvar, xvarsqrd, xbar, xsqvar_cuts, ndn_i):
        if xvar.is_integer():
            return ProxApproxManagerDiscrete(xvar, xvarsqrd, xbar, xsqvar_cuts, ndn_i)
        else:
            return ProxApproxManagerContinuous(xvar, xvarsqrd, xbar, xsqvar_cuts, ndn_i)

class _ProxApproxManager:
    '''
    A helper class to manage proximal approximations
    '''
    __slots__ = ()

    def __init__(self, xvar, xvarsqrd, xbar, xsqvar_cuts, ndn_i):
        self.xvar = xvar
        self.xvarsqrd = xvarsqrd
        self.xbar = xbar
        self.var_index = ndn_i
        self.cuts = xsqvar_cuts
        self.cut_index = 0
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

    def add_cut(self, val, persistent_solver=None):
        '''
        create a cut at val
        '''
        pass

    def check_tol_add_cut(self, tolerance, persistent_solver=None):
        '''
        add a cut if the tolerance is not satified
        '''
        x_pnt = self.xvar.value
        x_bar = self.xbar.value
        y_pnt = self.xvarsqrd.value
        f_val = x_pnt**2

        #print(f"y-distance: {actual_val - measured_val})")
        if y_pnt is None:
            self.add_cut(x_pnt, persistent_solver)
            if not isclose(x_pnt, x_bar, abs_tol=1e-6):
                self.add_cut(2*x_bar - x_pnt, persistent_solver)
            return True

        if (f_val - y_pnt) > tolerance:
            '''
            In this case, we project the point x_pnt, y_pnt onto
            the curve y = x**2 by finding the minimum distance
            between y = x**2 and x_pnt, y_pnt.

            This involves solving a cubic equation, so instead
            we start at x_pnt, y_pnt and run newtons algorithm
            to get an approximate good-enough solution.
            '''
            this_val = x_pnt
            #print(f"initial distance: {_f(this_val, x_pnt, y_pnt)**(0.5)}")
            #print(f"this_val: {this_val}")
            next_val = _newton_step(this_val, x_pnt, y_pnt)
            while not isclose(this_val, next_val, rel_tol=1e-6, abs_tol=1e-6):
                #print(f"newton step distance: {_f(next_val, x_pnt, y_pnt)**(0.5)}")
                #print(f"next_val: {next_val}")
                this_val = next_val
                next_val = _newton_step(this_val, x_pnt, y_pnt)
            self.add_cut(next_val, persistent_solver)
            if not isclose(next_val, x_bar, abs_tol=1e-6):
                self.add_cut(2*x_bar - next_val, persistent_solver)
            return True
        return False

class ProxApproxManagerContinuous(_ProxApproxManager):

    def add_cut(self, val, persistent_solver=None):
        '''
        create a cut at val using a taylor approximation
        '''
        # handled by bound
        if val == 0:
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

    def add_cut(self, val, persistent_solver=None):
        '''
        create up to two cuts at val, exploiting integrality
        '''
        val = int(round(val))

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
    m.xsqrd = pyo.Var(within=pyo.NonNegativeReals)

    m.zero = pyo.Param(initialize=-73.2, mutable=True)
    ## ( x - zero )^2 = x^2 - 2 x zero + zero^2
    m.obj = pyo.Objective( expr = m.xsqrd - 2*m.zero*m.x + m.zero**2 )

    m.xsqrdobj = pyo.Constraint([0], pyo.Integers)

    s = pyo.SolverFactory('xpress_persistent')
    prox_manager = ProxApproxManager(m.x, m.xsqrd, m.zero, m.xsqrdobj, 0)
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

    print(f"cuts: {len(m.xsqrdobj)}, iters: {iter_cnt}")
