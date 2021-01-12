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

    def __new__(cls, xvar, xvarsqrd, xsqvar_cuts, ndn_i, initial_cut_quantity=2):
        if xvar.is_integer():
            return ProxApproxManagerDiscrete(xvar, xvarsqrd, xsqvar_cuts, ndn_i, initial_cut_quantity)
        else:
            return ProxApproxManagerContinuous(xvar, xvarsqrd, xsqvar_cuts, ndn_i, initial_cut_quantity)

class _ProxApproxManager:
    '''
    A helper class to manage proximal approximations
    '''
    __slots__ = ()

    def __init__(self, xvar, xvarsqrd, xsqvar_cuts, ndn_i, initial_cut_quantity):
        self.xvar = xvar
        self.xvarsqrd = xvarsqrd
        self.var_index = ndn_i
        self.cuts = xsqvar_cuts
        self.cut_index = 0
        self._verify_store_bounds(xvar)
        self._create_initial_cuts(initial_cut_quantity)

    def _verify_store_bounds(self, xvar):
        if not (xvar.has_lb() and xvar.has_ub()):
            raise RuntimeError(f"linearize_nonbinary_proximal_terms requires all "
                                "nonanticipative variables to have bounds")
        self.lb = value(xvar.lb)
        self.ub = value(xvar.ub)

    def _get_additional_points(self, initial_cut_quantity):
        '''
        calculate additional points for initial cuts
        '''
        # we add 2 cuts at the bound
        if initial_cut_quantity <= 2:
            return ()

        lb, ub = self.lb, self.ub
        bound_range = ub - lb
        # n+1 points is n hyperplanes,
        # but we've already added the bounds
        delta = bound_range / (initial_cut_quantity-1)

        return (lb + i*delta for i in range(1,initial_cut_quantity-1))

    def _create_initial_cuts(self, initial_cut_quantity):
        '''
        create initial cuts at val
        '''
        pass

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
        y_pnt = self.xvarsqrd.value
        f_val = x_pnt**2

        #print(f"y-distance: {actual_val - measured_val})")

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
            #self.add_cut(x_pnt, persistent_solver)
            self.add_cut(next_val, persistent_solver)
            return True
        return False

class ProxApproxManagerContinuous(_ProxApproxManager):

    def _create_initial_cuts(self, initial_cut_quantity):

        lb, ub = self.lb, self.ub

        # we get zero for free
        if lb != 0.:
            self.add_cut(lb)

        if lb == ub:
            # var is fixed
            return

        if ub != 0.:
            self.add_cut(ub)

        additional_points = self._get_additional_points(initial_cut_quantity)
        for ptn in additional_points:
            self.add_cut(ptn)

    def add_cut(self, val, persistent_solver=None):
        '''
        create a cut at val using a taylor approximation
        '''
        #print(f"adding cut for {val}")
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

    def _create_initial_cuts(self, initial_cut_quantity):
        lb, ub = self.lb, self.ub

        if lb == ub:
            # var is fixed
            self.create_cut(lb)
            return

        #print(f"adding cut for lb {lb}")
        self.add_cut(lb)
        #print(f"adding cut for ub {ub}")
        self.add_cut(ub)

        # there's a left and right cut associated with each discrete point
        # so there's only half the points we cut on total
        # This rounds down, e.g., 7 cuts specified becomes 6
        additional_points = self._get_additional_points(initial_cut_quantity//2+1)
        for ptn in additional_points:
            self.add_cut(ptn)

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

        return cuts_added

if __name__ == '__main__':
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    bounds = (-100, 100)
    m.x = pyo.Var(bounds = bounds)
    #m.x = pyo.Var(within=pyo.Integers, bounds = bounds)
    m.xsqrd = pyo.Var(within=pyo.NonNegativeReals)

    zero = -73.2
    ## ( x - zero )^2 = x^2 - 2 x zero + zero^2
    m.obj = pyo.Objective( expr = m.xsqrd - 2*zero*m.x + zero**2 )

    m.xsqrdobj = pyo.Constraint([0], pyo.Integers)

    s = pyo.SolverFactory('gurobi_persistent')
    prox_manager = ProxApproxManager(m.x, m.xsqrd, m.xsqrdobj, 0, 2)
    s.set_instance(m)
    m.pprint()
    new_cuts = True
    iter_cnt = 0
    while new_cuts:
        print("")
        s.solve(m,tee=False)
        print(f"x: {pyo.value(m.x)}")
        new_cuts = prox_manager.check_tol_add_cut(1e-1, persistent_solver=s)
        #m.pprint()
        iter_cnt += 1

    print(f"objval: {pyo.value(m.obj)}, x: {pyo.value(m.x)}, iters: {iter_cnt}")
