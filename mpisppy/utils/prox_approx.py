# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

from enum import Enum
from pyomo.core.expr.numeric_expr import LinearExpression

def Side(Enum):
    LEFT = 1
    RIGHT = 2
    BOTH = 3

def ProxApproxManager:
    '''
    A helper class to manage proximal approximations
    '''
    __slots__ = ()
    def __new__(cls, xvar, xvarsqrd, xsqvar_cuts, ndn_i):
        if xvar.is_integer():
            return ProxApproxManagerDiscrete(xvar, xvarsqrd, xsqvar_cuts, ndn_i)
        else:
            return ProxApproxManagerContinuous(xvar, xvarsqrd, xsqvar_cuts, ndn_i)

    def __init__(self, xvar, xvarsqrd, xsqvar_cuts, ndn_i):
        self.xvar = xvar
        self.xvarsqrd = xvarsqrd
        self.var_index = ndn_i
        self.cuts = xsqvar_cuts
        self.cut_index = 0

    def create_initial_cuts(self):
        if self.xvar.lb is None or self.xvar.ub is None:
            raise RuntimeError(f"linearize_nonbinary_proximal_terms requires all "
                                "nonanticipative variables to have bounds")
        lb = self.xvar.lb
        ub = self.xvar.ub

        if lb == ub:
            # var is fixed
            self.create_cut(lb)
            return
        self.add_cut(lb, side=Side.RIGHT)
        self.add_cut(ub, side=Side.LEFT)

    def add_cut(self, val, persistent_solver=None):
        '''
        create a cut at val
        '''
        pass

    def check_tol_add_cut(self, tolerance, persistent_solver=None):
        '''
        add a cut if the tolerance is not satified
        '''
        measured_val = self.xvarsqrd.value
        actual_val = self.xvar.value**2

        if (actual_val - measured_val) > tolerance:
            self.add_cut(self.xvar.value, persistent_solver)
            return True
        return False

class ProxApproxManagerContinuous(ProxApproxManager):

    def add_cut(self, val, persistent_solver=None, side=Side.BOTH):
        '''
        create a cut at val using a taylor approximation
        '''
        assert side is None
        a = val
        f_a = val**2
        f_p_a = 2*val

        const = f_a - f_p_a*a

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

class ProxApproxManagerDiscrete(ProxApproxManager):

    def add_cut(self, val, persistent_solver=None, side=Side.BOTH):
        '''
        create up to two cuts at val, exploiting integrality
        '''
        val = int(round(val))

        # TODO: implement discrete. Issues:
        #       Discrete needs (up to) two cuts per pass,
        #       but one side may have already been added
        #       When adding upper/lower bounds, discrete
        #       only needs one side
        pass
