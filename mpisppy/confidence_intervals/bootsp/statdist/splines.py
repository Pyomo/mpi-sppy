###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
splines.py

This module should house all of the functions related
to fitting and evaluating splines.
"""

import math
from collections import OrderedDict

import numpy as np
from pyomo.environ import *

class Spline:
    """
    This fits a epi-spline to the data passed in the lists x and y.
    This has functions for evaluating the spline and computing the derivative
    of the spline, evaluate and derivative, respectively.

    Args:
        x (List[float]): A list of numbers
        y (List[float]): A list of numbers where y = f(x)
        positiveness_constraint (bool): Set to True if spline values should be
                                        positive
        increasingness_constraint (bool): Set to True if spline should be
                                          increasing
        seg_N (int): The desired number of knots for the spline
        seg_kappa (float): The bound on the curvature of the spline
        L1Linf_solver (str): The solver for the L1 norm minimizer
        L2Norm_solver (str): The solver for the L2 norm minimizer
    """
    def __init__(self, x, y, positiveness_constraint=False,
                 epifit_error_norm='L2',
                 seg_N=20, seg_kappa=100, L1Linf_solver='gurobi',
                 increasingness_constraint=False, L2Norm_solver='gurobi'):
        self.model = fit_epispline(x, y, positiveness_constraint,
                                   epifit_error_norm,
                                   seg_N, seg_kappa, L1Linf_solver,
                                   increasingness_constraint, L2Norm_solver)
        self.alpha = self.model.alpha.value
        self.beta = self.model.beta.value
        self.delta = self.model.delta.value

    def _interval_index(self, x):
        """
        Compute the index of the interval x is in in the spline.

        Args:
            x (float): The value x
        """
        l = int(math.ceil(float(x-self.alpha)/self.delta))
        if l == 0:
            l = 1

        return l

    def evaluate(self, x):
        """
        Evaluates the spline at a point x

        Args:
            x (float): The point to evaluate the spline at
        """
        if x < self.alpha or x > self.beta:
            raise ValueError("This spline is only defined on [{}, {}]".format(
                self.alpha, self.beta))

        m = self.model

        s0 = value(m.s0)
        v0 = value(m.v0)
        delta = value(m.delta)

        # We find what interval x is in
        l = self._interval_index(x)

        return (s0 + v0 * x + delta * sum(
                (x - j * delta + 0.5 * delta)
                * value(m.a[j]) for j in range(1, l))
                + 0.5 * value(m.a[l]) * (x - (l - 1) * delta) ** 2)

    __call__ = evaluate

    def derivative(self, x):
        """
        Evaluates the derivative of the spline at a point x

        Args:
            x (float): The point to evaluate the derivative at
        """
        l = self._interval_index(x)
        m = self.model

        v0 = value(m.v0)
        delta = value(m.delta)


        return (v0 + delta*sum(value(m.a[j]) for j in range(1,l))
                + value(m.a[l])*(x-(l-1)*delta))


def fit_epispline(x, y, positiveness_constraint=False, error_norm='L2',
                 seg_N=20, seg_kappa=100, L1Linf_solver='gurobi',
                 increasingness_constraint=False, L2Norm_solver='gurobi'):
    """
    This functions fits an epispline to the function based on passed in input.
    This approximates the function f(x) = y where x and y are passed in lists
    of data.

    Args:
        x (List[float]): A list of numbers
        y (List[float]): A list of numbers where y = f(x)
        positiveness_constraint (bool): Set to True if spline values should be
                                        positive
        increasingness_constraint (bool): Set to True if spline should be
                                          increasing
        seg_N (int): The desired number of knots for the spline
        seg_kappa (float): The bound on the curvature of the spline
        L1Linf_solver (str): The solver for the L1 norm minimizer
        L2Norm_solver (str): The solver for the L2 norm minimizer
    """
    if len(x) != len(y):
        raise RuntimeError('***ERROR: x and y must have the same length.')

    # We first create a new model
    model = ConcreteModel()

    # Sets
    model.I = Set(initialize=list(range(len(x))))
    model.intervals = RangeSet(int(seg_N))

    # Parameters
    model.N = Param(initialize=int(seg_N))
    model.kappa = Param(initialize=float(seg_kappa))

    model.alpha = Param(initialize=min(x))
    model.beta = Param(initialize=max(x))

    def x_init(m, i):
        return x[i]

    model.x = Param(model.I, initialize=x_init)

    def fx_init(m, i):
        return y[i]

    model.fx = Param(model.I, initialize=fx_init)

    model.delta = Param(initialize=float(model.beta - model.alpha) / model.N)

    def k_init(m, i):
        aux = int(math.ceil(float(m.x[i] - m.alpha.value) / m.delta))
        if aux == 0:
            aux = 1
        return aux

    model.k = Param(model.I, initialize=k_init)

    # Variables
    model.e = Var(model.I, within=Reals)
    model.s = Var(model.I, within=Reals)

    model.s0 = Var(within=Reals, initialize=0.0)
    model.v0 = Var(within=Reals, initialize=0.0)
    model.a = Var(model.intervals, bounds=(-model.kappa, model.kappa), initialize=0.0)

    # Constraints
    def compute_spline(m, i):
        return m.s[i] == m.s0 + m.v0 * m.x[i] + m.delta * sum(
            (m.x[i] - j * m.delta + 0.5 * m.delta) * m.a[j] for j in range(1, m.k[i])) \
                         + 0.5 * m.a[m.k[i]] * (m.x[i] - (m.k[i] - 1) * m.delta) ** 2

    model.ComputeSpline = Constraint(model.I, rule=compute_spline)

    # Positiveness
    if positiveness_constraint is True:
        eps = 0.01
        w = [float(i) for i in np.arange(min(x), max(x), eps)]
        model.J = Set(initialize=list(range(len(w))))

        def positive_spline(m, i):
            l = int(math.ceil(float(w[i] - m.alpha) / m.delta))
            if l == 0:
                l = 1
            return m.s0 + m.v0 * w[i] + m.delta * sum(
                (w[i] - j * m.delta + 0.5 * m.delta) * m.a[j] for j in
                range(1, l)) + 0.5 * m.a[l] * (w[i] - (l - 1) * m.delta) ** 2 >= 0

        model.PositiveSpline = Constraint(model.J, rule=positive_spline)

        # Increasingness
    if increasingness_constraint is True:

        # First derivative
        def increasing_spline(m, i):

            l = int(math.ceil(float(w[i] - m.alpha) / m.delta))
            if l == 0:
                l = 1
            return m.v0 + m.delta * sum(m.a[j] for j in
                                        range(1, l)) + m.a[l] * (w[i] - 2 * (l - 1) * m.delta) >= 0

        model.IncreasingSpline = Constraint(model.J, rule=increasing_spline)

    if error_norm == "L1":
        def ePositiveSide_rule(m, i):
            return m.e[i] >= m.fx[i] - m.s[i]

        model.eDefPos = Constraint(model.I, rule=ePositiveSide_rule)

        def eNegativeSide_rule(m, i):
            return m.e[i] >= - m.fx[i] + m.s[i]

        model.eDefNeg = Constraint(model.I, rule=eNegativeSide_rule)
    elif error_norm == "L2":
        def compute_error_rule(m, i):
            return m.e[i] == m.fx[i] - m.s[i]

        model.ComputeError = Constraint(model.I, rule=compute_error_rule)
    else:
        raise RuntimeError("***ERROR: Unknown error norm=" + error_norm + " selected")

    # Objective function
    if error_norm == 'L1':
        def Obj_rule(m):
            return summation(m.e)

        model.Obj = Objective(rule=Obj_rule)
    elif error_norm == 'L2':
        def Obj_rule(m):
            return sum(m.e[i] ** 2 for i in m.I)

        model.Obj = Objective(rule=Obj_rule, sense=minimize)
    else:
        raise RuntimeError("***ERROR: Unknown error norm=" + error_norm + " selected")

    # Instance creation and optimization
    model.preprocess()
    if error_norm == "L1":
        opt = SolverFactory(L1Linf_solver)
        opt.options.mip_tolerances_absmipgap = 0
        opt.options.mip_tolerances_mipgap = 0
        opt.options.mip_tolerances_integrality = 1e-9
    elif error_norm == 'L2':
        opt = SolverFactory(L2Norm_solver)
    else:
        raise RuntimeError("***ERROR: Unknown error norm=" + error_norm + " selected")

    opt.solve(model, tee=False)
    return model


def error_domain(e, dom=None):
    """
    This computes the parameters alpha and beta
    from a list or dictionary, errors, and a
    string dom which specifies how to compute alpha and beta

    alpha and beta will act as the bound on the domain of error distribution.
    If no domain is specified then these will be the minimum and maximum
    of the data

    dom should be a string of the following form "<field>,<field>,...,<field>"
    where <field> is replaced by one of the following:
        1. A number specifying how many standard deviations away from mean
           you want alpha and beta to be set to
        2. pos which fixes alpha to 0 if alpha was prior set to a negative value
        3. neg which sets beta to 0 if beta was prior set to a positive value
        4. min which sets alpha to min
        5. max which sets beta to max
    These fields are processed in order and set alpha and beta to subsequent
    values. This will set alpha (beta) to be the smallest (largest)
    value found while processing each field.

    Args:
        e (List[float]): A list of error values
        dom (str): The specified error string

    Returns (alpha, beta)
    """
    data = e
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    _min = min(data)
    _max = max(data)

    pos_error = ('***Error: You set the domain to be positive and there are ' +
                'some data with negative values')
    neg_error = ('***Error: You set the domain to be negative and there are ' + 
                'some data with positive values')

    if dom is None:
        return _min, _max
    elif isinstance(dom, (int, float)):
        return mu - dom*sigma, mu + dom*sigma
    elif isinstance(dom, str):
        # We set alpha (beta) to max (min) and decrease (increase) as we
        # process each field to ensure we get the smallest (largest) value
        # from all the fields
        alpha, beta = _max, _min
        fields = dom.split(',')
        for i, field in enumerate(fields):
            if is_number(field):
                a = mu-float(field)*sigma
                if a < alpha:
                    alpha = a
                a = mu+float(field)*sigma
                if a > beta:
                    beta = a
            else:
                if field == 'pos' and _min < 0:
                    raise RuntimeError(pos_error)
                elif field == 'neg' and _max > 0:
                    raise RuntimeError(neg_error)
                elif field == 'pos' and alpha < 0:
                    alpha = 0
                elif field == 'neg' and beta > 0:
                    beta = 0
                elif field == 'min' and alpha > _min:
                    alpha = _min
                elif field == 'max' and beta < _max:
                    beta = _max

        return alpha, beta
    else:
        raise RuntimeError("Unrecognized data type for domain")


def is_number(n):
    """
    This function checks if n can be coerced to a floating point.

    Args:
        n (str): Possibly a number string
    """
    try:
        float(n)
        return True
    except:
        return False


def fit_distribution(x, dom=None, specific_prob_constraint=None,
                    seg_N=20, seg_kappa=100,
                    non_negativity_constraint_distributions=0,
                    probability_constraint_of_distributions=1,
                    nonlinear_solver=None):
    """
    Fits a univariate epi-spline distribution to the given data.
    The additional parameter dom defines special characteristics of the support
    of the distribution. It can be pos (positive domain), neg (negative domain)
    or it can be also a float that defines how many standard deviations from
    the mean define the support.

    Args:
        x: list, dict or OrderedDict of data
        dom: A number (int or float) specifying how many standard deviations we
             want to consider as a domain of the distribution or a string that
             defines the sign of the domain (pos for positive and neg
             for negative).
        specific_prob_constraint: either a tuple or a list of length 2
                                  with values for alpha and beta
        seg_N (int): An integer specifying the number of knots
        seg_kappa (float): A bound on the curvature of the spline
        non_negativity_constraint_distributions: Set to 1 if u and w should be
            nonnegative
        probability_constraint_of_distributions: Set to 1 if integral should
            sum to 1
        nonlinear_solver (str): String specifying which solver to use
    Returns:
        (AbstractModel, float, float): tuple consisting of an instance of the
            model, alpha and beta

    Note:
        The data in the model is normalized to [0,1].
    """

    N = int(seg_N)
    kappa = float(seg_kappa)

    # -------------------------------------------------------
    # Model construction
    # -------------------------------------------------------

    model = AbstractModel()

    # -------------------------------------------------------
    # Parameters
    # -------------------------------------------------------

    if isinstance(x, OrderedDict) or isinstance(x, dict):
        days = list(x.keys())
    elif isinstance(x, list):
        days = list(range(len(x)))
    elif isinstance(x, np.ndarray):
        days = list(range(len(x)))
    else:
        raise RuntimeError('***ERROR: Unknown type of input data.')

    intervals = list(range(1, N + 1))
    delta = float(1) / float(N)

    model.N = Param(within=PositiveReals, initialize=N)
    model.delta = Param(within=PositiveReals, initialize=delta)

    if specific_prob_constraint is None:
        alpha, beta = error_domain(x, dom)
    else:
        if isinstance(specific_prob_constraint, tuple):
            alpha, beta = specific_prob_constraint
            alpha = float(alpha)
            beta = float(beta)  # avoid numpy64
        elif isinstance(specific_prob_constraint, list):
            if len(specific_prob_constraint) == 2:
                alpha = specific_prob_constraint[0]
                beta = specific_prob_constraint[1]
            else:
                raise RuntimeError('***ERROR: The list specific_prob_constraint has to have a length of 2.')

        elif isinstance(specific_prob_constraint, str):
            alpha, beta = error_domain(x, dom)
        else:
            raise RuntimeError('***ERROR: specific_prob_constraint has either to be a tuple or a list of length 2.')

    if alpha == beta:  # this means there is only a CONSTANT bias
        return model, alpha, beta

    # Here we normalize the data. Then, m.et is in [0,1].
    def et_init(modelo, j, k=None):
        if k != None:
            val = float(x[j, k] - alpha) / (beta - alpha)
            if val < 0.0:
                return 0.0
            if val > 1.0:
                return 1.0
            return val
        else:
            val = float(x[j] - alpha) / (beta - alpha)
            if val < 0.0:
                return 0.0
            if val > 1.0:
                return 1.0
            return val

    model.et = Param(days, initialize=et_init)

    def tau_init(modelo, i):
        return i * delta

    model.tau = Param(intervals, initialize=tau_init)

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    if non_negativity_constraint_distributions == 1:
        model.w0 = Var(within=NonNegativeReals)
        model.u0 = Var(within=NonNegativeReals)
    else:
        model.w0 = Var()
        model.u0 = Var()
    model.a = Var(intervals, bounds=(0, kappa))

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------

    if probability_constraint_of_distributions == 1:
        def prob_rule(modelo):  # The sum of the probabilities over all the domain must be 1.
            if specific_prob_constraint is not None:
                s = 0.01
                samp = numpy.arange(0.0, 1.0 + s, s)
                aux = 0
                for x in samp:
                    x = float(x)
                    k = int(math.ceil(N * x))
                    if k > 1:
                        tauk = modelo.tau[k - 1]
                    else:
                        tauk = 0
                        k = 1  # avoids erros when i = 0
                    aux += s * exp(-(modelo.w0 + modelo.u0 * x \
                                          + delta * sum(
                        (x - modelo.tau[j] + 0.5 * delta) * modelo.a[j] for j in range(1, k)) \
                                          + 0.5 * modelo.a[k] * (x - tauk) ** 2))
            else:
                aux = delta * exp(-modelo.w0)
                for i in intervals:
                    aux += delta * exp(-(modelo.w0 + modelo.u0 * modelo.tau[i] \
                                              + delta * sum(
                        (modelo.tau[i] - modelo.tau[j] + 0.5 * delta) * modelo.a[j] for j in range(1, i)) \
                                              + 0.5 * modelo.a[i] * delta ** 2))
            return aux == 1

        model.prob = Constraint(rule=prob_rule)

    # -------------------------------------------------------
    # Objective function
    # -------------------------------------------------------

    def fobj_rule(modelo):  # appending _rule we don't need to define rule=rulename
        aux = 0
        for d in days:
            k = int(math.ceil(N * modelo.et[d]))
            if k > 1:
                tauk = modelo.tau[k - 1]
            else:
                tauk = 0
                k = 1  # avoids erros when i = 0
            aux += modelo.w0 + modelo.u0 * modelo.et[d] \
                   + delta * sum((modelo.et[d] - modelo.tau[j] + 0.5 * delta) * modelo.a[j] for j in range(1, k)) \
                   + 0.5 * modelo.a[k] * (modelo.et[d] - tauk) ** 2
        aux /= len(days)

        # We add the integral
        if probability_constraint_of_distributions != 1:
            aux += delta * exp(-modelo.w0)
            for i in intervals:
                aux += delta * exp(-(modelo.w0 + modelo.u0 * modelo.tau[i] \
                                     + delta * sum(
                    (modelo.tau[i] - modelo.tau[j] + 0.5 * delta) * modelo.a[j] for j in range(1, i)) \
                                     + 0.5 * modelo.a[i] * delta ** 2))
        return aux

    model.fobj = Objective(rule=fobj_rule, sense=minimize)

    # -------------------------------------------------------
    # Instance creation and optimization
    # -------------------------------------------------------
    instance = model.create_instance()
    opt = SolverFactory(nonlinear_solver)
    opt.solve(instance, tee=False)

    return instance, alpha, beta

