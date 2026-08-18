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
        # A degenerate sample: there is no domain to spread a density over. The
        # model is still abstract at this point -- returning it only defers the
        # failure to the caller, which reads model.tau and gets an
        # AttributeError instead of anything describing the data.
        raise RuntimeError(
            "Cannot fit an epi-spline distribution to degenerate data: the "
            f"error domain collapsed to the single point {alpha}, i.e. every "
            "observation in the sample is the same value.")

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
                samp = np.arange(0.0, 1.0 + s, s)
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

