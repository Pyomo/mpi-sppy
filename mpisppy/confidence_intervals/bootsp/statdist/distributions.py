###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
distributions.py

This module houses a host of distribution classes which all adhere to the
interface defined in base_distribution.py
"""

import math
from collections import OrderedDict

import numpy as np
# scipy is an optional dependency; import it lazily so the empirical
# bootstrap path stays scipy-free (mmw_ci.py uses the same pattern).
from pyomo.common.dependencies import scipy

from mpisppy.confidence_intervals.bootsp.statdist.distribution_factory import register_distribution
from mpisppy.confidence_intervals.bootsp.statdist.base_distribution import Parameter
from mpisppy.confidence_intervals.bootsp.statdist.base_distribution import UnivariateDistribution
from mpisppy.confidence_intervals.bootsp.statdist.utilities import memoize_method
from mpisppy.confidence_intervals.bootsp.statdist import splines

epsilon = 1e-12

@register_distribution(name="univariate-unif",ndim=1)
class UnivariateUniformDistribution(UnivariateDistribution):
    """
    This class creates a univariate uniform distribution in the segment [a,b]

    Attributes:
        a (float): The lower bound of the support of the distribution
        b (float): The upper bound of the support of the distribution
    """

    def __init__(self, a, b):
        """
        To construct a UnivariateUniformDistribution object, one must pass in
        the lower and upper bounds for the support of the distribution.
        These are passed in through a and b.

        Args:
            a (float): The lower bound of the support of the distribution
            b (float): The upper bound of the support of the distribution
        """
        if a==b:
            raise ValueError("The bounds should be different")
        self.a=a
        self.b=b
        params = [Parameter('a', a), Parameter('b', b)]
        UnivariateDistribution.__init__(self, params)

    @classmethod
    def fit(cls, data):
        """
        This method will fit a uniform distribution to the data. This will
        set the lower bound of the distribution to the minimum of the data
        and the upper bound to the maximum.

        Args:
            data (List[float]): The list of values to fit the data to
        Returns:
            UnivariateUniformDistribution: The fitted uniform distribution
        """
        return UnivariateUniformDistribution(min(data), max(data))

    def pdf(self, x):
        """
        Args:
            x (float): The values where you want to compute the pdf

        Returns:
            (float) The value of the probability density function of this
                distribution on x.
        """
        if x<self.a or x>self.b:
            return 0
        else:
            return 1/(self.b-self.a)

    def cdf(self, x):
        """
        Args:
            x (float): The values where you want to compute the cdf

        Returns:
            (float) The value of the cumulative density function
        """
        if x<self.a:
            return 0
        elif x<self.b:
            return (x-self.a)/(self.b-self.a)
        else:
            return 1

    def cdf_inverse(self, x):
        return x * (self.b - self.a) + self.a

    def generates_X(self, n=1):
        return np.random.uniform(self.a, self.b, n)


@register_distribution(name="univariate-normal", ndim=1)
class UnivariateNormalDistribution(UnivariateDistribution):
    """
    This class creates a univariate normal distribution (i.e. with a
    gaussian density).
    """
    def __init__(self, var, mean):
        self.var = var
        self.mean = mean
        self.distribution = scipy.stats.norm(loc=self.mean, scale=np.sqrt(self.var))

        params = [Parameter('mean', mean), 
                  Parameter('variance', var, (0, None))]

        UnivariateDistribution.__init__(self, params)

    @classmethod
    def fit(cls, data):
        """
        This will fit a normal distribution to the passed in data.
        This will estimate the mean and variance of the distribution as
        the mean and variance of the data.

        Args:
            data (List[float]): The list of values to fit the data to
        Returns:
            UnivariateNormalDistribution: The fitted normal distribution
        """
        return UnivariateNormalDistribution(np.var(data), np.mean(data))

    def pdf(self, x):
        """
        The probability distribution function of the associated normal
        distribution.

        Args:
            x (float): The values where you want to compute the pdf

        Returns:
            float: The value of the probability density function of this
                distribution on x.
        """
        return self.distribution.pdf(x)

    def cdf(self, x):
        """
        The cumulative distribution function of the associated normal
        distribution.

        Args:
            x (float): The values where you want to compute the cdf

        Returns:
            float: The value of the cumulative density function of this
                distribution on x.
        """

        return self.distribution.cdf(x)

    def cdf_inverse(self, x):
        return self.distribution.ppf(x)

    def generates_X(self, n=1):
        return self.distribution.rvs(n)


@register_distribution(name="univariate-student",ndim=1)
class UnivariateStudentDistribution(UnivariateDistribution):
    """
    This class is a wrapper for the scipy Student's t distribution.
    """
    def __init__(self, df, mean, var):
        """
        To instantiate a student's t distribution, one must pass in the
        degrees of freedom, the mean, and the variance of the distribution.

        Args:
            df (int): The number of degrees of freedom
            mean (float): The mean parameter
            var (float): The variance parameter
        """
        self.df = df
        self.mean = mean
        self.var = var

        # We make the lower bound a very small number to exclude the
        # possibility of 0 for the degrees of freedom.
        params = [Parameter('mean', mean),
                  Parameter('variance', var, (0, None)),
                  Parameter('df', df, bounds=(epsilon, None))]

        UnivariateDistribution.__init__(self, params)

        self.distribution = scipy.stats.t(df=self.df, loc=self.mean,
                                   scale=np.sqrt(self.var))

    @classmethod
    def fit(cls, data):
        """
        This will fit a student's distribution to the passed in data.
        This will estimate the mean and variance of the distribution as
        the mean and variance of the data.

        Args:
            data (List[float]): The list of values to fit the data to
        Returns:
            UnivariateStudentDistribution: The fitted student's t distribution
        """
        var =  np.var(data)
        if var <= 1:
            print('input_data gives Var < 1: '
                  'Impossible to define a student distribution')
            print('Degree of freedom is by default set to 1')
            df=1
        else:
            df = 2*var/(var-1)
        mean = np.mean(data)

        return UnivariateStudentDistribution(df, mean, var)

    def pdf(self, x):
        """
        The probability distribution function of the student's t distribution.

        Args:
            x (float): The values where you want to compute the pdf

        Returns:
            float: The value of the probability density function of this
                distribution on x.
        """
        return self.distribution.pdf(x)

    def cdf(self, x):
        """
        The cumulative distribution function for a student's t distribution.

        Args:
            x (float): The values where you want to compute the cdf

        Returns:
            float: The value of the cumulative density function of this
                distribution on x.
        """
        return self.distribution.cdf(x)

    def cdf_inverse(self, x):
        """
        The inverse cumulative distribution function for a student's t
        distribution.

        Args:
            x (float): The values where you want to compute the inverse cdf

        Returns:
            float: The value of the cumulative density function of this
                distribution on x.
        """
        return self.distribution.ppf(x)

    def generates_X(self, n=1):
        return self.distribution.rvs(n)


@register_distribution(name="univariate-kernel", ndim=1)
class UnivariateGaussianKernelDistribution(UnivariateDistribution):
    """
    This class will fit an Gaussian kernel density estimtion to a vector of data.
    """
    def __init__(self, input_data, bw_method=None, dom_std=1):
        """
        Initializes the distribution.

        Args:
            input_data: list of data points
        """

        # Check the type of the input data and sort it.
        self.kernel = scipy.stats.gaussian_kde(input_data, bw_method)
        self.alpha = min(input_data) - dom_std*np.std(input_data)
        self.beta = max(input_data) +dom_std*np.std(input_data)
        UnivariateDistribution.__init__(self)


    @classmethod
    def fit(cls, data, bw_method=None):
        """
        This function will fit an empirical distribution to the data.

        Args:
            data (List[float]): The data to fit the distribution to
        Returns:
            UnivariateEmpiricalDistribution: The fitted distribution
        """
        return UnivariateGaussianKernelDistribution(data, bw_method=None)

    def pdf(self, x):
        """
        Args:
            x (float): The values where you want to compute the pdf

        Returns:
            (float) The value of the probability density function of this
                distribution on x.
        """
        return self.kernel.evaluate(x)

    def _cdf(self, x):
        """
        Args:
            x (float): The values where you want to compute the cdf

        Returns:
            (float) The value of the cumulative density function
        """
        return self.kernel.integrate_box_1d(self.lower, x)

    #use the cdf_inverse function in base_distribution

    def generate_X(self, n=1, seed=0):
        return self.kernel.resample([n, seed])

   


@register_distribution(name="univariate-epispline", ndim=1)
class UnivariateEpiSplineDistribution(UnivariateDistribution):
    def __init__(self, input_data, error_distribution_domain='4,min,max',
                 specific_prob_constraint=None, seg_N=20, seg_kappa=100,
                 non_negativity_constraint_distributions=0,
                 probability_constraint_of_distributions=1,
                 nonlinear_solver='ipopt'):
        """
        Initializes the distribution.

        Args:
            input_data: list, dict or OrderedDict of data
            dom:    A number (int or float) specifying how many standard
                deviations we want to consider as a domain of the distribution
                or a string that defines the sign of the domain (pos for
                positive and neg for negative).
            specific_prob_constraint: either a tuple or a list of length 2
                with values for alpha and beta
            seg_N (int): An integer specifying the number of knots
            seg_kappa (float): A bound on the curvature of the spline
            non_negativity_constraint_distributions: Set to 1 if u and w should
                be nonnegative
            probability_constraint_of_distributions: Set to 1 if integral of
                the distribution should sum to 1
            nonlinear_solver (str): String specifying which solver to use
        """

        self.dom = error_distribution_domain

        self.specific_prob_constraint = specific_prob_constraint

        self.seg_N = seg_N
        self.seg_kappa = seg_kappa
        self.non_negativity_constraint_distributions = \
            non_negativity_constraint_distributions
        self.probability_constraint_of_distributions = \
            probability_constraint_of_distributions
        self.nonlinear_solver = nonlinear_solver

        # Fits the epi-spline distribution and computes the lower and upper
        # bounds of the domain of the distribution
        model, self.alpha, self.beta = splines.fit_distribution(
            input_data, error_distribution_domain, specific_prob_constraint,
            seg_N, seg_kappa, non_negativity_constraint_distributions,
            probability_constraint_of_distributions, nonlinear_solver)

        # We only store the relevant parameters of the model and not the model
        # itself, as the model takes a significant amount of memory.
        self.tau = {i: model.tau[i] for i in model.tau}
        self.a = {i: model.a[i].value for i in model.a}
        self.delta = model.delta.value
        self.w0 = model.w0.value
        self.u0 = model.u0.value

        # Construction of Parameter objects
        params = []
        params.extend([Parameter('tau_{}'.format(i), self.tau[i]) for i in self.tau])
        params.extend([Parameter('a_{}'.format(i), self.a[i]) for i in self.a])
        params.append(Parameter('delta', self.delta))
        params.append(Parameter('w0', self.w0))
        params.append(Parameter('u0', self.u0))

        # This is an approximation of the integral of the normalized pdf
        # over the entire domain
        self.area = self.beta - self.alpha

        UnivariateDistribution.__init__(self, params, self.alpha, self.beta)

    @classmethod
    def fit(cls, data, **distr_options):
        """
        This function will fit a Univariate EpiSpline Distribution to the data
        passed in. See the docstring for UnivariateEpiSplineDistribution
        for a description of all the options that can be passed in.

        Args:
            data (List[float]): A list of the points to fit the distribution to
        Returns:
            UnivariateEpiSplineDistribution: The fitted distribution
        """
        return UnivariateEpiSplineDistribution(data, **distr_options)

    def _cdf_inverse(self, y):
        """
        prescient_1.0-style inverse calculation

        Integrates the normalized pdf to get cdfs.
        Then interpolates this cdf to get approximation
        of cdf_inverse. The inverse is in the interval [0,1]
        since it acts on the normalized pdf
        Args:
            y (float): number in [0,1] to calculate inverse of,
                or list of values
        Returns:
            Either a list of computed inverses or a single inverse depending
            on what has been passed in
        """
        xs = np.linspace(0,1,100)
        cdfs = [scipy.integrate.quad(self._normalized_pdf, 0, x)[0] for x in xs]
        if isinstance(y, float):
            return np.interp([y], cdfs, xs)
        else:
            return np.interp(y, cdfs, xs)

    def _normalized_pdf(self, x):
        """
        Evaluates the pdf as if it were defined on [0,1]
        Args:
            x (float): a number in [0,1]
        Returns:
            float: the pdf of the unnormalized data
        """
        k = int(math.ceil(self.seg_N * x))
        if k > 1:
            tauk = self.tau[k - 1]
        else:
            tauk = 0
            k = 1  # avoids errors when i = 0
        summation = sum((x - self.tau[j] + 0.5 * self.delta)
                        * self.a[j] for j in range(1, k))

        w = (self.w0 + self.u0 * x
            + self.delta * summation
            + 0.5 * self.a[k] * (x - tauk) ** 2)

        return math.exp(-w)

    @memoize_method
    def _normalized_cdf(self, x):
        return scipy.integrate.quad(self._normalized_pdf, 0, x)[0]

    def pdf(self, x):
        """
        Evaluates the probability density function at a given point x.

        Args:
            x (float): the point at which the pdf is to be evaluated

        Returns:
            float: the value of the pdf

        Note:
            The pdf values are calculated for the original scale of the data
            (i.e. not normalized to [0,1]).
        """
        # Set the pdf to 0 if the variable is out of bounds.
        if x > self.beta or x < self.alpha:
            return 0

        # Noramlize x for using the normalized model.
        norm_x = (x - self.alpha)/(self.beta - self.alpha)

        # Scale the return value to the original data.
        return self._normalized_pdf(norm_x) / self.area


@register_distribution(name="univariate-empirical", ndim=1)
class UnivariateEmpiricalDistribution(UnivariateDistribution):
    """
    This class will fit an empirical distribution to a vector of data.
    """
    def __init__(self, input_data):
        """
        Initializes the distribution.

        Args:
            input_data: list of data points
        """

        # Check the type of the input data and sort it.
        input_data = sorted(input_data)

        if len(input_data) == 0:
            raise ValueError("You must provide at least one value to fit an "
                             "empirical distribution to the data.")

        self.alpha = input_data[0]
        self.beta = input_data[len(input_data)-1]
        self.input_data = input_data
        UnivariateDistribution.__init__(self)

    @classmethod
    def fit(cls, data):
        """
        This function will fit an empirical distribution to the data.

        Args:
            data (List[float]): The data to fit the distribution to
        Returns:
            UnivariateEmpiricalDistribution: The fitted distribution
        """
        return UnivariateEmpiricalDistribution(data)

    def pdf(self, x):
        """
        Evaluates the discrete probability of a given point x.

        Args:
            x (float): the point at which the probability is to be evaluated

        Returns:
            float: the probability
        """

        # Count all self.input_data that are equal to x.
        number = sum(1 for y in self.input_data if x == y)

        return number/len(self.input_data)

    def cdf(self, x, lower_bound=None, upper_bound=None):
        """
        This method calculates a empirical cdf, which is fitted to the data by
        interpolation. If a lower bound is provided, any point smaller will
        have cdf value 0. If an upper bound is provided, any point larger will
        have cdf value 1. If either is not provided the value is estimated
        using the line between the nearest two self.input_data.

        Args:
            x (float): the point at which the cdf is to be evaluated
            lower_bound (float): the lower bound
            upper_bound (float): the upper bound

        Returns:
            float: the value of the cdf

        Notes:
            This method was copied from PINT's distributions class.
        """

        n = len(self.input_data)
        lower_neighbor = None
        lower_neighbor_index = None
        upper_neighbor = None
        upper_neighbor_index = None
        for index in range(n):
            if self.input_data[index] <= x:
                lower_neighbor = self.input_data[index]
                lower_neighbor_index = index
            if self.input_data[index] > x:
                upper_neighbor = self.input_data[index]
                upper_neighbor_index = index
                break

        if lower_neighbor == x:
            cdf_x = (lower_neighbor_index + 1) / (n + 1)

        elif lower_neighbor is None:  # x is smaller than all of the values
            if lower_bound is None:
                x1 = self.input_data[0]
                index1 = self._count_less_than_or_equal(self.input_data, x1)

                x2 = self.input_data[index1]
                index2 = self._count_less_than_or_equal(self.input_data, x2)

                y1 = index1 / (n + 1)
                y2 = index2 / (n + 1)
                interpolating_line = interpolate_line(x1, y1, x2, y2)
                cdf_x = max(0, interpolating_line(x))
            else:
                if lower_bound > x:
                    cdf_x = 0
                else:
                    x1 = lower_bound
                    x2 = upper_neighbor
                    y1 = 0
                    y2 = 1 / (n + 1)
                    interpolating_line = interpolate_line(x1, y1, x2, y2)
                    cdf_x = interpolating_line(x)

        elif upper_neighbor is None:  # x is greater than all of the values
            if upper_bound is None:
                j = n - 1
                while self.input_data[j] == self.input_data[n - 1]:
                    j -= 1
                x1 = self.input_data[j]
                x2 = self.input_data[n - 1]
                y1 = (j+1) / (n + 1)
                y2 = n / (n + 1)
                interpolating_line = interpolate_line(x1, y1, x2, y2)
                cdf_x = min(1, interpolating_line(x))
            else:
                if upper_bound < x:
                    cdf_x = 1
                else:
                    x1 = lower_neighbor
                    x2 = upper_bound
                    y1 = n / (n + 1)
                    y2 = 1
                    interpolating_line = interpolate_line(x1, y1, x2, y2)
                    cdf_x = interpolating_line(x)
        else:
            x1 = lower_neighbor
            x2 = upper_neighbor
            y1 = (lower_neighbor_index + 1) / (n + 1)
            y2 = (upper_neighbor_index + 1) / (n + 1)
            interpolating_line = interpolate_line(x1, y1, x2, y2)
            cdf_x = interpolating_line(x)

        return cdf_x

    def cdf_inverse(self, x, lower_bound=None, upper_bound=None):
        """
        This method calculates a empirical inverse cdf, which is fitted to the
        data by interpolation.

        Args:
            x (float): the point at which the inverse cdf is to be evaluated
            lower_bound (float): the lower bound
            upper_bound (float): the upper bound

        Returns:
            float: the value of the inverse cdf

        Notes:
            This method was copied from PINT's distributions class.
        """

        n = len(self.input_data)
        if x < 0 or x > 1:
            raise ValueError('x must be between 0 and 1!')
        # compute 'index' of this x
        index = x * (n + 1) - 1
        first_index = self._count_less_than_or_equal(
            self.input_data, self.input_data[0]) - 1

        if index < first_index:
            if lower_bound is None:
                # take linear function through (0, self.input_data[0]) and
                # (1, self.input_data[1])
                # input_data[0]) could occur several times,
                # so find highest index j with input_data[j] = input_data[0]
                first_index += 1
                second_index = self._count_less_than_or_equal(
                    self.input_data, self.input_data[first_index])
                interpolating_line = interpolate_line(
                    first_index / (n + 1), self.input_data[0],
                    second_index / (n + 1), self.input_data[first_index])

                return interpolating_line(x)
            else:
                return lower_bound * (1 / (n + 1) - x) / (1 / (n + 1)) + \
                       self.input_data[0] * x / (1 / (n + 1))
        elif index > n - 1:
            if upper_bound is None:
                # take linear function through (n-2, input_data[n-2]) and
                # (n-1, input_data[n-1])
                # NOTE: input_data[n-1] could occur several times,
                # so find lowest index j with input_data[j] = input_data[n-1]
                j = n - 1
                while self.input_data[j] == self.input_data[j - 1]:
                    j -= 1
                    if j - 1 == -len(self.input_data):
                        print("Warning: all input values are the same (",
                              self.input_data[j], ")")
                        return self.input_data[j]
                # g(x) = a*x + b
                a = self.input_data[j] - self.input_data[j - 1]
                b = self.input_data[j - 1] - (self.input_data[j]
                                              - self.input_data[j-1]) * (j-1)
                return a * index + b
            else:
                return self.input_data[n - 1] * \
                       (1 - x) / (1 - n / (n + 1)) + \
                       upper_bound * (x - n / (n + 1)) / (1 - n / (n + 1))
        else:
            if math.floor(index) == index:
                return self.input_data[math.floor(index)]
            else:
                interpolating_line = interpolate_line(
                    x1=math.floor(index),
                    y1=self.input_data[math.floor(index)],
                    x2=math.ceil(index), y2=self.input_data[math.ceil(index)])
                return interpolating_line(index)

    def _count_less_than_or_equal(self, xs, x):
        """
        Counts the number of elements less than or equal to x in
        a sorted list xs

        Args:
            xs: A sorted list of elements
            x: An element that you wish to find the number of elements less
                than it

        Returns:
            int: The number of elements in xs less than or equal to x
        """
        count = 0
        for elem in xs:
            if elem <= x:
                count += 1
            else:
                break
        return count


def interpolate_line(x1, y1, x2, y2):
    """
    This functions accepts two points (passed in as four arguments)
    and returns the function of the line which passes through the points.

    Args:
        x1 (float): x-value of point 1
        y1 (float): y-value of point 1
        x2 (float): x-value of point 2
        y2 (float): y-value of point 2

    Returns:
        callable: the function of the line
    """

    if x1 == x2:
        raise ValueError("x1 and x2 must be different values")

    def f(x):
        slope = (y2 - y1) / (x2 - x1)
        return slope * (x - x1) + y1

    return f

#=========
@register_distribution(name="univariate-discrete", ndim=1)
class UnivariateDiscrete(UnivariateDistribution):
    """
        This class creates a discrete univariate distribution.
        The constructor takes an ordered dict of breakpoints.
    """

    def __init__(self, breakpoints):
        """
        Univariate Discrete distribution constructor.
        args:
            breakpoints (OrderedDict): [value] := probability,
            which need to be in increasing value and with prob that sums to 1.
            Written for 3.x+
        """
        if not isinstance(breakpoints, OrderedDict):
            raise RuntimeError("DiscreteDistribution expecting breakpoints to be a dict")
            
        self.breakpoints = breakpoints
        # check the breakpoints
        tol = 1e-6
        sumprob = 0
        self.mean = 0
        Esqsum = 0
        lastval, prob = list(self.breakpoints.items())[0]
        for val, prob in self.breakpoints.items():
            sumprob += prob
            self.mean += prob * val
            Esqsum += prob * val * val
            if val < lastval:
                raise RuntimeError("DiscreteDistribution dict must be ordered by val:"+str(val)+" < "+str(lastval))
            lastval = val
        self.var = self.mean*self.mean - Esqsum
        if sumprob - 1 > tol: # could use gosm_options.cdf_tolerance
            raise ValueError("Discrete distribution with total prob="
                             +str(sumprob)+" tolerance="+str(tol))
        super(UnivariateDiscrete, self).__init__()

    def pdf(self, x):
        raise RuntimeError("pdf called for a discrete distribution.")
        
    def cdf(self, x):
        """
            Cummulative Distribution Function: prob(X < x), which is weird
            Args:
                x (float): The value where you want to compute the cdf
    
            Returns:
                (float) The value of the cumulative density function of this distribution on x.
        """
        lastval, prob = list(self.breakpoints.items())[0]
        if x < lastval:
            return 0
        elif x == lastval:
            return prob
        sumprob = 0
        for val, prob in self.breakpoints.items():
            sumprob += prob
            if x == val:
                return sumprob
            if x > lastval and x < val:
                return sumprob - prob
            lastval = val
        return sumprob  # should be one if we got this far

    def cdf_inverse(self, x):
        """
        Evaluates the inverse of the cdf at probability value x, but
        that does not really fly for discrete distrs...
        """
        raise RuntimeError("cdf called for a discrete distribution.")

    def sample_one(self):
        """
        Returns a single sample from the distribution

        Returns:
            float or int: the sample
        """
        p = np.random.uniform()
        sumprob = 0
        for val, prob in self.breakpoints.items():
            sumprob += prob
            if sumprob >= p:
                return val
        # if the probs dont' quite sum to one...
        val, prob = list(self.breakpoints.items())[-1]
        return val
    
    def rect_prob(self,down,up):
        """
        
        Args:
            up (float): the upper values where you want to compute the probability
            down (float): the upper values where you want to compute the probability

        Returns: the probability of being between up and down

        """
        return (self.cdf(up)-self.cdf(down))
