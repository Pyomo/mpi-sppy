###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
This abstract base class is the parent class of all distribution classes.
"""
from abc import ABCMeta, abstractmethod
from functools import wraps
import os

import numpy as np
# scipy is an optional dependency; import it lazily so the empirical
# bootstrap path stays scipy-free. matplotlib (also optional) is imported
# locally in the plotting methods below.
from pyomo.common.dependencies import scipy

from mpisppy.confidence_intervals.bootsp.statdist.utilities import memoize_method

class Parameter:
    """
    This class will encode the information to fully specify a parameter for
    a distribution. It will have a name, a value, bounds on what the value
    can be, and the type of the value.

    Attributes:
        name (str): The name of the parameter
        value (float): The value of the parameter, if None, the parameter
            is not instantiated
        bounds (tuple): An ordered pair (a, b) specifying the lower and upper
            bounds of the value inclusive, either may be None to specify a
            lack of bound.
        kind (type): The type of value the parameter has
    """
    def __init__(self, name, value=None, bounds=(None, None), kind=float):
        """
        Args:
            name (str): The name of the parameter
            value (float): The value of the parameter, if None, the parameter
                is not instantiated
            bounds (tuple): An ordered pair (a, b) specifying the lower and 
                upper bounds of the value inclusive. Either may be None to 
                specify a lack of bound.
            kind (type): The type of value the parameter has

        """
        self.name = name
        self.value = value
        self.instantiated = value is None
        self.bounds = bounds
        self.kind = kind

    def set_value(self, value):
        """
        Sets the value attribute.
        Args:
            value: The value to set the parameter to
        """
        self.value = value

    def __repr__(self):
        return "Parameter({},{})".format(self.name, self.value)

    __str__ = __repr__


class BaseDistribution(object):
    __metaclass__ = ABCMeta

    # --------------------------------------------------------------------
    # Abstract methods (have to be implemented within the subclass)
    # --------------------------------------------------------------------

    @abstractmethod
    def __init__(self, dimension=0, parameters=None):
        """
        Initializes the distribution.

        Args:
            dimension (int): the dimension of the distribution
            parameters (list[Parameter]): A list of parameters for the
                distribution
        """
        self.name = self.__class__.__name__
        self.dimension = dimension

        self.parameters = parameters if parameters else []

    @abstractmethod
    def pdf(self, x):
        """
        Evaluates the probability density function at a given point x.

        Args:
            x (float): the point at which the pdf is to be evaluated

        Returns:
            float: the value of the pdf
        """
        pass

    @classmethod
    @abstractmethod
    def fit(cls, data):
        """
        This function will fit the distribution of this class to the passed
        in data. This will return an instance of the class.

        Args:
            data (List[float]): The data the distribution is to be fit to
        Returns:
                baseDistribution: The fitted distribution
        """
        pass

    @staticmethod
    def seed_reset(seed=None):
        """
        Resets the random seed for sampling.
        If no argument is passed, the current time is used.

        Args:
            seed: the random seed
        """
        np.random.seed(seed)

    def __str__(self):
        string = self.name + ': '
        for parameter in self.parameters:
            string += '\n{}: {}'.format(parameter.name, parameter.value)
        return string

    def __repr__(self):
        return "Distribution({})".format(self.name)


class UnivariateDistribution(BaseDistribution):
    """
    This is the base for all univariate distributions. It will have specialized
    pdf and cdf methods which take a single argument.
    """
    __metaclass__ = ABCMeta

    def __init__(self, parameters=None, lower=None, upper=None):
        """
        Args:
            parameters (list[Parameter]): A list of parameters for the
                distribution
        """
        if lower is None:
            self.lower = -np.inf
        else:
            self.lower = lower

        if upper is None:
            self.upper = np.inf
        else:
            self.upper = upper

        BaseDistribution.__init__(self, 1, parameters)

    def plot(self, plot_pdf=True, plot_cdf=True, output_file=None, title=None,
             xlabel=None, ylabel=None, output_directory='.'):
        """
        Plots the pdf/cdf within the interval [alpha, beta].
        If no output file is specified, the plots are shown at
        runtime.

        Args:
            plot_pdf (bool): True if the plot should include the pdf
            plot_cdf (bool): True if the plot should include the cdf
            output_file (str): name of an output file to save the plot
            title (str): the title of the plot
            xlabel (str): the name of the x-axis
            ylabel (str): the name of the y-axis
            output_directory (str): The name of the directory to save the
                files, defaults to the current working directory
        """
        if self.lower == -np.inf:
            lower = -5
        else:
            lower = self.lower

        if self.upper == np.inf:
            upper = 5
        else:
            upper = self.upper

        directory = output_directory
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

        x_range = np.linspace(lower, upper, 100)
        import matplotlib.pyplot as plt
        fig = plt.figure()

        # Plot the pdf if required.
        if plot_pdf:
            y_range = []
            for x in x_range:
                y_range.append(self.pdf(x))
            plt.plot(x_range, y_range, label='PDF', color='blue')

        # Plot the cdf if required.
        if plot_cdf:
            y_range = []
            for x in x_range:
                y_range.append(self.cdf(x))
            plt.plot(x_range, y_range, label='CDF', color='red')

        # Display a legend.
        lgd = plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                         ncol=3, shadow=True)

        # Display a grid and the axes.
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')

        # Name the axes.
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.title(title, y=1.08)

        if output_file is None:
            # Display the plot.
            plt.show()
        else:
            # Save the plot.
            plt.savefig(directory + os.sep + output_file,
                        bbox_extra_artists=(lgd,), bbox_inches='tight')

        plt.close(fig)

    @memoize_method
    def cdf(self, x, epsabs=1e-4):
        """
        Evaluates the cumulative distribution function at a given point x.

        Args:
            x (float): the point at which the cdf is to be evaluated
            epsabs (float): The accuracy to which the cdf is to be calculated

        Returns:
            float: the value of the cdf
        """
        if x <= self.alpha:
            return 0
        elif x >= self.beta:
            return 1
        else:
            return scipy.integrate.quad(self.pdf, self.alpha, x, epsabs=epsabs)[0]

    @memoize_method
    def cdf_inverse(self, x, cdf_inverse_tolerance=1e-4,
                    cdf_inverse_max_refinements=10,
                    cdf_tolerance=1e-4):
        """
        Evaluates the inverse cumulative distribution function at a given
        point x.

        TODO: Explain better how this is calculated

        Args:
            x (float): the point at which the inverse cdf is to be evaluated
            cdf_inverse_tolerance (float): The accuracy which the inverse
                cdf is to be calculated to
            cdf_inverse_max_refinements (int): The number of times the
                the partition on the x-domain will be made finer
            cdf_tolerance (float): The accuracy to which the cdf is calculated
                to
        Returns:
            float: the value of the inverse cdf
        """

        # For ease in calculating the cdf, we define this temp function.
        cdf = lambda x: self.cdf(x, epsabs=cdf_tolerance)

        # This method calculates the cdf of start and then increases
        # (if the cdf value is less than or equal x) or decreases
        # (if the cdf value is greater than x) start iteratively by one
        # stepsize until x is passed. It returns the increased (or decreased)
        # start value and its cdf value.
        def approximate_inverse_value(start):
            cdf_val = cdf(start)
            if x >= cdf_val:
                while x >= cdf_val:
                    start += stepsize
                    cdf_val = cdf(start)
            else:
                while x <= cdf_val:
                    start -= stepsize
                    cdf_val = cdf(start)
            return cdf_val, start

        # Handle some special cases.
        if x < 0 or x > 1:
            return None
        elif abs(x) <= cdf_inverse_tolerance:
            return self.alpha
        elif abs(x-1) <= cdf_inverse_tolerance:
            return self.beta
        else:

            # Initialize variables.
            approx_x = 0
            result = None
            number_of_refinement = 0

            # The starting stepsize was chosen arbitrarily.
            stepsize = (self.beta - self.alpha)/10

            while abs(approx_x - x) > cdf_inverse_tolerance \
                    and number_of_refinement <= cdf_inverse_max_refinements:

                # If this is the first iteration, start at one of the bounds
                # of the domain.
                if number_of_refinement == 0:

                    # If x is greater than or equal 0.5, start the
                    # approximation at the upper bound of the domain.
                    if x >= 0.5:
                        approx_x, result = approximate_inverse_value(self.beta)

                    # If x is less than 0.5, start the approximation at
                    # the lower bound of the domain.
                    else:
                        approx_x, result = approximate_inverse_value(
                            self.alpha)
                else:

                    # If this is not the first iteration, halve the stepsize
                    # and call the approximation method.
                    stepsize /= 2
                    approx_x, result = approximate_inverse_value(result)

                number_of_refinement += 1

            return result

    def mean(self):
        """
        Computes the mean value (expectation) of the distribution.

        Returns:
            float: the mean value
        """

        # Use region_expectation to compute the mean value.
        return self.region_expectation((self.alpha, self.beta))

    @memoize_method
    def region_expectation(self, region):
        """
        Computes the mean value (expectation) of a specified region.

        Args:
            region: the region (tuple of dimension 2) of which the expectation
                is to be computed

        Returns:
            float: the expectation
        """

        # Check whether region is a tuple of dimension 2.
        if isinstance(region, tuple) and len(region) == 2:
            a, b = region
            if a > b:
                raise ValueError("Error: The upper bound of 'region' can't be "
                                 "less than the lower bound.")
        else:
            raise TypeError("Error: Parameter 'region' must be a tuple of "
                            "dimension 2.")

        integral, _ = scipy.integrate.quad(lambda x: x * self.pdf(x), a, b)

        return integral

    @memoize_method
    def region_probability(self, region):
        """
        Computes the probability of a specified region.

        Args:
            region: the region of which the probability is to be computed

        Returns:
            float: the probability
        """

        # Compute the region's probability by integration,

        # Check whether region is a tuple of dimension 2.
        if isinstance(region, tuple) and len(region) == 2:
            a, b = region
            integral, _ = scipy.integrate.quad(self.pdf, a, b)
        else:
            raise ValueError("Error: Parameter 'region' must be a tuple of"
                             " dimension 2.")

        return integral

    def conditional_expectation(self, interval, cdf_inverse_tolerance=1e-4,
                    cdf_inverse_max_refinements=10,
                    cdf_tolerance=1e-4):
        """
        This computes the conditional expectation of the distribution
        conditioned on being in the hyperrectangle passed in.
        The hyperrectangle will actually for this be just an interval contained
        in [0, 1] potentially with some cutouts. This will work for
        1-dimensional hyperrectangles, the multivariate distribution subclass
        should implement a different version of this.

        If the region is (a, b), this will compute the expectation on
        [cdf^-1(a), cdf^-1(b)] and divide it by (b-a).

        Args:
            Interval (Interval): An interval on which
                the conditional expectation is to be computed on
            cdf_inverse_tolerance (float): The accuracy which the inverse
                cdf is to be calculated to
            cdf_inverse_max_refinements (int): The number of times the
                the partition on the x-domain will be made finer
            cdf_tolerance (float): The accuracy to which the cdf is calculated
                to
        """
        a, b = interval.a, interval.b
        cdf_inverse = lambda x: self.cdf_inverse(x, cdf_inverse_tolerance,
                                                 cdf_inverse_max_refinements,
                                                 cdf_tolerance)

        lower, upper = cdf_inverse(a), cdf_inverse(b)
        expectation = self.region_expectation((lower, upper))
        probability = b-a

        # A hyperrectangle may subtract some intervals from the larger interval
        if hasattr(interval, 'cutouts'):
            for cutout in interval.cutouts:
                a, b = cutout.a, cutout.b
                lower, upper = cdf_inverse(a), cdf_inverse(b)
                expectation -= self.region_expectation((lower, upper))
                probability -= b-a
        return expectation / probability

    def sample_one(self):
        """
        Returns a single sample of the distribution

        Returns:
            float: the sample
        """

        return self.cdf_inverse(np.random.uniform())

    def sample_on_interval(self, a, b):
        """
        This samples from the distribution conditioned on X being in [a, b].
        This does this by sampling uniformly on [F(a), F(b)] and then applying
        the inverse transform to the result.

        Args:
            a (float): The lower limit of the interval
            b (float): The upper limit of the interval
        Returns:
            float: The sampled value in the interval
        """
        return self.sample_between_quantiles(self.cdf(a), self.cdf(b))

    def sample_between_quantiles(self, a, b):
        """
        This samples from the distribution conditioned on the quantile of the
        point being between a and b, i.e., it generates X given that
        a < F(X) < b. It does this by sampling from a uniform distribution on
        (a,b) and then applying the inverse transform to the point.

        Args:
            a (float): The lower quantile, must be between 0 and 1.
            b (float): The upper quantile, must be between 0 and 1.
        Returns:
            float: The sampled value
        """
        y = np.random.uniform(a, b)
        return self.cdf_inverse(y)

    def log_likelihood(self, data):
        """
        This method will return the log likelihood of the observed data
        given the fitted model.

        Args:
            data (list[float]): A list of observed values
        Returns:
            float: The computed log-likelihood
        """
        return sum(np.log(self.pdf(x)) for x in data)


class MultivariateDistribution(BaseDistribution):
    """
    This class is an abstract base class for all multivariate distributions
    TODO: This docstring should be improved greatly!!
    """
    __metaclass__ = ABCMeta

    def __init__(self, dimension, dimkeys=None, parameters=None, lower=None,
                 upper=None, bounds=None):
        """
        Args:
            dimension (int): The dimension of the distribution
            dimkeys (List): A list of the names of the dimensions, by default,
                these will just be the indices. If passed in, this will enable
                you to refer to values by the dimension name in certain
                functions
            parameters (list[Parameter]): A list of parameters for the
                distribution
            lower (list[float]): A list of the lower bounds of the support
                of the distribution
            upper (list[float]): A list of the upper bounds of the support
                of the distribution
            bounds (list|dict): A colection of bounds on the support for
                each dimension. We assume the support is on a rectangular
                region. If it is passed as a dictionary it should map
                dimension names to ordered pairs of lower and upper bounds. A
                None indicates that there is no lower or upper bound for a
                given dimension.
        """
        BaseDistribution.__init__(self, dimension, parameters)
        self.ndim = dimension
        if dimkeys is None:
            # We default to using the integers if no dimkeys are passed in.
            self.dimkeys = list(range(self.ndim))
        else:
            self.dimkeys = dimkeys

        if lower is None:
            self.lower = [None for _ in range(dimension)]
        else:
            self.lower = lower
        if upper is None:
            self.upper = [None for _ in range(dimension)]
        else:
            self.upper = upper

        if bounds is None:
            self.bounds = [(-np.inf, np.inf)] * dimension
        elif isinstance(bounds, list):
            self.bounds = bounds
        elif isinstance(bounds, dict):
            self.bounds = [bounds[dim] for dim in self.dimkeys]

    def pdf(self, *xs):
        raise NotImplementedError

    def log_likelihood(self, data):
        """
        This method will return the log likelihood of the observed data
        given the fitted model.

        This method just naively computes the pdf and applies the logarithm.
        It would be more efficient in subclasses to find an expression for
        the log-likelihood.

        The argument data can either be a list of vectors for each dimension
        of the data or it can be a dictionary mapping dimension names to the
        corresponding vector of data.

        Args:
            data (list[list[float]] | dict[list[float]]): The observed values
        Returns:
            float: The computed log-likelihood
        """
        if isinstance(data, dict):
            vects = [data[dimkey] for dimkey in dimkeys]
        else:
            vects = data

        return sum(np.log(self.pdf(*xs)) for xs in zip(*vects))

    def plot(self, func, lower=None, upper=None):
        """
        Args:
            func (str): The function to plot, either 'pdf' or 'cdf'
            lower (list[float]): A list of the lower bounds for the plot,
                will default to the lower bounds of the support if None
            upper (list[float]): A list of the upper bounds of the plot
                will default to the upper bounds of the support if None
        """
        if self.dimension != 2:
            raise ValueError("This plot method is only functional for 2-d "
                             "distributions.")

        if lower is None:
            lower = self.lower
            if lower[0] is None:
                lower[0] = -5
            if lower[1] is None:
                lower[1] = 5
        if upper is None:
            upper = self.upper
            if upper[0] is None:
                upper[0] = -5
            if upper[1] is None:
                upper[1] = 5

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X = np.arange(lower[0], upper[0], 0.1)
        Y = np.arange(lower[1], upper[1], 0.1)
        X, Y = np.meshgrid(X, Y)

        Z = np.zeros_like(X)
        for i, row in enumerate(X):
            for j, x in enumerate(row):
                y = Y[i,j]
                if func == 'pdf':
                    z = self.pdf(x, y)
                elif func == 'cdf':
                    z = self.cdf(x, y)

                Z[i,j] = z

        ax.plot_surface(X, Y, Z)
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        return ax

    def contour_plot(self, func, lower=None, upper=None):
        """
        Args:
            func (str): The function to plot, either 'pdf' or 'cdf'
            lower (list[float]): A list of the lower bounds for the plot,
                will default to the lower bounds of the support if None
            upper (list[float]): A list of the upper bounds of the plot
                will default to the upper bounds of the support if None
        """
        if self.dimension != 2:
            raise ValueError("This plot method is only functional for 2-d "
                             "distributions.")

        if lower is None:
            lower = self.lower
        if upper is None:
            upper = self.upper

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        X = np.arange(lower[0], upper[0], 0.1)
        Y = np.arange(lower[1], upper[1], 0.1)
        X, Y = np.meshgrid(X, Y)

        Z = np.zeros_like(X)
        for i, row in enumerate(X):
            for j, x in enumerate(row):
                y = Y[i,j]
                if func == 'pdf':
                    z = self.pdf(x, y)
                elif func == 'cdf':
                    z = self.cdf(x, y)

                Z[i,j] = z

        ax.contour(X, Y, Z)
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        return ax

    @memoize_method
    def rect_prob(self, lowerdict, upperdict):
        tempdict = dict.fromkeys(self.dimkeys)
        def f(n):

            # recursive function that will calculate the cdf
            # It has a structure of binary tree
            if n == 0:
                return self.cdf(tempdict)
            else:
                tempdict[self.dimkeys[n - 1]] = upperdict[self.dimkeys[n - 1]]
                leftresult = f(n - 1)
                tempdict[self.dimkeys[n - 1]] = lowerdict[self.dimkeys[n - 1]]
                rightresult = f(n - 1)
                return leftresult - rightresult

        return f(self.dimension)

    def marginal(self, ys, bounds = None, error_tolerance=None):
        """
        This function will evaluate the marginal distribution of the joint
        cdf which is composed of the dimensions passed in through ys.
        It will evaluate it at the point ys.

        Args:
            ys (dict): A dictionary mapping dimension names to their
                corresponding values
            error_tolerance (int): Value to increase the error tolerance
                of the integration process by powers of 10
        Returns:
            float: The value of the marginal
        """

        other_dims = [(i, dim) for i, dim in enumerate(self.dimkeys)
                      if dim not in ys]

        point_dict = ys.copy()

        def pdf_x(*xs):
            for (_, dim), x in zip(other_dims, xs):
                point_dict[dim] = x
            #print(self.pdf(point_dict))
            return self.pdf(point_dict)
        if bounds == None:
            bounds = [self.bounds[i] for i, _ in other_dims]

        if error_tolerance:
            tol = error_tolerance
        else:
            tol = 0

        return scipy.integrate.nquad(pdf_x, bounds, opts={'epsabs': (1.49e-08 * (10**tol)), 'epsrel': (1.49e-08 * (10**tol))} )[0]

    def conditional_pdf(self, xs, cond_xs, marginal_cdf=None):
        """
        This will evaluate the conditional pdf at the point xs given
        that the dimensions in cond_names

        Args:
            xs (dict): A dictionary mapping dimension names to values
            cond_xs (dict): A dictionary mapping the dimension names
                of the conditioned variables to their values
        Returns:
            float: The value fo the conditional pdf
        """
        if marginal_cdf == None:
            marg = self.marginal(cond_xs)
        else:
            marg = marginal_cdf

        point_dict = {}
        for dim, x in xs.items():
            point_dict[dim] = x
        for dim, x in cond_xs.items():
            point_dict[dim] = x
        return self.pdf(point_dict) / marg

    def conditional_cdf(self, xs, cond_xs, marginal_cdf = None):
        """
        This will evaluate the conditional cdf at the point xs given
        that the dimensions in cond_names are set to the values in cond_xs.

        Args:
            xs (dict): A dictionary mapping dimension names to values
            cond_xs (dict): A dictionary mapping the dimension names
                of the conditioned variables to their values
        Returns:
            float: The value fo the conditional cdf
        """
        bounds = []

        dimkeys = list(xs.keys())

        for dim in dimkeys:
            dim_index = self.dimkeys.index(dim)
            lower_bound = self.bounds[dim_index][0]
            bounds.append([lower_bound, xs[dim]])

        point_dict = cond_xs.copy()
        def f(*xs):
            for dim, x in zip(dimkeys, xs):
                point_dict[dim] = x
            return self.pdf(point_dict)

        if marginal_cdf == None:
            marg = self.marginal(cond_xs)
        else:
            marg = marginal_cdf

        try:
            return scipy.integrate.nquad(f, bounds)[0] / marg
        except:
            return 0

    def conditional_cdf_inverse(self, cond_xs, cdf_value, dim, marginal,
                                capacity = 4000, n = 100,
                                method = 'combination', xtol = 0.001):
        """
        This function computes the inverse of a conditional cdf value
        conditioned on a given point. Therefore 3 different methods are
        provided:
        - linear interpolation: The conditional cdf is evaluated at several
            points in a given interval. Since two points which conditional cdfs
            wrap the cdf_value, a linear interpolation between these two
            points is used to compute the inverse of the cdf_value.
        - bisection: The bisection method from scipy is used to solve the
            equation 0 = conditional_cdf - cdf_value.
        - combination of both: First a bisection method is used to find the
            two wrapping points like in the linear interpolation method. After
            that a linear interpolation is used to compute the inverse.

        Args:
             cond_xs (dict): A dictionary mapping dimension the dimension
                names of the conditioned variables to their values.
             cdf_value: The value you want to compute the inverse for.
             dim (int or str): The name of the dimension you want to get the
                inverse for (e.g. F(X|Y=500) = 0.2: You want to compute the
                value of X under the condition that Y=500, so that F equals
                0.2. In that case "dim" equals X.).
             marginal (distribution like): The marginal of dimension dim.
             capacity (float): The capacity for that day.
             n (int): The number of intersection of the interval
                [-capacity, capacity], which specify the points which are
                evaluated for the linear interpolation method. The number
                specifies also a break criteria for the bisection part in the
                combined  method.
             method (str): The method you want to use. "default" refers to the
                linear interpolation method, "bisect" to the bisection method
                and "combination" to the combined method.
             xtol (float): The tolerance for the bisection method.
                (break criteria)

        Returns:
            The inverse value of the passed in cdf_value conditioned on the
            point cond_xs.
        """
        marginal_cdf = self.marginal(cond_xs)  # This value is needed a lot. So
                                               # it is computed here once.

        if method == 'default':
            """
            For the default or linear interpolation method first a list of 
            points are created. These points are evaluated one after the other
            with the conditional_cdf function. After that it is checked, if the
            given cdf_value is wrapped by two consecutive points' conditional
            cdf. If thats the case, these two points and there conditional cdfs
            are used to compute a linear interpolation between them. This 
            linear interpolation then is used to compute the inverse of the
            given cdf_value. Because the conditional_cdf lives in the copula
            space (which is [0,1]^n), the points have to be converted to [0,1]. 
            For the purpose of getting power values as a return, the computed
            inverse values have to be transformed back in the end.
            """
            points = np.linspace(-capacity, capacity, n)
            x = []
            for point in points:
                x.append(marginal.cdf(point))
            y = []
            j = 0
            for i in x:
                xs = {dim: i}
                yi = self.conditional_cdf(xs, cond_xs)
                y.append(yi)
                if (j==0) and (yi > cdf_value):
                    inverse = marginal.cdf(-capacity)
                    break
                elif (j != 0):
                    if y[j-1] <= cdf_value <= y[j]:   #linear interpolation
                        lin = scipy.interpolate.interp1d([y[j-1], y[j]], [x[j-1], x[j]])
                        inverse = lin(cdf_value)      #computing the inverse
                        break
                j += 1
            else:
                inverse = marginal.cdf(capacity)
            return marginal.cdf_inverse(inverse)
        elif method == 'bisect':
            """
            In this method a help function is defined. After that the root
            of this function is computed using the bisection mehtod from
            scipy. For more information see the scipy documentation.
            The transformation of the values is done for the same reason like
            above.
            """
            def help(d):
                dict = {dim: d}
                return self.conditional_cdf(dict, cond_xs,
                                            marginal_cdf=marginal_cdf) \
                       - cdf_value
            if help(marginal.cdf(-capacity)) > 0:
                inverse = -capacity
            elif help(marginal.cdf(capacity)) < 0:
                inverse = capacity
            else:
                inverse = marginal.cdf_inverse(scipy.optimize.bisect(help,
                                                          marginal.cdf(-capacity),
                                                          marginal.cdf(capacity),
                                                          xtol=xtol))

            return inverse


        elif method == 'combination':
            """
            In this method not every single point is evaluated. There is some-
            thing like a bisection method used to find faster the wrapping
            points. After they are found, the linear interpolation is used
            to compute the inverse value.
            The transformation of the values is done for the same reason like
            above.
            """
            if capacity is None:
                capacity = 0
            l = -capacity
            u = capacity
            l_cdf = self.conditional_cdf({dim: marginal.cdf(l)}, cond_xs,
                                         marginal_cdf=marginal_cdf)
            u_cdf = self.conditional_cdf({dim: marginal.cdf(u)}, cond_xs,
                                         marginal_cdf=marginal_cdf)
            if l_cdf >= cdf_value:
                print('cdf', cdf_value)
                print('lower', l_cdf)
                return l
            elif u_cdf <= cdf_value:
                print('cdf', cdf_value)
                print('upper', u_cdf)
                return u
            tol = (capacity * 2) / n
            k = 0
            while ((u - l) > tol) and (k < n):
                m = (u + l) / 2
                m_cdf = self.conditional_cdf({dim: marginal.cdf(m)}, cond_xs,
                                             marginal_cdf=marginal_cdf)
                if m_cdf < cdf_value:
                    l = m
                    l_cdf = m_cdf
                elif m_cdf > cdf_value:
                    u = m
                    u_cdf = m_cdf
                else:
                    return m
                k = k + 1
            lin = scipy.interpolate.interp1d([l_cdf, u_cdf], [marginal.cdf(l), marginal.cdf(u)])
            return marginal.cdf_inverse(lin(cdf_value))


def fit_wrapper(method):
    """
    This is a function decorator which will wrap the fit method for
    multivariate distributions. It will allow for data to be passed using
    a dictionary mapping names to lists of data.

    Internally this transforms the data into a lists of lists and then fits
    the distribution to that data. Then it assigns to the dimkeys attribute
    the list of names.

    Args:
        method: The class method fit of a multivariate distribution
    Returns:
        method: The modified method to handle dictionaries of input data
    """
    @wraps(method)
    def fit(cls, data, dimkeys=None, **kwargs):
        """
        This function converts the dictionary into a list, passes it to the
        fit method and then assigns to the distribution the dimkeys attribute.
        """
        vectors = []
        if isinstance(data, dict):
            for key in dimkeys:
                vectors.append(data[key])
        else:
            vectors = data

        distribution = method(cls, vectors, dimkeys, **kwargs)
        return distribution

    return fit


def accepts_dict(method):
    """
    This function decorator will allow any of the methods which accept separate
    values for each dimension to also accept a dictionary which has keys
    mapping to each dimension.

    For example, the pdf for any distribution generally has the prototype
        def pdf(self, *x):
    This decorator will unpack the dictionary into its respective dimensions
    and pass it to the function.

    The function that this decorator is applied to must have a prototype like
        def f(self, *x)

    This will enable you to call a function in the following three ways.

    Suppose distr is a Distribution with distr.dimkeys = ['foo', 'bar', 'baz']
    If pdf is decorated with accepts_dict, we can call it like so
        1) distr.pdf(1, 2, 3)
        2) distr.pdf(foo=1, bar=2, baz=3)
        3) distr.pdf({'foo': 1, 'bar': 2, 'baz': 3})

    Args:
        method: The method accepting the different values for each dimensions
    Returns:
        method: The modified method to handle dictionaries of input data
    """
    @wraps(method)
    def f(self, *xs, **kwargs):
        if xs:
            # If xs is passed in, we check if the user passed it as each
            # dimension separately or as a dictionary
            if isinstance(xs[0], dict):
                # If the first element of xs, is a dict, assume only element.
                value_dict = xs[0]
                values = [value_dict[key] for key in self.dimkeys]
            else:
                # Otherwise, it is a list of the values at each dimension
                values = xs
        else:
            # Otherwise, we expect the values to be passed as keyword args.
            values = [kwargs[key] for key in self.dimkeys]
        return method(self, *values)

    return f


def returns_dict(method):
    """
    This function decorator will allow descendants of MultivariateDistribution
    which have methods which return values for each dimension to instead
    return a dictionary of values mapping dimension name to value.

    This adds an as_dict argument which if set to True, will pack the return
    value into a dictionary assuming the order is in that of the dimkeys
    attribute of the distribution.

    The as_dict argument must be passed by keyword.

    Args:
        method: The method which returns a list of values for each dimension
    Returns:
        method: The modified method to return a dictionary if specified to
    """

    @wraps(method)
    def f(self, *pargs, as_dict=False, **kwargs):
        values = method(self, *pargs, **kwargs)
        if as_dict:
            output = {key: value for key, value in zip(self.dimkeys, values)}
            return output
        else:
            return values

    return f


def params_as_args(arg_names):
    def decorator(method):
        """

        """
        @wraps(method)
        def f(cls, x, params=None):
            if params is None:
                params = {}
                for name in arg_names:
                    value = getattr(cls, name).value
                    if value is None:
                        message = """The {} parameter is unset. To use this method
                                  it must be either called from an instance of
                                  the distribution class or it must be called
                                  directly from the class with a dictionary of
                                  the parameters passed
                                  with the params keyword.""".format(name)
                        raise ValueError(message)
                    params[name] = value
            return method(cls, x, params)
        return f
    return decorator

def params_as_args2(arg_names):
    def decorator(method):
        """

        """
        @wraps(method)
        def f(cls, x, y, params=None):
            if params is None:
                params = {}
                for name in arg_names:
                    value = getattr(cls, name).value
                    if value is None:
                        message = """The {} parameter is unset. To use this method
                                  it must be either called from an instance of 
                                  the distribution class or it must be called
                                  directly from the class with a dictionary of
                                  the parameters passed
                                  with the params keyword.""".format(name)
                        raise ValueError(message)
                    params[name] = value
            return method(cls, x, y, params)
        return f
    return decorator
