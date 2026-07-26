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
        # a parameter is instantiated once it has a value (the docstring above:
        # "if None, the parameter is not instantiated")
        self.instantiated = value is not None
        self.bounds = bounds
        self.kind = kind

    def set_value(self, value):
        """
        Sets the value attribute.
        Args:
            value: The value to set the parameter to
        """
        self.value = value
        self.instantiated = value is not None

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

