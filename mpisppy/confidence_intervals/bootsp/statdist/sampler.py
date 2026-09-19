###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# pseudo-random numbers from distributions


class Sampler:
    """"
        This class enables generation of pseudo random numbers from distributions
        args:
            distributions (list of BaseDistribution): we sample from inverse of the cdf; len implies sample dimension
            stream (np.random): should be seeded and reseeded by the caller
    """
    def __init__(self, distributions, stream):
        self.distributions = distributions
        self.stream = stream

    def sample_one(self):
        """
            Return a single sample from the distribution as a list
        """
        # independent variables
        retval = []

        for distr in self.distributions:
            unorm = self.stream.uniform(0,1)
            # print(f"{unorm=}")
            retval.append(distr.cdf_inverse(unorm))
        return retval