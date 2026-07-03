###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
distribution_factory.py

This module will export a distribution_factory function which should essentially
accept a name for a distribution and return the class associated with that distribution.
This will work by performing a scan through all the modules in this directory and finding
all the classes that are "registered" as distributions.

Registering a  entails using the class decorator register which is also
exported from this module.
"""


distribution_registry = {}


# Right now, this function simply scans through all modules in the current working directory
# It is perhaps more desirable (and safer) to have a list of modules to scan through
# This would be even easier to implement

def import_all_classes():
    """
    Imports all classes in the current directory and stores
    all registered distribution classes in the distribution_registry object
    """
    if distribution_registry:
        # already populated; the registry is stable, so scan only once
        return
    from . import base_distribution
    from . import distributions
    for mod in (base_distribution, distributions):
        for var in mod.__dict__:
            obj = getattr(mod, var)
            if (hasattr(obj, "is_registered_distribution") and
                getattr(obj, "is_registered_distribution")):

                distribution_registry[obj.registered_name] = obj


def register_distribution(name, ndim=None):
    """
    Class decorator with arguments to register a class for use in distribution_factory
    One must specify the name of the distribution to register under and the number of
    dimensions the input argument must be (if there is a requirement).

    This will add the attributes is_registered_distribution, registered_name, and registered_ndim to the class

    Names will be case insensitive.

    Examples:
        To register a function, simply use this as a decorator before any 
        class to be registered

        @register(name='clayton', ndim=2)
        class ClaytonCopula(CopulaBase):
            ...

    Args:
        name (str): The name the distribution will be registered under
        ndim (:obj: `int`, optional): The required dimensionality of input. 
        Defaults to None signaling no requirement

    Returns:
        A class decorator to register a class as a distribution
    """

    def class_decorator(cls):
        cls.is_registered_distribution = True
        cls.registered_name = name.lower()
        cls.registered_ndim = ndim
        return cls

    return class_decorator


def distribution_factory(name):
    """
    This function will accept a name and return the associated
    distribution class raising an error if the name is unrecognized.
    Names should be case insensitive

    Args:
        name (str): The name of the distribution wanted
    Returns:
        The distribution class associated with name
    """

    import_all_classes()

    try:
        distribution = distribution_registry[name.lower()]
    except KeyError:
        possible_names = '\n'.join(sorted(distribution_registry.keys(), key=str.lower))
        raise NameError("The specified distribution {} is unrecognized. "
                        "Possible distributions are:\n{}".format(name, possible_names))

    return distribution
