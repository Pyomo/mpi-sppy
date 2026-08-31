###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
utilities.py

This module will contain any miscellaneous utilities for processing data
or enhancing functions or anything else.

This currently exports a decorator for memoizing methods.
"""

from functools import partial

class memoize_method:
    """
    This class will be used as a class method decorator to internally cache
    the results of a method in an instance-level dictionary. This differs
    from the function decorator memoize in that it will store any results
    with the instance meaning that once the instance goes out of scope, the
    cache will be garbage collected and this will not lead to memory leaks.

    Any objects passed to a memoized method should be hashable; a call with an
    unhashable argument (a list or a dictionary, say) cannot be cached, so it
    is simply passed through to the method every time.

    This will internally store in any object which has a method decorated
    with this class a dictionary with the name _memoize_method__cache which
    maps functions and their arguments to the corresponding values.

    Example Usage:
        class Obj:
            @memoize_method
            def super_expensive_function(self, arg):
                ...

        obj = Obj()
        obj.super_expensive_function(1) # This time, it will be computed
        obj.super_expensive_function(1) # This time, it will be faster

    This will only compute the function on the first call. On any
    subsequent call, it will look it up in the instance cache.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        """
        This method will turn the decorator into a descriptor. This means
        that trying to access the memoized method will not return the normal
        method, but a slightly modified method.

        In this case, if an instance is calling the method, it will return
        the partially applied __call__ method to the instance. If a class
        is calling the method, it will just return the method.
        """
        if instance is None:
            # This means we are calling it from the class directly
            # We need to pass in all the arguments including instance
            # This is not memoized
            return self.func
        else:
            # Calling from the instance, just need arguments, not instance
            # This will call __call__ and replace the first element of pargs
            # with instance.
            return partial(self, instance)

    def __call__(self, *pargs, **kwargs):
        # The first argument to any instance method is always the instance
        obj = pargs[0]

        # Because the attribute is __cache, the real attribute name is mangled
        # to have the callable name first (in this case memoize_method)
        if hasattr(obj, '_memoize_method__cache'):
            cache = obj.__cache
        else:
            cache = obj.__cache = {}

        # the keyword *values* have to be part of the key: cdf(x, epsabs=1e-4)
        # and cdf(x, epsabs=1e-9) are different questions
        key = (self.func, pargs[1:], frozenset(kwargs.items()))

        try:
            value = cache[key]
        except KeyError:
            value = cache[key] = self.func(*pargs, **kwargs)
        except TypeError:
            # an unhashable argument: call through rather than hiding the
            # method's own behavior behind "unhashable type: 'list'"
            value = self.func(*pargs, **kwargs)
        return value



