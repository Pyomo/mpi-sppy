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

This currently exports tools for memoizing functions and a context manager
which enables the use of changing the program level arguments.
"""

import sys
import inspect
from functools import partial, wraps
from contextlib import contextmanager

def normalize_args(func, pargs, kwargs):
    """
    This function puts the arguments into a dictionary mapping
    keywords to arguments. To do this it must look up the function spec
    for positional arguments.
    """

    # This should be a list of the names of the arguments
    spec = inspect.getargs(func.__code__).args

    # Convert pargs to a list temporarily if need to change any mutable
    # types to immutable types
    pargs = list(pargs)
    # We normalize any list or dictionary arguments to tuples
    for i, parg in enumerate(pargs):
        if isinstance(parg, list):
            pargs[i] = tuple(parg)
        elif isinstance(parg, dict):
            pargs[i] = tuple(sorted(parg.items()))

    for key, value in kwargs.items():
        if isinstance(value, list):
            kwargs[key] = tuple(value)
        elif isinstance(value, dict):
            kwargs[key] = tuple(sorted(value.items()))

    return dict(list(kwargs.items()) + list(zip(spec, pargs)))

def memoize(func):
    """
    This function implements memoization of a function by internally
    storing a dictionary which stores argument-return value pairs. This
    is to be used as a function decorator.

    Note that this only works with functions which has hashable types as
    arguments. This function is designed in particular
    to work with functions which have referential transparency and thus, the
    calculation of a function with the same arguments should be the same every
    time.

    This will convert any list or dictionary arguments to tuples so that
    they can be stored in a dictionary

    Warning: If this function is used over a long period of time with a variety
        of arguments, it can use up a large amount of memory. Do not use this
        with class methods as the cache will exist beyond the life of the
        instance.

    Args:
        func: The function to be memoized
    """

    results = {}

    @wraps(func)
    def f(*pargs, **kwargs):
        args = normalize_args(func, pargs, kwargs)
        arg_key = tuple(sorted(args.items()))
        if arg_key not in results:
            results[arg_key] = func(*pargs, **kwargs)
        return results[arg_key]

    return f


class memoize_method:
    """
    This class will be used as a class method decorator to internally cache
    the results of a method in an instance-level dictionary. This differs
    from the function decorator memoize in that it will store any results
    with the instance meaning that once the instance goes out of scope, the
    cache will be garbage collected and this will not lead to memory leaks.

    In general, any objects passed to a memoized method should be hashable,
    however this will convert any lists or dictionaries passed in to hashable
    tuples to store their values in the cache.

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

        key = (self.func, pargs[1:], frozenset(kwargs))

        try:
            value = cache[key]
        except KeyError:
            value = cache[key] = self.func(*pargs, **kwargs)
        return value


@contextmanager
def set_arguments(args):
    """
    This function will act as a context manager and will set the sys.argv
    variable to the list of arguments passed in. This will enable calling
    other scripts from within python as if they were called from the command
    line.

    Example:
        Say you had a script which simply printed out the system arguments
        defined like such in the file print_args.py:
            import sys 
            def main():
                print(sys.argv)

        Then in a separate file, you could call this function and set the
        system arguments to whatever you want for the entirety of the with
        block and the arguments would be restored at the end.

        In call_print_args.py called like python call_print_args.py 1 2,
            import print_args
            if __name__ == '__main__':
                print(sys.argv) # ['call_print_args.py', '1', '2']
                with set_arguments(['arg1', 'arg2', 'arg3']):
                    print_args.main() # ['arg1', 'arg2', 'arg3'] 
                print(sys.argv) # ['call_print_args.py', '1', '2']
    Args:
        args (List[str]): A list of strings which will become the arguments
    """
    sys.argv_ = sys.argv
    sys.argv = args
    yield
    sys.argv = sys.argv_
