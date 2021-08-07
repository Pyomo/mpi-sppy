#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# This file was originally part of PySP and Pyomo, available: https://github.com/Pyomo/pysp
# Copied with modification from pysp/util/misc.py

__all__ = (
           "load_external_module",
          )

import logging
import sys

from pyutilib.misc import import_file
from pyomo.common.dependencies import attempt_import

logger = logging.getLogger('pysp')

def _generate_unique_module_name():
    import uuid
    name = str(uuid.uuid4())
    while name in sys.modules:
        name = str(uuid.uuid4())
    return name

def load_external_module(module_name,
                         unique=False,
                         clear_cache=False,
                         verbose=False):
    try:
        # make sure "." is in the PATH.
        original_path = list(sys.path)
        sys.path.insert(0,'.')

        sys_modules_key = None
        module_to_find = None
        #
        # Getting around CPython implementation detail:
        #   sys.modules contains dummy entries set to None.
        #   It is related to relative imports. Long story short,
        #   we must check that both module_name is in sys.modules
        #   AND its entry is not None.
        #
        if (module_name in sys.modules) and \
           (sys.modules[module_name] is not None):
            sys_modules_key = module_name
            if clear_cache:
                if unique:
                    sys_modules_key = _generate_unique_module_name()
                    if verbose:
                        print("Module="+module_name+" is already imported - "
                              "forcing re-import using unique module id="
                              +str(sys_modules_key))
                    module_to_find = import_file(module_name, name=sys_modules_key)
                    if verbose:
                        print("Module successfully loaded")
                else:
                    if verbose:
                        print("Module="+module_name+" is already imported - "
                              "forcing re-import")
                    module_to_find = import_file(module_name, clear_cache=True)
                    if verbose:
                        print("Module successfully loaded")
            else:
                if verbose:
                    print("Module="+module_name+" is already imported - skipping")
                module_to_find = sys.modules[module_name]
        else:
            if unique:
                sys_modules_key = _generate_unique_module_name()
                if verbose:
                    print("Importing module="+module_name+" using "
                          "unique module id="+str(sys_modules_key))
                module_to_find = import_file(module_name, name=sys_modules_key)
                if verbose:
                    print("Module successfully loaded")
            else:
                if verbose:
                    print("Importing module="+module_name)
                _context = {}
                module_to_find = import_file(module_name, context=_context, clear_cache=clear_cache)
                assert len(_context) == 1
                sys_modules_key = list(_context.keys())[0]
                if verbose:
                    print("Module successfully loaded")

    finally:
        # restore to what it was
        sys.path[:] = original_path

    return module_to_find, sys_modules_key
