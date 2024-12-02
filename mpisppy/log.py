###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
This is the logging configuration for mpisppy.

The documentation below is primarily for mpisppy developers.

Examples
========
To use the logger in your code, add the following 
after your import
.. code-block:: python
   
   import logging
   logger = logging.getLogger('mpisppy.path.to.module')

Then, you can use the standard logging functions
.. code-block:: python
   
   logger.debug('message')
   logger.info('message')
   logger.warning('message')
   logger.error('message')
   logger.critical('message')
   
Note that by default, any message that has a logging level
of warning or higher (warning, error, critical) will be
logged.

To log an exception and capture the stack trace
.. code-block:: python

   try:
      c = a / b
   except Exception as e:
      logging.error("Exception occurred", exc_info=True)

"""
import sys
import logging
log_format = '%(message)s'

# configure the root logger for mpisppy
logger = logging.getLogger('mpisppy')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
fmtr = logging.Formatter(log_format)
console_handler.setFormatter(fmtr)
logger.addHandler(console_handler)

def setup_logger(name, out, level=logging.DEBUG, mode='w', fmt=None):
    ''' Set up a custom logger quickly
        https://stackoverflow.com/a/17037016/8516804
    '''
    if fmt is None:
        fmt = "(%(asctime)s) %(message)s"
    log = logging.getLogger(name)
    log.setLevel(level)
    log.propagate = False
    formatter = logging.Formatter(fmt)
    if out in (sys.stdout, sys.stderr):
        handler = logging.StreamHandler(out)
    else: # out is a filename
        handler = logging.FileHandler(out, mode=mode) 
    handler.setFormatter(formatter)
    log.addHandler(handler)
