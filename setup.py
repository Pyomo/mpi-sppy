#!/bin/usr/env python3
###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import sys

# We raise an error if trying to install with python2
if sys.version[0] == '2':
    print("Error: This package must be installed with python3")
    sys.exit(1)

from setuptools import find_packages
from distutils.core import setup
from pathlib import Path

packages = find_packages()

this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text()

# intentionally leaving out mpi4py to help readthedocs
setup(
    name='mpi-sppy',
    version='0.12.2.dev0',
    description="mpi-sppy",
    long_description=long_description,
    url='https://github.com/Pyomo/mpi-sppy',
    author='David Woodruff',
    author_email='dlwoodruff@ucdavis.edu',
    packages=packages,
    python_requires='>=3.9',
    install_requires=[
        'sortedcollections',
        'numpy<2',
        'scipy',
        'pyomo>=6.4',
    ],
    extras_require={
        'doc': [
            'sphinx_rtd_theme',
            'sphinx',
        ]
    },
)
