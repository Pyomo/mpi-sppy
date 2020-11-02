# This software is distributed under the 3-clause BSD License.
#!/bin/usr/env python3
import glob
import sys
import os

# We raise an error if trying to install with python2
if sys.version[0] == '2':
    print("Error: This package must be installed with python3")
    sys.exit(1)

from setuptools import find_packages
from distutils.core import setup

packages = find_packages()

setup(
    name='mpi-sppy',
    version='0.5',
    description="mpi-sppy",
    url='https://github.com/Pyomo/mpi-sppy',
    author='David Woodruff',
    author_email='dlwoodruff@ucdavis.edu',
    packages=packages,
    install_requires=[
        'numpy>=1.19',
        'pyomo>=5.7.1'
    ]
)
