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
from pathlib import Path

packages = find_packages()

this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text()

# intentionally leaving out mpi4py to help readthedocs
setup(
    name='mpi-sppy',
    version='0.12.dev0',
    description="mpi-sppy",
    long_description=long_description,
    url='https://github.com/Pyomo/mpi-sppy',
    author='David Woodruff',
    author_email='dlwoodruff@ucdavis.edu',
    packages=packages,
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scipy',
        'pyomo>=6.4',
    ]
)
