# This software is distributed under the 3-clause BSD License.

# setup.py for the mock mpi4py; this should be run only under very special circumstances.

from setuptools import find_packages
from distutils.core import setup

packages = find_packages()

setup(
    name='mpi4py',
    version='0.0.1',
    description="mock mpi4py",
    url='https://github.com/Pyomo/mpi-sppy',
    author='David Woodruff',
    author_email='dlwoodruff@ucdavis.edu',
    packages=packages,
    install_requires=[]
)
