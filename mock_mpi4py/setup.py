# This software is distributed under the 3-clause BSD License.

# setup.py for the mock mpi4py; this should be run only under very special circumstances.

import os

import setuptools  # this *is* necessary
from distutils.core import setup

os.chdir("mock_mpi4py")

setup(
    name='mpi4py',
    version='99.99.99',
    description="mock mpi4py",
    url='https://github.com/Pyomo/mpi-sppy',
    author='David Woodruff',
    author_email='dlwoodruff@ucdavis.edu',
    packages=['mpi4py'],   #  packages,
    install_requires=[]
)
os.chdir("..")  # probably not needed
