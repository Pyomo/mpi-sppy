# This software is distributed under the 3-clause BSD License.
import sys
from runtests.mpi import Tester
import os.path

if __name__ == '__main__':
    tester = Tester(os.path.join(os.path.abspath(__file__)), ".")
    tester.main(sys.argv[1:])
