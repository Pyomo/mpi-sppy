# straight tests (with no unittest, which is a bummer but we need mpiexec)

import sys
import os

def _doone(cmdstr):
    print("testing:", cmdstr)
    ret = os.system(cmdstr)
    if ret != 0:
        raise RuntimeError(f"Test failed with code {ret}:\n{cmdstr}")

# farmer
example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples', 'farmer')
fpath = os.path.join(example_dir, 'farmer_cylinders.py')


cmdstr = f"python {fpath} --help"
_doone(cmdstr)

cmdstr = f"mpiexec -np 1 python {fpath} --help"
_doone(cmdstr)
