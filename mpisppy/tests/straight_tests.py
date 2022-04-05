# straight smoke tests (with no unittest, which is a bummer but we need mpiexec)

import os
import sys
import tempfile

from mpisppy.tests.utils import get_solver, round_pos_sig
solver_available, solvername, persistent_available, persistentsolvername= get_solver()

badguys = list()

def _doone(cmdstr):
    print("testing:", cmdstr)
    ret = os.system(cmdstr)
    if ret != 0:
        badguys.append(f"Test failed with code {ret}:\n{cmdstr}")

#####################################################
# farmer
example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples', 'farmer')
fpath = os.path.join(example_dir, 'farmer_cylinders.py')

cmdstr = f"python {fpath} --help"
_doone(cmdstr)

cmdstr = f"mpiexec -np 1 python {fpath} --help"
_doone(cmdstr)


# aircond
example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples', 'aircond')
fpath = os.path.join(example_dir, 'aircond_cylinders.py')
jpath = os.path.join(example_dir, 'lagranger_factors.json')

# PH and lagranger rescale factors w/ FWPH
cmdstr = f"mpiexec -np 4 python -m mpi4py {fpath} --bundles-per-rank=0 --max-iterations=5 --default-rho=1 --solver-name={solvername} --branching-factors 4 3 2 --Capacity 200 --QuadShortCoeff 0.3  --BeginInventory 50 --rel-gap 0.01 --mu-dev 0 --sigma-dev 40 --max-solver-threads 2 --start-seed 0  --no-lagrangian --with-lagranger --lagranger-rho-rescale-factors-json {jpath}"
_doone(cmdstr)


#######################################################
if len(badguys) > 0:
    print("\nstraight_tests.py failed commands:")
    for i in badguys:
        print(i)
    raise RuntimeError("straight_tests.py had failed commands")
else:
    print("straight_test.py: all OK.")
