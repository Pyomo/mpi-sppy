# Mock mpi4py for special uses; e.g., readthedocs
# This software is distributed under the 3-clause BSD License.

class Mock_mpi4py:
    def Get_rank(self):
        return -1
    def Get_size(self):
        return -1

COMM_WORLD = Mock_mpi4py()
