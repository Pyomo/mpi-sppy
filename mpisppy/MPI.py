###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

try:
    from mpi4py.MPI import *  # noqa: F403
    haveMPI = True

except ImportError:
    import numpy as _np
    import copy as _cp

    UNDEFINED = -1
    SUM = _np.sum
    MAX = _np.max 
    MIN = _np.min
    LOR = _np.logical_or
    DOUBLE = _np.double
    INT = _np.intc
    haveMPI = False

    class _MockMPIComm:
    
        @property
        def rank(self):
            return 0
    
        @property
        def size(self):
            return 1

        def allreduce(self, sendobj, op=SUM):
            return _cp.deepcopy(sendobj)
    
        def barrier(self):
            pass

        def bcast(self, data, root=0):
            return data
    
        def gather(self, obj, root=0):
            if root != self.rank:
                return
            else:
                return [obj]
    
        def Gatherv(self, sendbuf, recvbuf, root=0):
            # TBD: check this
            _set_data(sendbuf, recvbuf)            
    
        def Allreduce(self, sendbuf, recvbuf, op=SUM):
            _set_data(sendbuf, recvbuf)
    
        def Barrier(self):
            pass

        def Bcast(self, data, root=0):
            pass
    
        def Get_rank(self):
            return self.rank
    
        def Get_size(self):
            return self.size
    
        def Split(self, color=0, key=0):
            return _MockMPIComm()
    
    def _process_BufSpec(bufspec):
        if isinstance(bufspec, (list, tuple)):
            bufspec = bufspec[0]
        return bufspec, bufspec.size, bufspec.dtype
    
    def _set_data(sendbuf, recvbuf):
        send_data, send_size, send_type = _process_BufSpec(sendbuf)
        recv_data, recv_size, recv_type = _process_BufSpec(recvbuf)
    
        if send_size != recv_size:
            raise RuntimeError("Send and receive buffers should be of the same size")
        if send_type != recv_type:
            raise RuntimeError("Send and receive buffers should be of the same type")
    
        recv_data[:] = send_data

    COMM_WORLD = _MockMPIComm()
