# This software is distributed under the 3-clause BSD License.

try:
    from mpi4py.MPI import *
    _haveMPI = True

except ImportError:
    import numpy as _np
    import copy as _cp

    UNDEFINED = -1
    SUM = _np.sum
    MAX = _np.max 
    MIN = _np.min
    DOUBLE = _np.double
    INT = _np.intc
    _haveMPI = False

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
            raise RuntimeError(f"Send and receive buffers should be of the same size")
        if send_type != recv_type:
            raise RuntimeError(f"Send and receive buffers should be of the same type")
    
        recv_data[:] = send_data

    COMM_WORLD = _MockMPIComm()
