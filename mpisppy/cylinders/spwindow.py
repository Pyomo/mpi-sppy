###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2025, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from mpisppy import MPI

import numpy as np
import numpy.typing as nptyping

import enum

class Field(enum.IntEnum):
    SHUTDOWN=-1000
    NONANT=1
    DUALS=2
    BOUNDS=100
    INNER_BOUND=101
    OUTER_BOUND=102
    EXPECTED_REDUCED_COST=200
    SCENARIO_REDUCED_COST=201
    WHOLE=1_000_000

class SPWindow:

    def __init__(self, my_fields: dict, strata_comm: MPI.Comm, field_order=None):

        self.strata_comm = strata_comm
        self.strata_rank = strata_comm.Get_rank()

        # Sorted by the integer value of the enumeration value
        if field_order is None:
            self.field_order = sorted(my_fields.keys())
        else:
            self.field_order = field_order

        offset = 0
        layout = {}
        for field in self.field_order:
            length = my_fields[field]
            length += 1 # Add 1 for the read id field
            layout[field] = (offset, length, MPI.DOUBLE)
            offset += length
        ## End for

        # If not present already, add field for WHOLE buffer so the entire window
        # buffer can be copied or set in one go
        if Field.WHOLE not in layout:
            layout[Field.WHOLE] = (0, offset, MPI.DOUBLE)
        ## End if

        self.buffer_layout = layout
        total_buffer_length = offset
        window_size_bytes = MPI.DOUBLE.size * total_buffer_length

        # print("Rank: {}    Total Buffer Length: {}".format(self.strata_rank, total_buffer_length))

        self.buffer_length = total_buffer_length
        self.window = MPI.Win.Allocate(window_size_bytes, MPI.DOUBLE.size, comm=strata_comm)
        self.buff = np.ndarray(dtype="d", shape=(total_buffer_length,), buffer=self.window.tomemory())
        self.buff[:] = np.nan

        for field in self.buffer_layout.keys():
            (offset, length, mpi_type) = self.buffer_layout[field]
            self.buff[offset + length - 1] = 0.0
        ## End for

        # print(self.buff)

        self.strata_buffer_layouts = strata_comm.allgather(self.buffer_layout)

        self.window_constructed = True

        return


    def free(self):

        if self.window_constructed:
            self.window.Free()
            self.buff = None
            self.buffer_layout = None
            self.buffer_length = 0
            self.window = None
            self.strata_buffer_layouts = None
            self.window_constructed = False
        ## End if

        return

    #### Functions ####
    def get(self, dest: nptyping.ArrayLike, strata_rank: int, field: Field):

        # print("Target Rank: {}  Target Field: {}".format(strata_rank, field))

        that_layout = self.strata_buffer_layouts[strata_rank]
        assert field in that_layout.keys()

        (offset, length, mpi_type) = that_layout[field]
        assert np.size(dest) == length

        window = self.window
        window.Lock(strata_rank, MPI.LOCK_SHARED)
        window.Get((dest, length, mpi_type), strata_rank, offset)
        window.Unlock(strata_rank)

        return

    def put(self, values: nptyping.ArrayLike, field: Field):

        (offset, length, mpi_type) = self.buffer_layout[field]

        assert(np.size(values) == length)

        window = self.window
        window.Lock(self.strata_rank, MPI.LOCK_EXCLUSIVE)
        window.Put((values, length, mpi_type), self.strata_rank, offset)
        window.Unlock(self.strata_rank)

        return


## End SPWindow
