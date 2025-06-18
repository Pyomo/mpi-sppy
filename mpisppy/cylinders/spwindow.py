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

import pyomo.environ as pyo

class Field(enum.IntEnum):
    SHUTDOWN=-1000
    NONANT=1
    DUALS=2
    RELAXED_NONANT=3
    BEST_OBJECTIVE_BOUNDS=100 # Both inner and outer bounds from the hub. Layout: [OUTER INNER ID]
    OBJECTIVE_INNER_BOUND=101
    OBJECTIVE_OUTER_BOUND=102
    EXPECTED_REDUCED_COST=200
    SCENARIO_REDUCED_COST=201
    CROSS_SCENARIO_CUT=300
    CROSS_SCENARIO_COST=400
    NONANT_LOWER_BOUNDS=500
    NONANT_UPPER_BOUNDS=501
    BEST_XHAT=600 # buffer having the best xhat and its total cost per scenario
    RECENT_XHATS=601 # buffer having some recent xhats and their total cost per scenario
    WHOLE=1_000_000


field_length_components = pyo.ConcreteModel()
field_length_components._local_nonant_length = pyo.Param(mutable=True)
field_length_components._local_scenario_length = pyo.Param(mutable=True)
field_length_components._total_number_nonants = pyo.Param(mutable=True)
field_length_components._total_number_scenarios = pyo.Param(mutable=True)

# these could be modified by the user...
field_length_components.total_number_recent_xhats = pyo.Param(mutable=True, initialize=10, within=pyo.NonNegativeIntegers)

_field_lengths = {
        Field.SHUTDOWN : 1,
        Field.NONANT : field_length_components._local_nonant_length,
        Field.DUALS : field_length_components._local_nonant_length,
        Field.RELAXED_NONANT : field_length_components._local_nonant_length,
        Field.BEST_OBJECTIVE_BOUNDS : 2,
        Field.OBJECTIVE_INNER_BOUND : 1,
        Field.OBJECTIVE_OUTER_BOUND : 1,
        Field.EXPECTED_REDUCED_COST : field_length_components._total_number_nonants,
        Field.SCENARIO_REDUCED_COST : field_length_components._local_nonant_length,
        Field.CROSS_SCENARIO_CUT : field_length_components._total_number_scenarios * (field_length_components._total_number_nonants + 1 + 1),
        Field.CROSS_SCENARIO_COST : field_length_components._total_number_scenarios * field_length_components._total_number_scenarios,
        Field.NONANT_LOWER_BOUNDS : field_length_components._total_number_nonants,
        Field.NONANT_UPPER_BOUNDS : field_length_components._total_number_nonants,
        Field.BEST_XHAT : field_length_components._local_nonant_length + field_length_components._local_scenario_length,
        Field.RECENT_XHATS : field_length_components.total_number_recent_xhats * (field_length_components._local_nonant_length + field_length_components._local_scenario_length),
}


class FieldLengths:
    def __init__(self, opt):
        number_nonants = (
            sum(
                len(s._mpisppy_data.nonant_indices)
                for s in opt.local_scenarios.values()
               )
        )

        field_length_components._local_nonant_length.value = number_nonants
        field_length_components._local_scenario_length.value = len(opt.local_scenarios)
        field_length_components._total_number_nonants.value = opt.nonant_length
        field_length_components._total_number_scenarios.value = len(opt.local_scenarios)

        self._field_lengths = {k : pyo.value(v) for k, v in _field_lengths.items()}

        # reset the field_length_components
        for p in field_length_components.component_data_objects():
            # leave user-set parameter alone, just clear the
            # "private" parameters
            if p.name.startswith("_"):
                p.clear()

    def __getitem__(self, field: Field):
        return self._field_lengths[field]


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
            # length += 1 # Add 1 for the read id field
            # layout[field] = (offset, length, MPI.DOUBLE)
            layout[field] = (offset, length)
            offset += length
        ## End for

        # If not present already, add field for WHOLE buffer so the entire window
        # buffer can be copied or set in one go
        if Field.WHOLE not in layout:
            # layout[Field.WHOLE] = (0, offset, MPI.DOUBLE)
            layout[Field.WHOLE] = (0, offset)
        ## End if

        self.buffer_layout = layout
        total_buffer_length = offset
        window_size_bytes = MPI.DOUBLE.size * total_buffer_length

        self.buffer_length = total_buffer_length
        self.window = MPI.Win.Allocate(window_size_bytes, MPI.DOUBLE.size, comm=strata_comm)
        # ensure the memory allocated for the window is freed
        self.buff = np.ndarray(dtype="d", shape=(total_buffer_length,), buffer=self.window.tomemory())
        self.buff[:] = np.nan

        for field in self.buffer_layout.keys():
            # (offset, length, mpi_type) = self.buffer_layout[field]
            (offset, length) = self.buffer_layout[field]
            self.buff[offset + length - 1] = 0.0
        ## End for

        self.strata_buffer_layouts = strata_comm.allgather(self.buffer_layout)

        return

    def free(self):
        if self.window is not None:
            self.window.Free()
            self.buff = None
            self.buffer_layout = None
            self.buffer_length = 0
            self.window = None
            self.strata_buffer_layouts = None
            self.window = None
        return

    #### Functions ####
    def get(self, dest: nptyping.ArrayLike, strata_rank: int, field: Field):

        assert(strata_rank >= 0 and strata_rank < len(self.strata_buffer_layouts))

        that_layout = self.strata_buffer_layouts[strata_rank]
        assert field in that_layout.keys()

        # (offset, length, mpi_type) = that_layout[field]
        (offset, length) = that_layout[field]
        assert np.size(dest) == length

        window = self.window
        window.Lock(strata_rank, MPI.LOCK_SHARED)
        # window.Get((dest, length, mpi_type), strata_rank, offset)
        window.Get((dest, length, MPI.DOUBLE), strata_rank, offset)
        window.Unlock(strata_rank)

        return

    def put(self, values: nptyping.ArrayLike, field: Field):

        # (offset, length, mpi_type) = self.buffer_layout[field]
        (offset, length) = self.buffer_layout[field]

        assert(np.size(values) == length)

        window = self.window
        window.Lock(self.strata_rank, MPI.LOCK_EXCLUSIVE)
        # window.Put((values, length, mpi_type), self.strata_rank, offset)
        window.Put((values, length, MPI.DOUBLE), self.strata_rank, offset)
        window.Unlock(self.strata_rank)

        return

## End SPWindow
