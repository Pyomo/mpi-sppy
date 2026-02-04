###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2025, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

# Feb 2026: the layout tuple is (offset, logical_len, padded_len) with offset += padded_len
#   (we are padding to be 512 bit boundaries to avoid collisions with solvers who do that)

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


def _padded_len_8doubles(logical_len: int) -> int:
    """Round up length (in doubles) to a multiple of 8 doubles (64 bytes)."""
    return ((logical_len + 7) // 8) * 8


class SPWindow:

    def __init__(self, my_fields: dict, strata_comm: MPI.Comm, field_order=None):
        """
        Design A (padded transfers):
          - Layout tuple is (offset, logical_len, padded_len) and offset advances by padded_len.
          - put/get always transfer padded_len doubles.
          - ID slot is at (offset + logical_len - 1).
        """
        self.strata_comm = strata_comm
        self.strata_rank = strata_comm.Get_rank()

        # Sorted by the integer value of the enumeration value
        if field_order is None:
            self.field_order = sorted(f for f in my_fields.keys() if f != Field.WHOLE)
        else:
            self.field_order = [f for f in field_order if f != Field.WHOLE]

        offset = 0
        layout = {}

        for field in self.field_order:
            entry = my_fields[field]

            # In your spcommunicator.py, entry is (logical_len, padded_len)
            if isinstance(entry, (tuple, list)):
                logical_len = int(entry[0])
                padded_len = int(entry[1])
            else:
                # Fallback: caller provided padded only (not recommended)
                padded_len = int(entry)
                logical_len = padded_len

            if padded_len < logical_len:
                raise ValueError(f"{field=} has {padded_len=} < {logical_len=}")

            # padded_len must be a multiple of 8 doubles (64 bytes)
            expected_padded = _padded_len_8doubles(logical_len)
            if padded_len != expected_padded:
                raise ValueError(
                    f"{field=} has {logical_len=} but {padded_len=}; expected padded_len={expected_padded}"
                )

            layout[field] = (offset, logical_len, padded_len)
            offset += padded_len

        # WHOLE covers the entire padded window extent
        if Field.WHOLE not in layout:
            total_logical = sum(layout[f][1] for f in layout.keys() if f != Field.WHOLE)
            layout[Field.WHOLE] = (0, total_logical, offset)

        self.buffer_layout = layout
        total_buffer_length = offset
        window_size_bytes = MPI.DOUBLE.size * total_buffer_length

        self.buffer_length = total_buffer_length
        self.window = MPI.Win.Allocate(window_size_bytes, MPI.DOUBLE.size, comm=strata_comm)

        # Bind numpy view to the window memory
        self.buff = np.ndarray(dtype="d", shape=(total_buffer_length,), buffer=self.window.tomemory())
        self.buff[:] = np.nan

        # Initialize ID slots (logical end) to 0.0
        for field, (off, logical_len, padded_len) in self.buffer_layout.items():
            if field == Field.WHOLE:
                continue
            self.buff[off + logical_len - 1] = 0.0

        # Gather layouts across ranks
        self.strata_buffer_layouts = strata_comm.allgather(self.buffer_layout)

    def free(self):
        if self.window is not None:
            self.window.Free()
            self.buff = None
            self.buffer_layout = None
            self.buffer_length = 0
            self.window = None
            self.strata_buffer_layouts = None
        return

    #### Functions ####
    def get(self, dest: nptyping.ArrayLike, strata_rank: int, field: Field):
        assert (0 <= strata_rank < len(self.strata_buffer_layouts))

        that_layout = self.strata_buffer_layouts[strata_rank]
        assert field in that_layout

        (offset, logical_len, padded_len) = that_layout[field]
        assert np.size(dest) == padded_len

        window = self.window
        window.Lock(strata_rank, MPI.LOCK_SHARED)
        window.Get((dest, padded_len, MPI.DOUBLE), strata_rank, offset)
        window.Unlock(strata_rank)
        return

    def put(self, values: nptyping.ArrayLike, field: Field):
        (offset, logical_len, padded_len) = self.buffer_layout[field]
        assert np.size(values) == padded_len

        window = self.window
        window.Lock(self.strata_rank, MPI.LOCK_EXCLUSIVE)
        window.Put((values, padded_len, MPI.DOUBLE), self.strata_rank, offset)
        window.Unlock(self.strata_rank)
        return

## End SPWindow
