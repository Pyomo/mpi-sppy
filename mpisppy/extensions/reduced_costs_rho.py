###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import numpy as np
from mpisppy import global_toc
from mpisppy.extensions.sensi_rho import _SensiRhoBase

from mpisppy.cylinders.spwindow import Field

class ReducedCostsRho(_SensiRhoBase):
    """
    Rho determination algorithm using nonant reduced costs,
    calculated in a ReducedCostsSpoke
    """

    def __init__(self, ph, comm=None):
        super().__init__(ph, comm=comm)
        self.ph = ph

        self.multiplier = 1.0
        self.verbose = ph.options["verbose"]

        if (
            "reduced_costs_rho_options" in ph.options
            and "multiplier" in ph.options["reduced_costs_rho_options"]
        ):
            self.multiplier = ph.options["reduced_costs_rho_options"]["multiplier"]
        self.cfg = ph.options["reduced_costs_rho_options"]["cfg"]

        scenario_buffer_len = 0
        for s in ph.local_scenarios.values():
            scenario_buffer_len += len(s._mpisppy_data.nonant_indices)
        self._scenario_rc_buffer = np.empty(scenario_buffer_len)
        self._scenario_rc_buffer.fill(np.nan)

        self._last_serial_number = -1
        self.reduced_costs_spoke_index = None

    def register_receive_fields(self):
        spcomm = self.opt.spcomm
        reduced_cost_ranks = spcomm.fields_to_ranks[Field.SCENARIO_REDUCED_COST]
        assert len(reduced_cost_ranks) == 1
        self.reduced_costs_spoke_index = reduced_cost_ranks[0]

        self.scenario_reduced_cost_buf = spcomm.register_recv_field(
            Field.SCENARIO_REDUCED_COST,
            self.reduced_costs_spoke_index,
        )

    def sync_with_spokes(self):
        self.opt.spcomm.get_receive_buffer(
            self.scenario_reduced_cost_buf,
            Field.SCENARIO_REDUCED_COST,
            self.reduced_costs_spoke_index,
        )
        if self.scenario_reduced_cost_buf.is_new():
            self._scenario_rc_buffer[:] = self.scenario_reduced_cost_buf.value_array()
            # print(f"In ReducedCostsRho; {self._scenario_rc_buffer=}")
        else:
            if self.opt.cylinder_rank == 0 and self.verbose:
                print("No new reduced costs!")
        # These may be `nan` if the nonant is fixed
        self._scenario_rc_buffer = np.nan_to_num(self._scenario_rc_buffer, copy=False)

    def pre_iter0(self):
        self.nonant_length = self.opt.nonant_length

    def post_iter0(self):
        # need to wait for spoke sync
        pass

    def get_nonant_sensitivites(self):
        # dict of dicts [s][ndn_i]
        nonant_sensis = {}
        ci = 0
        for s in self.ph.local_subproblems.values():
            nonant_sensis[s] = {}
            for ndn_i in s._mpisppy_data.nonant_indices:
                nonant_sensis[s][ndn_i] = self._scenario_rc_buffer[ci]
                ci += 1
        # print(f"{nonant_sensis=}")
        return nonant_sensis

    def post_iter0_after_sync(self):
        global_toc("Using reduced cost rho setter")
        self.update_caches()
        # wait until the spoke has data
        if self.scenario_reduced_cost_buf.id() == 0:
            while not self.ph.spcomm.get_receive_buffer(self.scenario_reduced_cost_buf, Field.SCENARIO_REDUCED_COST, self.reduced_costs_spoke_index):
                continue
            self.sync_with_spokes()
        self.compute_and_update_rho()
