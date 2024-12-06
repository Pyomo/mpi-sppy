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
from mpisppy.cylinders.reduced_costs_spoke import ReducedCostsSpoke 

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

    def initialize_spoke_indices(self):
        for (i, spoke) in enumerate(self.opt.spcomm.spokes):
            if spoke["spoke_class"] == ReducedCostsSpoke:
                self.reduced_costs_spoke_index = i + 1
        if self.reduced_costs_spoke_index is None:
            raise RuntimeError("ReducedCostsRho requires a ReducedCostsSpoke for calculations")

    def _get_serial_number(self):
        return int(round(self.opt.spcomm.outerbound_receive_buffers[self.reduced_costs_spoke_index][-1]))

    def _get_reduced_costs_from_spoke(self):
        return self.opt.spcomm.outerbound_receive_buffers[self.reduced_costs_spoke_index][1+self.nonant_length:1+self.nonant_length+len(self._scenario_rc_buffer)]

    def sync_with_spokes(self):
        serial_number = self._get_serial_number()
        if serial_number > self._last_serial_number:
            self._last_serial_number = serial_number
            self._scenario_rc_buffer[:] = self._get_reduced_costs_from_spoke()
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
        if self._get_serial_number() == 0:
            while not self.ph.spcomm.hub_from_spoke(self.opt.spcomm.outerbound_receive_buffers[self.reduced_costs_spoke_index], self.reduced_costs_spoke_index):
                continue
            self.sync_with_spokes()
        self.compute_and_update_rho()
