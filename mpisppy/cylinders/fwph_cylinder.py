###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
from mpisppy.cylinders.spwindow import Field
from mpisppy.cylinders.spcommunicator import SPCommunicator, RecvCircularBuffer

class FWPH_Cylinder(SPCommunicator):

    send_fields = ()
    receive_fields = (Field.BEST_XHAT, Field.RECENT_XHATS, )

    def register_receive_fields(self):
        super().register_receive_fields()
        self._recent_xhat_recv_circular_buffers = [
            (idx, cls, RecvCircularBuffer(
                    recv_buf,
                    self._field_lengths[Field.BEST_XHAT],
                    self._field_lengths[Field.RECENT_XHATS] // self._field_lengths[Field.BEST_XHAT],
                )
            )
            for (idx, cls, recv_buf) in self.receive_field_spcomms[Field.RECENT_XHATS] 
        ]

    def add_cylinder_columns(self):
        self.add_best_xhat_column()
        self.add_recent_xhat_columns()
        # self.add_fwph_sdm_column()

    def add_best_xhat_column(self):
        for idx, cls, recv_buf in self.receive_field_spcomms[Field.BEST_XHAT]:
            is_new = self.get_receive_buffer(recv_buf, Field.BEST_XHAT, idx)
            if is_new:
                self._add_QP_columns_from_buf(recv_buf.value_array())

    def add_recent_xhat_columns(self):
        for idx, cls, recv_buf_circular in self._recent_xhat_recv_circular_buffers:
            is_new = self.get_receive_buffer(recv_buf_circular.data, Field.RECENT_XHATS, idx)
            if is_new:
                for value_array in recv_buf_circular.next_value_arrays():
                    self._add_QP_columns_from_buf(value_array)

    def _add_QP_columns_from_buf(self, xhat_array):
        self.opt._save_nonants()
        inner_bound_cache = self._cache_inner_bounds()
        # NOTE: this does not work with "loose" bundles
        # print(f"{self.cylinder_rank=} got {xhat_array=}")
        ci = 0
        for k,s in self.opt.local_scenarios.items():
            for ndn_var in s._mpisppy_data.nonant_indices.values():
                ndn_var._value = xhat_array[ci]
                ci += 1
            s._mpisppy_data.inner_bound = xhat_array[ci]
            ci += 1
            self.opt._add_QP_column(k, disable_W=True)

        self._restore_inner_bounds(inner_bound_cache)
        self.opt._restore_nonants(update_persistent=False)

    def _cache_inner_bounds(self):
        return {s : s._mpisppy_data.inner_bound for s in self.opt.local_scenarios.values()}

    def _restore_inner_bounds(self, inner_bound_cache):
        for s in self.opt.local_scenarios.values():
            s._mpisppy_data.inner_bound = inner_bound_cache[s]
