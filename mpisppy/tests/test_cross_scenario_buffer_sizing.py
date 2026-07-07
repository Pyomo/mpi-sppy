###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
Regression test for issue #727: the CROSS_SCENARIO_COST and CROSS_SCENARIO_CUT
window buffers must be sized consistently when a cylinder spans more than one
rank, i.e. when the number of *local* scenarios on a rank is smaller than the
*global* scenario count.

The bug was that FieldLengths set ``_total_number_scenarios`` to the local
scenario count, so:

  * CROSS_SCENARIO_COST (logically nscen * local) was sized local * local, and
  * CROSS_SCENARIO_CUT (logically nscen * (nonants + 2)) was sized
    local * (nonants + 2),

while the CrossScenarioExtension sender advertised nscen * nscen for the cost
field and the spoke wrote nscen rows into the cut buffer. All three sizings
only agree when local == nscen (one rank per cylinder), which is why the bug
was invisible in single-rank-per-cylinder runs.

These checks are deterministic and serial: they drive the length computation
directly with a stand-in ``opt`` for which local < nscen, so they pin the
sizes without needing an MPI window. They fail on the pre-fix code (which
yields local*local and nscen*nscen) and pass once every site agrees on
nscen * local.
"""

import types
import unittest

from mpisppy.cylinders.spwindow import Field, FieldLengths
from mpisppy.extensions.cross_scen_extension import CrossScenarioExtension

# local < nscen: the regime where the three sizings used to diverge.
NSCEN = 5
LOCAL = 2
NONANTS_PER_SCEN = 3


def _fake_opt_for_field_lengths(nscen, local, nonants_per_scen):
    """Minimal stand-in carrying exactly what FieldLengths.__init__ reads."""
    def _scen():
        # nonant_indices only needs a length here.
        return types.SimpleNamespace(
            _mpisppy_data=types.SimpleNamespace(
                nonant_indices=list(range(nonants_per_scen))
            )
        )

    return types.SimpleNamespace(
        local_scenarios={f"Scen{i}": _scen() for i in range(local)},
        all_scenario_names=[f"Scen{i}" for i in range(nscen)],
        # nonant_length is the global (per-scenario, first-stage) nonant count.
        nonant_length=nonants_per_scen,
    )


class _RecordingSpcomm:
    """Records register_send_field lengths so the sender can be inspected."""

    def __init__(self):
        self.registered = {}

    def is_send_field_registered(self, field):
        return field in self.registered

    def register_send_field(self, field, length):
        self.registered[field] = length
        return [0.0] * length


def _fake_opt_for_extension(nscen, local, nonants_per_scen):
    return types.SimpleNamespace(
        multistage=False,
        options={},
        spcomm=_RecordingSpcomm(),
        all_scenario_names=[f"Scen{i}" for i in range(nscen)],
        local_scenario_names=[f"Scen{i}" for i in range(local)],
        nonant_length=nonants_per_scen,
    )


class TestCrossScenarioBufferSizing(unittest.TestCase):

    def setUp(self):
        self.fl = FieldLengths(
            _fake_opt_for_field_lengths(NSCEN, LOCAL, NONANTS_PER_SCEN)
        )

    def test_cross_scenario_cost_is_nscen_times_local(self):
        # one eta per global scenario, for each local scenario
        self.assertEqual(self.fl[Field.CROSS_SCENARIO_COST], NSCEN * LOCAL)
        # guard: the pre-fix sizing was local*local; with local < nscen these
        # differ, so this assertion would have caught the bug.
        self.assertNotEqual(NSCEN * LOCAL, LOCAL * LOCAL)

    def test_cross_scenario_cut_is_global_sized(self):
        # one cut row [const, eta_coef, *nonant_coefs] per global scenario
        row_len = NONANTS_PER_SCEN + 1 + 1
        self.assertEqual(self.fl[Field.CROSS_SCENARIO_CUT], NSCEN * row_len)
        # guard: the pre-fix sizing was local * row_len.
        self.assertNotEqual(NSCEN * row_len, LOCAL * row_len)

    def test_sender_cost_registration_matches_receiver_default(self):
        opt = _fake_opt_for_extension(NSCEN, LOCAL, NONANTS_PER_SCEN)
        ext = CrossScenarioExtension(opt)
        ext.register_send_fields()

        registered = opt.spcomm.registered
        # The sender (extension) and the receiver default (FieldLengths) must
        # agree, both at nscen * local -- not the pre-fix nscen * nscen.
        self.assertEqual(
            registered[Field.CROSS_SCENARIO_COST], NSCEN * LOCAL
        )
        self.assertEqual(
            registered[Field.CROSS_SCENARIO_COST],
            self.fl[Field.CROSS_SCENARIO_COST],
        )
        # the nonant send field is local-sized as before
        self.assertEqual(
            registered[Field.NONANTS_VALS], LOCAL * NONANTS_PER_SCEN
        )


if __name__ == "__main__":
    unittest.main()
