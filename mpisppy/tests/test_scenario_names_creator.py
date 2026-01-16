###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from unittest import TestCase

from mpisppy.utils import scenario_names_creator


class TestScenarioNamesCeator(TestCase):
    def test_scenario_names_creator_default(self):
        self.assertEqual(
            scenario_names_creator(3),
            ["scenario0", "scenario1", "scenario2"],
            "Should be a list with 'scenario0', 'scenario1' and 'scenario2'",
        )

    def test_scenario_names_creator_defining_prefix(self):
        self.assertEqual(
            scenario_names_creator(3, prefix="s"),
            ["s0", "s1", "s2"],
            "Should be a list with 's0', 's1' and 's2'",
        )

    def test_scenario_names_creator_setting_start_value(self):
        self.assertEqual(
            scenario_names_creator(3, start=1),
            ["scenario1", "scenario2", "scenario3"],
            "Should be a list with 'scenario1', 'scenario2' and 'scenario3'",
        )

    def test_scenario_names_creator_start_value_is_None(self):
        # to avoid migration errors from earlier implementations start is also allowed to be None
        self.assertEqual(
            scenario_names_creator(3, start=None),
            ["scenario0", "scenario1", "scenario2"],
            "Should be a list with 'scenario10', 'scenario1' and 'scenario2'",
        )
