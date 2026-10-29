###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Configuration and hub wiring tests for --final-xhat."""

import unittest
from unittest.mock import patch

from mpisppy.extensions.extension import MultiExtension
from mpisppy.extensions.xhatxbar import XhatXbar
from mpisppy.generic.extensions import configure_extensions
from mpisppy.generic.parsing import add_decomp_args
from mpisppy.opt.ph import PH
from mpisppy.utils.config import Config


def _cfg(final_xhat="none"):
    cfg = Config()
    add_decomp_args(cfg)
    # configure_extensions also consumes this driver-level option, which is
    # registered outside add_decomp_args by add_driver_args in production.
    cfg.add_to_config("write_scenario_lp_mps_files_dir", description="",
                      domain=str, default=None)
    cfg.final_xhat = final_xhat
    return cfg


def _hub(opt_class=PH, extension=None, extension_kwargs=None):
    return {
        "opt_class": opt_class,
        "opt_kwargs": {
            "options": {"iterk_solver_options": {"mipgap": 0.01}},
            "extensions": extension,
            "extension_kwargs": extension_kwargs,
        },
    }


class TestFinalXhat(unittest.TestCase):

    def test_option_defaults_off_and_accepts_xbar(self):
        cfg = _cfg()
        self.assertEqual(cfg.final_xhat, "none")
        cfg.final_xhat = "xbar"
        self.assertEqual(cfg.final_xhat, "xbar")
        with self.assertRaises(ValueError):
            cfg.final_xhat = "not-a-heuristic"

    def test_xbar_parses_from_command_line(self):
        cfg = Config()
        cfg.final_xhat_args()
        with patch("sys.argv", ["prog", "--final-xhat", "xbar"]):
            cfg.parse_command_line("prog")
        self.assertEqual(cfg.final_xhat, "xbar")

    def test_none_does_not_attach_extension(self):
        hub = _hub()
        configure_extensions(hub, module=None, cfg=_cfg())
        self.assertIsNone(hub["opt_kwargs"]["extensions"])
        self.assertNotIn("xhat_xbar_options", hub["opt_kwargs"]["options"])

    def test_xbar_attaches_to_ph_hub_and_reuses_iterk_options(self):
        hub = _hub()
        configure_extensions(hub, module=None, cfg=_cfg("xbar"))
        self.assertIs(hub["opt_kwargs"]["extensions"], XhatXbar)
        xhat_options = hub["opt_kwargs"]["options"]["xhat_xbar_options"]
        self.assertIs(
            xhat_options["xhat_solver_options"],
            hub["opt_kwargs"]["options"]["iterk_solver_options"],
        )
        self.assertTrue(xhat_options["keep_solution"])

    def test_xbar_composes_with_existing_extensions(self):
        class ExistingExtension:
            pass

        hub = _hub(extension=ExistingExtension)
        configure_extensions(hub, module=None, cfg=_cfg("xbar"))
        self.assertIs(hub["opt_kwargs"]["extensions"], MultiExtension)
        classes = hub["opt_kwargs"]["extension_kwargs"]["ext_classes"]
        self.assertEqual(classes, [ExistingExtension, XhatXbar])

    def test_xbar_rejects_non_ph_hub(self):
        class OtherAlgorithm:
            pass

        with self.assertRaisesRegex(RuntimeError, "synchronous PH hub"):
            configure_extensions(
                _hub(opt_class=OtherAlgorithm), module=None, cfg=_cfg("xbar"))


if __name__ == "__main__":
    unittest.main()
