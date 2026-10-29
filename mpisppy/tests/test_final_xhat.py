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
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mpisppy.extensions.extension import MultiExtension
from mpisppy.extensions.xhatclosest import XhatClosest
from mpisppy.extensions.xhatxbar import XhatXbar
from mpisppy.generic.extensions import configure_extensions
from mpisppy.generic.parsing import add_decomp_args
from mpisppy.opt.aph import APH
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

    def test_option_defaults_off_and_accepts_final_heuristics(self):
        cfg = _cfg()
        self.assertEqual(cfg.final_xhat, "none")
        for heuristic in ("xbar", "closest"):
            cfg.final_xhat = heuristic
            self.assertEqual(cfg.final_xhat, heuristic)
        with self.assertRaises(ValueError):
            cfg.final_xhat = "not-a-heuristic"

    def test_final_heuristics_parse_from_command_line(self):
        for heuristic in ("xbar", "closest"):
            with self.subTest(heuristic=heuristic):
                cfg = Config()
                cfg.final_xhat_args()
                with patch("sys.argv", ["prog", "--final-xhat", heuristic]):
                    cfg.parse_command_line("prog")
                self.assertEqual(cfg.final_xhat, heuristic)

    def test_none_does_not_attach_extension(self):
        hub = _hub()
        configure_extensions(hub, module=None, cfg=_cfg())
        self.assertIsNone(hub["opt_kwargs"]["extensions"])
        self.assertNotIn("xhat_xbar_options", hub["opt_kwargs"]["options"])

    def test_xbar_attaches_to_ph_hub_and_resolves_options_lazily(self):
        hub = _hub()
        configure_extensions(hub, module=None, cfg=_cfg("xbar"))
        self.assertIs(hub["opt_kwargs"]["extensions"], XhatXbar)
        xhat_options = hub["opt_kwargs"]["options"]["xhat_xbar_options"]
        self.assertIsNone(xhat_options["xhat_solver_options"])
        self.assertTrue(xhat_options["use_ph_solver_options"])
        self.assertTrue(xhat_options["keep_solution"])

    def test_closest_attaches_to_ph_hub_and_resolves_options_lazily(self):
        hub = _hub()
        configure_extensions(hub, module=None, cfg=_cfg("closest"))
        self.assertIs(hub["opt_kwargs"]["extensions"], XhatClosest)
        xhat_options = hub["opt_kwargs"]["options"]["xhat_closest_options"]
        self.assertIsNone(xhat_options["xhat_solver_options"])
        self.assertTrue(xhat_options["use_ph_solver_options"])
        self.assertTrue(xhat_options["keep_solution"])

    def test_closest_reads_final_effective_solver_options(self):
        effective_options = {"mipgap": 0.002, "timelimit": 17}
        opt = SimpleNamespace(
            cylinder_rank=0,
            n_proc=1,
            _PHIter=9,
            _effective_solver_options=Mock(return_value=effective_options),
            options={
                "verbose": False,
                "xhat_closest_options": {
                    "xhat_solver_options": None,
                    "use_ph_solver_options": True,
                },
            },
            scenario_names_to_rank={"ROOT": {"Scenario0": 0}},
        )
        extension = XhatClosest(opt)

        self.assertIs(
            extension._solver_options_for_evaluation(), effective_options)
        opt._effective_solver_options.assert_called_once_with(9)

    def test_xbar_composes_with_existing_extensions(self):
        class ExistingExtension:
            pass

        hub = _hub(extension=ExistingExtension)
        configure_extensions(hub, module=None, cfg=_cfg("xbar"))
        self.assertIs(hub["opt_kwargs"]["extensions"], MultiExtension)
        classes = hub["opt_kwargs"]["extension_kwargs"]["ext_classes"]
        self.assertEqual(classes, [ExistingExtension, XhatXbar])

    def test_xbar_attaches_to_aph_hub(self):
        hub = _hub(opt_class=APH)
        configure_extensions(hub, module=None, cfg=_cfg("xbar"))
        self.assertIs(hub["opt_kwargs"]["extensions"], XhatXbar)

    def test_xbar_rejects_non_ph_hub(self):
        class OtherAlgorithm:
            pass

        with self.assertRaisesRegex(RuntimeError, "only PH and APH hubs"):
            configure_extensions(
                _hub(opt_class=OtherAlgorithm), module=None, cfg=_cfg("xbar"))


if __name__ == "__main__":
    unittest.main()
