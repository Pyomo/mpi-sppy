###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Tests for solver_options_layers. The contract this file pins is:
# the layered representation folds (per-iteration) to dicts
# identical to the legacy iter0_solver_options / iterk_solver_options
# dicts produced by shared_options and apply_solver_specs. Later
# changes to per-spoke merge semantics will lean on this equivalence.

import copy
import unittest

from mpisppy.utils import config
from mpisppy.utils.cfg_vanilla import shared_options, apply_solver_specs
from mpisppy.utils.sputils import (
    fold_solver_options_layers,
    solver_options_layer,
)


def _bare_cfg():
    cfg = config.Config()
    cfg.popular_args()
    cfg.add_mipgap_specs()
    return cfg


def _spoke_cfg(spoke_name):
    cfg = _bare_cfg()
    cfg.add_solver_specs(prefix=spoke_name)
    cfg.add_mipgap_specs(prefix=spoke_name)
    return cfg


class TestSharedOptionsLayers(unittest.TestCase):

    def test_empty_cfg_yields_empty_layers(self):
        cfg = _bare_cfg()
        sh = shared_options(cfg)
        self.assertEqual(sh["solver_options_layers"], [])
        self.assertEqual(
            fold_solver_options_layers(sh["solver_options_layers"], 0), {})
        self.assertEqual(
            fold_solver_options_layers(sh["solver_options_layers"], 1), {})

    def test_solver_options_string_is_default_layer(self):
        cfg = _bare_cfg()
        cfg.solver_options = "mipgap=0.01 logfile=run.log"
        sh = shared_options(cfg)
        layers = sh["solver_options_layers"]
        expected = {"mipgap": 0.01, "logfile": "run.log"}
        self.assertEqual(fold_solver_options_layers(layers, 0), expected)
        self.assertEqual(fold_solver_options_layers(layers, 1), expected)
        self.assertEqual(fold_solver_options_layers(layers, 7), expected)

    def test_iter0_iterk_mipgap_yield_predicate_layers(self):
        cfg = _bare_cfg()
        cfg.iter0_mipgap = 0.01
        cfg.iterk_mipgap = 0.02
        sh = shared_options(cfg)
        layers = sh["solver_options_layers"]
        self.assertEqual(
            fold_solver_options_layers(layers, 0), {"mipgap": 0.01})
        self.assertEqual(
            fold_solver_options_layers(layers, 1), {"mipgap": 0.02})
        self.assertEqual(
            fold_solver_options_layers(layers, 5), {"mipgap": 0.02})

    def test_max_solver_threads_overrides_solver_options_threads(self):
        # Mirrors today's behavior at cfg_vanilla.py:83-85: the global
        # thread cap overwrites whatever 'threads' the user wrote inline.
        cfg = _bare_cfg()
        cfg.solver_options = "mipgap=0.01 threads=2"
        cfg.max_solver_threads = 4
        sh = shared_options(cfg)
        layers = sh["solver_options_layers"]
        folded = fold_solver_options_layers(layers, 0)
        self.assertEqual(folded["mipgap"], 0.01)
        self.assertEqual(folded["threads"], 4)

    def test_combined_fold_equals_legacy_iter0_iterk_dicts(self):
        # Regression contract: under any cfg combination,
        # fold(layers, 0) must match iter0_solver_options and
        # fold(layers, k>=1) must match iterk_solver_options.
        cfg = _bare_cfg()
        cfg.solver_options = "logfile=run.log threads=2"
        cfg.max_solver_threads = 4
        cfg.iter0_mipgap = 0.01
        cfg.iterk_mipgap = 0.02
        sh = shared_options(cfg)
        layers = sh["solver_options_layers"]
        self.assertEqual(
            fold_solver_options_layers(layers, 0),
            sh["iter0_solver_options"],
        )
        self.assertEqual(
            fold_solver_options_layers(layers, 1),
            sh["iterk_solver_options"],
        )
        self.assertEqual(
            fold_solver_options_layers(layers, 7),
            sh["iterk_solver_options"],
        )


class TestApplySolverSpecsLayers(unittest.TestCase):

    def _spoke_dict_from(self, sh):
        # apply_solver_specs operates on spoke["opt_kwargs"]["options"];
        # it expects a deepcopy of shared_options output.
        return {"opt_kwargs": {"options": copy.deepcopy(sh)}}

    def test_per_spoke_solver_options_replace_layers(self):
        # Today's per-spoke semantics are replace-not-overlay in
        # apply_solver_specs: each --{name}-solver-options call
        # discards the global --solver-options dict. A later phase
        # will change this to overlay; this test pins the legacy
        # contract until then.
        cfg = _spoke_cfg("lagrangian")
        cfg.solver_options = "logfile=run.log"
        cfg.lagrangian_solver_options = "mipgap=0.001"
        sh = shared_options(cfg)
        spoke = self._spoke_dict_from(sh)
        apply_solver_specs("lagrangian", spoke, cfg)
        opts = spoke["opt_kwargs"]["options"]
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 0),
            opts["iter0_solver_options"],
        )
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 1),
            opts["iterk_solver_options"],
        )
        self.assertEqual(opts["iter0_solver_options"], {"mipgap": 0.001})

    def test_per_spoke_iter0_iterk_mipgap_layer_predicate(self):
        cfg = _spoke_cfg("lagrangian")
        cfg.solver_options = "logfile=run.log"
        cfg.lagrangian_iter0_mipgap = 0.01
        cfg.lagrangian_iterk_mipgap = 0.02
        sh = shared_options(cfg)
        spoke = self._spoke_dict_from(sh)
        apply_solver_specs("lagrangian", spoke, cfg)
        opts = spoke["opt_kwargs"]["options"]
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 0),
            opts["iter0_solver_options"],
        )
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 1),
            opts["iterk_solver_options"],
        )

    def test_per_spoke_threads_reapplied(self):
        # apply_solver_specs re-applies max_solver_threads at the end
        # (cfg_vanilla.py:127-129). The layered version must produce
        # the same final folded dict for both predicates.
        cfg = _spoke_cfg("lagrangian")
        cfg.solver_options = "mipgap=0.01"
        cfg.lagrangian_solver_options = "presolve=1"
        cfg.max_solver_threads = 8
        sh = shared_options(cfg)
        spoke = self._spoke_dict_from(sh)
        apply_solver_specs("lagrangian", spoke, cfg)
        opts = spoke["opt_kwargs"]["options"]
        self.assertEqual(opts["iter0_solver_options"].get("threads"), 8)
        self.assertEqual(opts["iterk_solver_options"].get("threads"), 8)
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 0),
            opts["iter0_solver_options"],
        )
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 1),
            opts["iterk_solver_options"],
        )


class TestPredicateValidation(unittest.TestCase):

    def test_valid_predicates_accepted(self):
        for when in ("default", "iter0", "iterk",
                     ("after_iter", 0), ("after_iter", 5)):
            solver_options_layer(when, {"mipgap": 0.01})

    def test_unknown_string_predicate_rejected(self):
        with self.assertRaises(ValueError):
            solver_options_layer("sometimes", {})

    def test_after_iter_negative_rejected(self):
        with self.assertRaises(ValueError):
            solver_options_layer(("after_iter", -1), {})

    def test_after_iter_non_int_rejected(self):
        with self.assertRaises(ValueError):
            solver_options_layer(("after_iter", 1.5), {})
        with self.assertRaises(ValueError):
            solver_options_layer(("after_iter", "5"), {})

    def test_after_iter_bool_rejected(self):
        # bool is an int subclass — guard against (after_iter, True)
        with self.assertRaises(ValueError):
            solver_options_layer(("after_iter", True), {})

    def test_fold_rejects_invalid_predicate(self):
        # Hand-built layer (bypasses solver_options_layer) is still validated
        # by fold time.
        bad_layers = [{"when": "huh", "options": {}}]
        with self.assertRaises(ValueError):
            fold_solver_options_layers(bad_layers, 0)


class TestEffectiveSolverOptions(unittest.TestCase):
    """The PHBase consumption contract.

    Pins three properties of PHBase._effective_solver_options(k):
      1. It folds solver_options_layers in list order, picking layers
         whose "when" predicate matches k and last-write-wins per key.
      2. current_solver_options is overlaid last (so Gapper auto-mode
         writes to current_solver_options surface in the solve).
      3. after_iter layers fire on iterations >= N, matching the
         per-iter mipgap semantics --mipgaps-json now produces via
         add_gapper.
    """

    @staticmethod
    def _effective(layers, current, k):
        """Call PHBase._effective_solver_options against a fake.

        Avoids constructing a full PHBase (which needs scenarios) by
        invoking the unbound method on a duck-typed object.
        """
        from mpisppy.phbase import PHBase
        fake = type("FakePH", (), {})()
        fake.solver_options_layers = layers
        fake.current_solver_options = current
        return PHBase._effective_solver_options(fake, k)

    def test_iter0_iterk_layers_fold_per_k(self):
        layers = [
            solver_options_layer("iter0", {"mipgap": 0.01}),
            solver_options_layer("iterk", {"mipgap": 0.001}),
        ]
        self.assertEqual(
            self._effective(layers, {}, 0), {"mipgap": 0.01})
        self.assertEqual(
            self._effective(layers, {}, 1), {"mipgap": 0.001})
        self.assertEqual(
            self._effective(layers, {}, 5), {"mipgap": 0.001})

    def test_current_solver_options_overlays_last(self):
        # Gapper auto-mode writes to current_solver_options and that
        # write must surface in the next solve, on top of layer fold.
        layers = [solver_options_layer("default", {"mipgap": 0.1})]
        current = {"mipgap": 0.001}  # Gapper-style override
        self.assertEqual(
            self._effective(layers, current, 0), {"mipgap": 0.001})
        self.assertEqual(
            self._effective(layers, current, 5), {"mipgap": 0.001})

    def test_after_iter_layers_match_mipgaps_json_semantics(self):
        # cfg_vanilla.add_gapper turns --mipgaps-json {0: G0, 5: G5,
        # 10: G10} into a list of after_iter layers; assert the fold
        # gives the expected per-iter mipgap.
        layers = [
            solver_options_layer(("after_iter", 0), {"mipgap": 0.10}),
            solver_options_layer(("after_iter", 5), {"mipgap": 0.01}),
            solver_options_layer(("after_iter", 10), {"mipgap": 0.005}),
        ]
        self.assertEqual(self._effective(layers, {}, 0)["mipgap"], 0.10)
        self.assertEqual(self._effective(layers, {}, 4)["mipgap"], 0.10)
        self.assertEqual(self._effective(layers, {}, 5)["mipgap"], 0.01)
        self.assertEqual(self._effective(layers, {}, 9)["mipgap"], 0.01)
        self.assertEqual(self._effective(layers, {}, 10)["mipgap"], 0.005)


class TestAddGapperMipgapsJsonLayers(unittest.TestCase):
    """add_gapper's static-schedule path now appends after_iter
    layers and skips the Gapper extension. The resulting per-iter
    mipgap behavior is pinned by
    TestEffectiveSolverOptions.test_after_iter_layers_match_mipgaps_json_semantics
    above; this class pins the cfg_vanilla wiring that produces
    those layers from the JSON file.
    """

    def _hub_dict(self):
        return {
            "opt_kwargs": {
                "options": {"solver_options_layers": []},
            },
        }

    def _write_mipgaps_json(self, schedule):
        import json
        import tempfile
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False).name
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in schedule.items()}, f)
        return path

    def test_mipgaps_json_appends_after_iter_layers(self):
        import os
        from mpisppy.utils.cfg_vanilla import add_gapper
        cfg = _bare_cfg()
        cfg.add_to_config(
            "mipgaps_json", description="x", domain=str, default=None)
        cfg.add_to_config(
            "starting_mipgap", description="x", domain=float, default=None)
        cfg.add_to_config(
            "mipgap_ratio", description="x", domain=float, default=None)
        path = self._write_mipgaps_json({0: 0.10, 5: 0.01, 10: 0.005})
        try:
            cfg.mipgaps_json = path
            hub_dict = self._hub_dict()
            add_gapper(hub_dict, cfg)
            layers = hub_dict["opt_kwargs"]["options"]["solver_options_layers"]
            # 3 after_iter layers in ascending-N order
            self.assertEqual(len(layers), 3)
            self.assertEqual(layers[0]["when"], ("after_iter", 0))
            self.assertEqual(layers[1]["when"], ("after_iter", 5))
            self.assertEqual(layers[2]["when"], ("after_iter", 10))
            self.assertEqual(
                [layer["options"] for layer in layers],
                [{"mipgap": 0.10}, {"mipgap": 0.01}, {"mipgap": 0.005}],
            )
            # No Gapper extension was wired (gapperoptions not added)
            self.assertNotIn(
                "gapperoptions", hub_dict["opt_kwargs"]["options"])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
