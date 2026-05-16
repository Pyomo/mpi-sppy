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

    def test_per_spoke_solver_options_overlay_layers(self):
        # Per-spoke option specs overlay on top of the global
        # --solver-options dict: keys named by the spoke flag win,
        # keys it doesn't mention survive from the global flag. A
        # spoke's logfile, presolve flag, etc. set globally should
        # still be in the spoke's effective dict.
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
        self.assertEqual(
            opts["iter0_solver_options"],
            {"logfile": "run.log", "mipgap": 0.001},
        )

    def test_per_spoke_overlay_combines_global_and_spoke_keys(self):
        # The worked example from the design doc: global supplies
        # presolve+threads, the spoke flag supplies mipgap, and the
        # lagrangian-spoke effective dict is the union of all three.
        cfg = _spoke_cfg("lagrangian")
        cfg.solver_options = "presolve=2 threads=4"
        cfg.lagrangian_solver_options = "mipgap=0.01"
        sh = shared_options(cfg)
        spoke = self._spoke_dict_from(sh)
        apply_solver_specs("lagrangian", spoke, cfg)
        opts = spoke["opt_kwargs"]["options"]
        expected = {"presolve": 2, "threads": 4, "mipgap": 0.01}
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 0),
            expected,
        )
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 1),
            expected,
        )

    def test_per_spoke_overlay_overrides_shared_key(self):
        # When the spoke flag overrides a key the global flag already
        # set, the spoke's value wins (last write in the fold) but the
        # other global keys survive.
        cfg = _spoke_cfg("lagrangian")
        cfg.solver_options = "mipgap=0.01 logfile=run.log"
        cfg.lagrangian_solver_options = "mipgap=0.001"
        sh = shared_options(cfg)
        spoke = self._spoke_dict_from(sh)
        apply_solver_specs("lagrangian", spoke, cfg)
        opts = spoke["opt_kwargs"]["options"]
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 0),
            {"logfile": "run.log", "mipgap": 0.001},
        )

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
                     ("starting_at_iter", 1), ("starting_at_iter", 5)):
            solver_options_layer(when, {"mipgap": 0.01})

    def test_starting_at_iter_zero_rejected(self):
        # N=0 would silently outrank iter0/iterk in axis-1
        # specificity; reject with a hint pointing at `default`.
        with self.assertRaisesRegex(ValueError, "default"):
            solver_options_layer(("starting_at_iter", 0), {"mipgap": 0.01})

    def test_unknown_string_predicate_rejected(self):
        with self.assertRaises(ValueError):
            solver_options_layer("sometimes", {})

    def test_starting_at_iter_negative_rejected(self):
        with self.assertRaises(ValueError):
            solver_options_layer(("starting_at_iter", -1), {})

    def test_starting_at_iter_non_int_rejected(self):
        with self.assertRaises(ValueError):
            solver_options_layer(("starting_at_iter", 1.5), {})
        with self.assertRaises(ValueError):
            solver_options_layer(("starting_at_iter", "5"), {})

    def test_starting_at_iter_bool_rejected(self):
        # bool is an int subclass — guard against (starting_at_iter, True)
        with self.assertRaises(ValueError):
            solver_options_layer(("starting_at_iter", True), {})

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
      3. starting_at_iter layers fire on iterations >= N, matching the
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

    def test_starting_at_iter_layers_match_mipgaps_json_semantics(self):
        # cfg_vanilla.add_gapper turns --mipgaps-json {0: G0, 5: G5,
        # 10: G10} into a list of layers; the N=0 entry is rendered
        # as a `default` layer (since `starting_at_iter:0` is
        # rejected by the validator), the rest as starting_at_iter:N.
        # Assert the fold gives the expected per-iter mipgap.
        layers = [
            solver_options_layer("default", {"mipgap": 0.10}),
            solver_options_layer(("starting_at_iter", 5), {"mipgap": 0.01}),
            solver_options_layer(("starting_at_iter", 10), {"mipgap": 0.005}),
        ]
        self.assertEqual(self._effective(layers, {}, 0)["mipgap"], 0.10)
        self.assertEqual(self._effective(layers, {}, 4)["mipgap"], 0.10)
        self.assertEqual(self._effective(layers, {}, 5)["mipgap"], 0.01)
        self.assertEqual(self._effective(layers, {}, 9)["mipgap"], 0.01)
        self.assertEqual(self._effective(layers, {}, 10)["mipgap"], 0.005)


class TestAddGapperMipgapsJsonLayers(unittest.TestCase):
    """add_gapper's static-schedule path now appends starting_at_iter
    layers and skips the Gapper extension. The resulting per-iter
    mipgap behavior is pinned by
    TestEffectiveSolverOptions.test_starting_at_iter_layers_match_mipgaps_json_semantics
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

    def test_mipgaps_json_appends_starting_at_iter_layers(self):
        import os
        import warnings
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
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                add_gapper(hub_dict, cfg)
            self.assertTrue(
                any(issubclass(w.category, DeprecationWarning)
                    and "--mipgaps-json" in str(w.message) for w in caught),
                f"Expected DeprecationWarning naming --mipgaps-json; "
                f"got {[(w.category.__name__, str(w.message)) for w in caught]}",
            )
            layers = hub_dict["opt_kwargs"]["options"]["solver_options_layers"]
            # 3 layers: the N=0 entry is rendered as `default`,
            # subsequent entries as starting_at_iter:N.
            self.assertEqual(len(layers), 3)
            self.assertEqual(layers[0]["when"], "default")
            self.assertEqual(layers[1]["when"], ("starting_at_iter", 5))
            self.assertEqual(layers[2]["when"], ("starting_at_iter", 10))
            self.assertEqual(
                [layer["options"] for layer in layers],
                [{"mipgap": 0.10}, {"mipgap": 0.01}, {"mipgap": 0.005}],
            )
            # No Gapper extension was wired (gapperoptions not added)
            self.assertNotIn(
                "gapperoptions", hub_dict["opt_kwargs"]["options"])
        finally:
            os.unlink(path)


class TestTranslateSolverOptions(unittest.TestCase):
    """translate_solver_options renames mpi-sppy's canonical option
    keys (currently mipgap and threads) to the target solver's
    native key, and passes everything else through unchanged.
    Stored options remain solver-agnostic; translation is the very
    last step before the keys hit the Pyomo solver plugin.
    """

    def _t(self, opts, solver_name):
        from mpisppy.utils.sputils import translate_solver_options
        return translate_solver_options(opts, solver_name)

    # --- edge cases ---

    def test_none_opts_returns_none(self):
        self.assertIsNone(self._t(None, "highs"))

    def test_empty_opts_returns_empty(self):
        self.assertEqual(self._t({}, "highs"), {})

    def test_none_solver_name_is_passthrough(self):
        opts = {"mipgap": 0.01, "threads": 4}
        self.assertEqual(self._t(opts, None), opts)

    def test_empty_solver_name_is_passthrough(self):
        opts = {"mipgap": 0.01, "threads": 4}
        self.assertEqual(self._t(opts, ""), opts)

    def test_no_canonical_keys_is_passthrough(self):
        opts = {"presolve": 2, "logfile": "run.log"}
        self.assertEqual(self._t(opts, "highs"), opts)
        self.assertEqual(self._t(opts, "gurobi"), opts)

    def test_input_dict_not_mutated(self):
        opts = {"mipgap": 0.01}
        _ = self._t(opts, "highs")
        self.assertEqual(opts, {"mipgap": 0.01})

    # --- per-solver translation correctness ---

    def test_cplex_needs_no_rename(self):
        for name in ("cplex", "cplex_persistent"):
            self.assertEqual(
                self._t({"mipgap": 0.01, "threads": 4}, name),
                {"mipgap": 0.01, "threads": 4})

    def test_xpress_needs_no_rename(self):
        for name in ("xpress", "xpress_persistent"):
            self.assertEqual(
                self._t({"mipgap": 0.01, "threads": 4}, name),
                {"mipgap": 0.01, "threads": 4})

    def test_gurobi_renames_threads(self):
        for name in ("gurobi", "gurobi_persistent", "appsi_gurobi"):
            self.assertEqual(
                self._t({"mipgap": 0.01, "threads": 4}, name),
                {"mipgap": 0.01, "Threads": 4},
                f"failed for solver_name={name!r}")

    def test_highs_renames_mipgap(self):
        for name in ("highs", "appsi_highs"):
            self.assertEqual(
                self._t({"mipgap": 0.01, "threads": 4}, name),
                {"mip_rel_gap": 0.01, "threads": 4},
                f"failed for solver_name={name!r}")

    # --- collision rule ---

    def test_highs_collision_keeps_native_drops_canonical(self):
        # User passed mip_rel_gap directly *and* --iter0-mipgap turned
        # into a layer with mipgap; the native key wins and the
        # canonical one is dropped (so the solver doesn't see both).
        result = self._t(
            {"mipgap": 0.01, "mip_rel_gap": 0.0001}, "highs")
        self.assertEqual(result, {"mip_rel_gap": 0.0001})

    def test_gurobi_collision_keeps_native_drops_canonical(self):
        result = self._t(
            {"threads": 4, "Threads": 1}, "gurobi")
        self.assertEqual(result, {"Threads": 1})

    # --- pass-through of non-canonical solver-specific keys ---

    def test_solver_specific_unknown_keys_passthrough(self):
        # e.g. mip_tolerances_mipgap (CPLEX-native) — translation
        # doesn't touch non-canonical keys regardless of solver.
        opts = {"mip_tolerances_mipgap": 0.001, "Cuts": 2}
        self.assertEqual(self._t(opts, "cplex"), opts)
        self.assertEqual(self._t(opts, "gurobi"), opts)


class TestDynamicGapperLayer(unittest.TestCase):
    """Gapper.set_mipgap writes to PHBase's reserved dynamic_gapper
    layer; subsequent _effective_solver_options(k) folds pick up the
    new mipgap and it overrides any CLI-configured value.
    """

    def _fake_ph(self, layers=None, current=None):
        """Minimal PHBase-shaped object with the layer state Gapper
        and _effective_solver_options need. Avoids MPI/scenario
        boilerplate.
        """
        from mpisppy.utils.sputils import solver_options_layer
        fake = type("FakePH", (), {})()
        fake.solver_options_layers = list(layers) if layers else []
        # Reserved dynamic layer (mirrors PHBase.__init__).
        fake._dynamic_solver_options_layer = solver_options_layer(
            "default", {})
        fake.solver_options_layers.append(
            fake._dynamic_solver_options_layer)
        fake.current_solver_options = current or {}
        return fake

    def _effective(self, fake, k):
        from mpisppy.phbase import PHBase
        return PHBase._effective_solver_options(fake, k)

    def test_set_mipgap_writes_to_dynamic_layer(self):
        # Direct unit test of the new contract: writing to the
        # dynamic layer shows up immediately in the effective dict.
        fake = self._fake_ph()
        fake._dynamic_solver_options_layer["options"]["mipgap"] = 0.001
        self.assertEqual(self._effective(fake, 0), {"mipgap": 0.001})
        self.assertEqual(self._effective(fake, 5), {"mipgap": 0.001})

    def test_dynamic_layer_overrides_cli_layers(self):
        # CLI-configured iter0/iterk layers + a dynamic write should
        # produce the dynamic value at every k, since the dynamic
        # layer is appended last.
        from mpisppy.utils.sputils import solver_options_layer
        cli = [
            solver_options_layer("iter0", {"mipgap": 0.1}),
            solver_options_layer("iterk", {"mipgap": 0.02}),
        ]
        fake = self._fake_ph(layers=cli)
        fake._dynamic_solver_options_layer["options"]["mipgap"] = 0.005
        self.assertEqual(
            self._effective(fake, 0)["mipgap"], 0.005)
        self.assertEqual(
            self._effective(fake, 1)["mipgap"], 0.005)
        self.assertEqual(
            self._effective(fake, 9)["mipgap"], 0.005)

    def test_gapper_set_mipgap_routes_to_dynamic_layer(self):
        # End-to-end: instantiate Gapper, call set_mipgap, assert
        # the dynamic layer reflects it and current_solver_options
        # is untouched (the old back-channel is now unused).
        from mpisppy.extensions.mipgapper import Gapper
        fake = self._fake_ph()
        fake.cylinder_rank = 0
        fake.options = {"gapperoptions": {
            "starting_mipgap": 0.1, "mipgap_ratio": 0.1,
        }}
        fake._get_cylinder_name = lambda: "TestCylinder"
        gapper = Gapper(fake)
        gapper.set_mipgap(0.01)
        self.assertEqual(
            fake._dynamic_solver_options_layer["options"]["mipgap"],
            0.01)
        # The back-compat dict was not mutated (only the layer was).
        self.assertNotIn("mipgap", fake.current_solver_options)

    def test_gapper_legacy_mipgapdict_translates_to_layers(self):
        # Compatibility shim: programmatic callers that still pass
        # mipgapdict in gapperoptions get a DeprecationWarning and
        # the schedule is translated into starting_at_iter layers on the
        # host PHBase (so their existing scripts keep producing the
        # same per-iteration mipgap). The Gapper extension itself
        # becomes a runtime no-op in this mode.
        import warnings
        from mpisppy.extensions.mipgapper import Gapper
        fake = self._fake_ph()
        fake.cylinder_rank = 0
        fake.options = {"gapperoptions": {
            "mipgapdict": {0: 0.10, 5: 0.005},
            "starting_mipgap": None,
        }}
        fake._get_cylinder_name = lambda: "TestCylinder"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gapper = Gapper(fake)
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning)
                and "mipgapdict" in str(w.message) for w in caught),
            f"Expected DeprecationWarning naming mipgapdict; "
            f"got {[(w.category.__name__, str(w.message)) for w in caught]}",
        )
        # Two layers, inserted before the dynamic layer. The N=0
        # entry becomes a `default` layer (since starting_at_iter:0
        # is rejected by the validator); N=5 becomes
        # starting_at_iter:5. solver_options_layers starts as
        # [_dynamic]; after the shim runs it should be
        # [default, starting_at_iter:5, _dynamic].
        layers = fake.solver_options_layers
        self.assertEqual(len(layers), 3)
        self.assertEqual(layers[0]["when"], "default")
        self.assertEqual(layers[0]["options"], {"mipgap": 0.10})
        self.assertEqual(layers[1]["when"], ("starting_at_iter", 5))
        self.assertEqual(layers[1]["options"], {"mipgap": 0.005})
        self.assertIs(
            layers[2], fake._dynamic_solver_options_layer)
        # Gapper's runtime hooks are no-ops in compat mode.
        self.assertTrue(gapper._static_compat)
        # pre_iter0 / miditer must not touch the dynamic layer.
        gapper.pre_iter0()
        gapper.miditer()
        self.assertEqual(
            fake._dynamic_solver_options_layer["options"], {})


class TestPerSpokeMipgapsJsonLayers(unittest.TestCase):
    """cfg_vanilla.add_gapper handles --{name}-mipgaps-json the same
    way as the global flag: per-spoke starting_at_iter layers are appended
    to the spoke dict's solver_options_layers.
    """

    def _write_mipgaps_json(self, schedule):
        import json
        import tempfile
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False).name
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in schedule.items()}, f)
        return path

    def _spoke_dict(self):
        return {
            "opt_kwargs": {
                "options": {"solver_options_layers": []},
            },
        }

    def test_per_spoke_mipgaps_json_appends_layers(self):
        import os
        import warnings
        from mpisppy.utils.cfg_vanilla import add_gapper
        cfg = _bare_cfg()
        cfg.gapper_args(name="lagrangian")
        path = self._write_mipgaps_json({0: 0.05, 3: 0.005})
        try:
            cfg.lagrangian_mipgaps_json = path
            spoke = self._spoke_dict()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                add_gapper(spoke, cfg, "lagrangian")
            self.assertTrue(
                any(issubclass(w.category, DeprecationWarning)
                    and "--lagrangian-mipgaps-json" in str(w.message)
                    for w in caught),
                f"Expected DeprecationWarning naming "
                f"--lagrangian-mipgaps-json; got "
                f"{[(w.category.__name__, str(w.message)) for w in caught]}",
            )
            layers = spoke["opt_kwargs"]["options"]["solver_options_layers"]
            self.assertEqual(len(layers), 2)
            self.assertEqual(layers[0]["when"], "default")
            self.assertEqual(layers[1]["when"], ("starting_at_iter", 3))
            self.assertEqual(
                [layer["options"] for layer in layers],
                [{"mipgap": 0.05}, {"mipgap": 0.005}],
            )
            self.assertNotIn(
                "gapperoptions", spoke["opt_kwargs"]["options"])
        finally:
            os.unlink(path)

    def test_per_spoke_static_and_auto_modes_exclusive(self):
        import os
        from mpisppy.utils.cfg_vanilla import add_gapper
        cfg = _bare_cfg()
        cfg.gapper_args(name="lagrangian")
        path = self._write_mipgaps_json({0: 0.05})
        try:
            cfg.lagrangian_mipgaps_json = path
            cfg.lagrangian_starting_mipgap = 0.1
            with self.assertRaisesRegex(
                    RuntimeError, "lagrangian-mipgaps-json"):
                add_gapper(self._spoke_dict(), cfg, "lagrangian")
        finally:
            os.unlink(path)


class TestLoadSolverOptionsFile(unittest.TestCase):
    """sputils.load_solver_options_file: schema validation, sub-block
    defaults, starting_at_iter int-coercion, error messages.
    """

    def _write(self, payload):
        import json
        import tempfile
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False).name
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    def test_happy_path_full_schema(self):
        import os
        from mpisppy.utils.sputils import load_solver_options_file
        path = self._write({
            "default": {"threads": 4},
            "iter0": {"mipgap": 1e-4},
            "iterk": {"mipgap": 1e-3},
            "starting_at_iter": {"5": {"mipgap": 1e-5}, "10": {"mipgap": 1e-6}},
            "spokes": {
                "lagrangian": {"default": {"mipgap": 0.01}},
                "reduced_costs": {"iter0": {"mipgap": 0.001}},
            },
        })
        try:
            data = load_solver_options_file(path)
            self.assertEqual(data["default"], {"threads": 4})
            self.assertEqual(data["iter0"], {"mipgap": 1e-4})
            self.assertEqual(data["iterk"], {"mipgap": 1e-3})
            self.assertEqual(
                data["starting_at_iter"], {5: {"mipgap": 1e-5}, 10: {"mipgap": 1e-6}})
            self.assertEqual(
                data["spokes"]["lagrangian"]["default"], {"mipgap": 0.01})
            self.assertEqual(
                data["spokes"]["reduced_costs"]["iter0"], {"mipgap": 0.001})
        finally:
            os.unlink(path)

    def test_missing_sub_blocks_default_to_empty(self):
        import os
        from mpisppy.utils.sputils import load_solver_options_file
        path = self._write({"iter0": {"mipgap": 0.01}})
        try:
            data = load_solver_options_file(path)
            self.assertEqual(data["default"], {})
            self.assertEqual(data["iter0"], {"mipgap": 0.01})
            self.assertEqual(data["iterk"], {})
            self.assertEqual(data["starting_at_iter"], {})
            self.assertEqual(data["spokes"], {})
        finally:
            os.unlink(path)

    def test_unknown_top_level_key_rejected(self):
        import os
        from mpisppy.utils.sputils import load_solver_options_file
        path = self._write({"defualt": {"mipgap": 0.01}})  # typo
        try:
            with self.assertRaisesRegex(ValueError, "defualt"):
                load_solver_options_file(path)
        finally:
            os.unlink(path)

    def test_starting_at_iter_non_integer_key_rejected(self):
        import os
        from mpisppy.utils.sputils import load_solver_options_file
        path = self._write({"starting_at_iter": {"five": {"mipgap": 1e-5}}})
        try:
            with self.assertRaisesRegex(ValueError, "five"):
                load_solver_options_file(path)
        finally:
            os.unlink(path)

    def test_starting_at_iter_negative_key_rejected(self):
        import os
        from mpisppy.utils.sputils import load_solver_options_file
        path = self._write({"starting_at_iter": {"-3": {"mipgap": 1e-5}}})
        try:
            with self.assertRaisesRegex(ValueError, ">= 1"):
                load_solver_options_file(path)
        finally:
            os.unlink(path)

    def test_starting_at_iter_zero_key_rejected_with_hint(self):
        # N=0 in the file should fail with a message that points the
        # caller at the `default` sub-block (since "starting at iter 0"
        # really means "applies always", which is what `default` is for).
        import os
        from mpisppy.utils.sputils import load_solver_options_file
        path = self._write({"starting_at_iter": {"0": {"mipgap": 1e-5}}})
        try:
            with self.assertRaisesRegex(ValueError, "default"):
                load_solver_options_file(path)
        finally:
            os.unlink(path)

    def test_unknown_key_in_spoke_subblock_rejected(self):
        import os
        from mpisppy.utils.sputils import load_solver_options_file
        path = self._write({
            "spokes": {"lagrangian": {"spokes": {"x": {}}}}
        })
        try:
            # Nested "spokes" inside a spoke sub-block is not allowed.
            with self.assertRaisesRegex(
                    ValueError, "spokes.lagrangian.*spokes"):
                load_solver_options_file(path)
        finally:
            os.unlink(path)

    def test_malformed_json_raises_valueerror(self):
        import os
        import tempfile
        from mpisppy.utils.sputils import load_solver_options_file
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False).name
        try:
            with open(path, "w") as f:
                f.write("not json {")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_solver_options_file(path)
        finally:
            os.unlink(path)


class TestSolverOptionsFileWiredIntoSharedOptions(unittest.TestCase):
    """End-to-end: --solver-options-file plumbing through
    cfg_vanilla.shared_options into solver_options_layers + the legacy
    iter0/iterk dicts.
    """

    def _write(self, payload):
        import json
        import tempfile
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False).name
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    def test_file_default_iter0_iterk_appear_as_layers(self):
        import os
        cfg = _bare_cfg()
        path = self._write({
            "default": {"threads": 2, "presolve": 1},
            "iter0": {"mipgap": 1e-4},
            "iterk": {"mipgap": 1e-3},
            "starting_at_iter": {"5": {"mipgap": 1e-5}},
        })
        try:
            cfg.solver_options_file = path
            sh = shared_options(cfg)
            self.assertEqual(
                fold_solver_options_layers(sh["solver_options_layers"], 0),
                {"threads": 2, "presolve": 1, "mipgap": 1e-4},
            )
            self.assertEqual(
                fold_solver_options_layers(sh["solver_options_layers"], 3),
                {"threads": 2, "presolve": 1, "mipgap": 1e-3},
            )
            self.assertEqual(
                fold_solver_options_layers(sh["solver_options_layers"], 7),
                {"threads": 2, "presolve": 1, "mipgap": 1e-5},
            )
        finally:
            os.unlink(path)

    def test_inline_solver_options_overrides_file_default(self):
        # Axis 2: --solver-options is above --solver-options-file in
        # the default predicate, so inline keys win.
        import os
        cfg = _bare_cfg()
        path = self._write({"default": {"mipgap": 0.5, "threads": 2}})
        try:
            cfg.solver_options_file = path
            cfg.solver_options = "mipgap=0.001"
            sh = shared_options(cfg)
            folded = fold_solver_options_layers(sh["solver_options_layers"], 0)
            self.assertEqual(folded["mipgap"], 0.001)  # inline wins
            self.assertEqual(folded["threads"], 2)     # file survives
        finally:
            os.unlink(path)

    def test_iter0_mipgap_sugar_overrides_file_iter0(self):
        import os
        cfg = _bare_cfg()
        path = self._write({"iter0": {"mipgap": 0.5}})
        try:
            cfg.solver_options_file = path
            cfg.iter0_mipgap = 0.001
            sh = shared_options(cfg)
            folded = fold_solver_options_layers(sh["solver_options_layers"], 0)
            self.assertEqual(folded["mipgap"], 0.001)
        finally:
            os.unlink(path)

    def test_iterk_mipgap_sugar_overrides_file_iterk(self):
        import os
        cfg = _bare_cfg()
        path = self._write({"iterk": {"mipgap": 0.5}})
        try:
            cfg.solver_options_file = path
            cfg.iterk_mipgap = 0.001
            sh = shared_options(cfg)
            folded = fold_solver_options_layers(sh["solver_options_layers"], 3)
            self.assertEqual(folded["mipgap"], 0.001)
        finally:
            os.unlink(path)

    def test_file_starting_at_iter_persists_until_next_starting_at_iter(self):
        import os
        cfg = _bare_cfg()
        # `default` sets the always-applies base; starting_at_iter:5
        # kicks in at k>=5 and overrides for the rest of the run.
        path = self._write({
            "default": {"mipgap": 0.1},
            "starting_at_iter": {"5": {"mipgap": 0.01}},
        })
        try:
            cfg.solver_options_file = path
            sh = shared_options(cfg)
            self.assertEqual(
                fold_solver_options_layers(sh["solver_options_layers"], 0)["mipgap"],
                0.1)
            self.assertEqual(
                fold_solver_options_layers(sh["solver_options_layers"], 4)["mipgap"],
                0.1)
            self.assertEqual(
                fold_solver_options_layers(sh["solver_options_layers"], 5)["mipgap"],
                0.01)
        finally:
            os.unlink(path)


class TestSolverOptionsFilePerSpoke(unittest.TestCase):
    """Per-spoke layers from (a) the global file's "spokes" sub-block
    and (b) a dedicated --{name}-solver-options-file flag.
    """

    def _write(self, payload):
        import json
        import tempfile
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False).name
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    def _spoke_dict_from(self, sh):
        return {"opt_kwargs": {"options": copy.deepcopy(sh)}}

    def test_global_files_spokes_subblock_adds_spoke_layers(self):
        import os
        cfg = _spoke_cfg("lagrangian")
        path = self._write({
            "default": {"threads": 2},
            "spokes": {
                "lagrangian": {
                    "default": {"mipgap": 0.01},
                    "iter0": {"mipgap": 0.001},
                },
            },
        })
        try:
            cfg.solver_options_file = path
            sh = shared_options(cfg)
            spoke = self._spoke_dict_from(sh)
            apply_solver_specs("lagrangian", spoke, cfg)
            opts = spoke["opt_kwargs"]["options"]
            self.assertEqual(
                fold_solver_options_layers(opts["solver_options_layers"], 0),
                {"threads": 2, "mipgap": 0.001},
            )
            self.assertEqual(
                fold_solver_options_layers(opts["solver_options_layers"], 1),
                {"threads": 2, "mipgap": 0.01},
            )
        finally:
            os.unlink(path)

    def test_dedicated_spoke_file_adds_spoke_layers(self):
        import os
        cfg = _spoke_cfg("lagrangian")
        path = self._write({"default": {"mipgap": 0.005}})
        try:
            cfg.lagrangian_solver_options_file = path
            sh = shared_options(cfg)
            spoke = self._spoke_dict_from(sh)
            apply_solver_specs("lagrangian", spoke, cfg)
            opts = spoke["opt_kwargs"]["options"]
            self.assertEqual(
                fold_solver_options_layers(
                    opts["solver_options_layers"], 0)["mipgap"],
                0.005)
        finally:
            os.unlink(path)

    def test_spoke_starting_at_iter_subblock_honored(self):
        # Spoke sub-blocks support every predicate the top-level
        # supports, including starting_at_iter. Verify the schema example
        # in the docs holds at runtime.
        import os
        cfg = _spoke_cfg("lagrangian")
        path = self._write({
            "spokes": {
                "lagrangian": {
                    "default":    {"mipgap": 0.01},
                    "starting_at_iter": {"5": {"mipgap": 0.001}},
                },
            },
        })
        try:
            cfg.solver_options_file = path
            sh = shared_options(cfg)
            spoke = self._spoke_dict_from(sh)
            apply_solver_specs("lagrangian", spoke, cfg)
            opts = spoke["opt_kwargs"]["options"]
            self.assertEqual(
                fold_solver_options_layers(
                    opts["solver_options_layers"], 0)["mipgap"],
                0.01)
            self.assertEqual(
                fold_solver_options_layers(
                    opts["solver_options_layers"], 4)["mipgap"],
                0.01)
            self.assertEqual(
                fold_solver_options_layers(
                    opts["solver_options_layers"], 5)["mipgap"],
                0.001)
            self.assertEqual(
                fold_solver_options_layers(
                    opts["solver_options_layers"], 9)["mipgap"],
                0.001)
        finally:
            os.unlink(path)

    def test_spoke_inline_overrides_spoke_file(self):
        # Axis 2 within a spoke: --{name}-solver-options is above the
        # spoke's file contribution.
        import os
        cfg = _spoke_cfg("lagrangian")
        path = self._write({"default": {"mipgap": 0.5, "presolve": 1}})
        try:
            cfg.lagrangian_solver_options_file = path
            cfg.lagrangian_solver_options = "mipgap=0.001"
            sh = shared_options(cfg)
            spoke = self._spoke_dict_from(sh)
            apply_solver_specs("lagrangian", spoke, cfg)
            opts = spoke["opt_kwargs"]["options"]
            folded = fold_solver_options_layers(
                opts["solver_options_layers"], 0)
            self.assertEqual(folded["mipgap"], 0.001)  # inline wins
            self.assertEqual(folded["presolve"], 1)    # file survives


        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
