###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for pyfory-rebuild checkpoint cloning, NL digests, and resume."""

import copy
import json
import os
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import mpisppy.utils.checkpointing as checkpointing
from mpisppy.extensions.checkpointer import Checkpointer
from mpisppy.extensions.extension import Extension
from mpisppy.tests.test_checkpoint import (
    SCENARIO_NAMES,
    CheckpointPayloadExtension,
    _flat_snapshot,
    _make_ph,
    _options,
    solver_available,
)
from mpisppy.tests.utils import subprocess_env


class RebuildSafeStateExtension(Extension):
    """Add late model structure that pyfory-rebuild must recreate."""

    checkpoint_rebuild_safe = True

    def checkpoint_state(self):
        return None

    def restore_state(self, state):
        pass

    @staticmethod
    def _install(scenario, state):
        import pyomo.environ as pyo

        var_name = state["variable_name"]
        constraint_name = state["constraint_name"]
        if scenario.find_component(var_name) is not None:
            return
        marker = pyo.Var(initialize=-5.0)
        scenario.add_component(var_name, marker)
        # A scalar Var is not constructed until it is attached to the
        # concrete scenario model. Fix it after add_component has constructed
        # it, so this extension accurately models a late-added Pyomo variable.
        marker.fix(-5.0)
        scenario.add_component(
            constraint_name,
            pyo.Constraint(expr=marker >= state["lower_bound"]))

    def enditer(self):
        state = self.checkpoint_model_state()
        for index, scenario in enumerate(self.opt.local_scenarios.values()):
            self._install(scenario, state)
            scenario.find_component(state["variable_name"]).set_value(
                73.0 + index)

    def checkpoint_model_state(self):
        return {
            "scenario_names": sorted(self.opt.local_scenarios),
            "variable_name": "rebuild_extension_marker",
            "constraint_name": "rebuild_extension_marker_lb",
            "lower_bound": -2.0,
        }

    def restore_checkpoint_model_state(self, state):
        if (not isinstance(state, dict)
                or state.get("scenario_names")
                != sorted(self.opt.local_scenarios)):
            raise ValueError(
                "RebuildSafeStateExtension state does not match its local "
                "scenarios.")
        for scenario in self.opt.local_scenarios.values():
            self._install(scenario, state)
        self.restored_model_state = state


class RebuildMissingModelStateHooksExtension(Extension):
    """Declare rebuild support without implementing its model-state hooks."""

    checkpoint_rebuild_safe = True

    def checkpoint_state(self):
        return None

    def restore_state(self, state):
        pass


class RebuildUnencodableModelStateExtension(Extension):
    """Return model state outside the plain pyfory checkpoint format."""

    checkpoint_rebuild_safe = True

    def checkpoint_state(self):
        return None

    def restore_state(self, state):
        pass

    def checkpoint_model_state(self):
        return object()

    def restore_checkpoint_model_state(self, state):
        pass


def _resume_rebuild_extension_and_dump(ckpt_dir, out_path):
    """Resume extension-added structure in a clean interpreter and report it."""
    options = _options(
        0, ckpt_dir=ckpt_dir, resume_from=ckpt_dir,
        checkpoint_backend=checkpointing.PYFORY_REBUILD_BACKEND)
    ph = _make_ph(options, extension_name="rebuild_safe")
    ph.PH_Prep()
    ph.Iter0()
    scenarios = {}
    for sname, scenario in ph.local_scenarios.items():
        marker = scenario.find_component("rebuild_extension_marker")
        constraint = scenario.find_component(
            "rebuild_extension_marker_lb")
        scenarios[sname] = {
            "marker_value": float(marker.value),
            "constraint_present": constraint is not None,
            "component_layout": checkpointing._component_layout(scenario),
        }
    with open(out_path, "w") as stream:
        json.dump({
            "resumed": bool(getattr(ph, "_resumed_from_checkpoint", False)),
            "resume_iteration": int(getattr(ph, "_resume_iteration", 0)),
            "scenarios": scenarios,
        }, stream)


class TestPyforyRebuildProbeClone(unittest.TestCase):
    def test_dict_keys_set_initializers_survive_probe_clone(self):
        import pyomo.core.base.component as pyomo_component
        import pyomo.environ as pyo

        model = pyo.ConcreteModel()
        model.empty = pyo.Set(initialize={}.keys())
        model.nested = pyo.Block()
        model.nested.choices = pyo.Set(
            initialize={"first": 1, "second": 2}.keys())
        original = {
            name: model.find_component(name)._init_values
            for name in ("empty", "nested.choices")
        }

        with mock.patch.object(pyomo_component.logger, "error") as log_error:
            clone = checkpointing._clone_rebuild_probe_model(model)
        log_error.assert_not_called()

        for name, initializer in original.items():
            source = model.find_component(name)
            copied = clone.find_component(name)
            self.assertIs(source._init_values, initializer)
            self.assertIsNotNone(copied._init_values)
            self.assertEqual(tuple(source), tuple(copied))
            self.assertIsInstance(copied._init_values._init.val, tuple)


class TestRebuildConstraintActivity(unittest.TestCase):
    def test_changed_indexed_constraint_activity_refuses_resume(self):
        import pyomo.environ as pyo

        def scenario():
            model = pyo.ConcreteModel(name="scenario")
            model.x = pyo.Var()
            model.rows = pyo.Constraint(
                [1, 2], rule=lambda m, index: m.x >= index)
            model._mpisppy_data = types.SimpleNamespace(
                inner_bound=None, outer_bound=None)
            return model

        saved = scenario()
        saved.rows[2].deactivate()
        snapshot = checkpointing._snapshot_rebuild_model(saved)
        rebuilt = scenario()
        opt = types.SimpleNamespace(local_scenarios={"scenario": rebuilt})

        self.assertTrue(saved.rows.active)
        self.assertTrue(rebuilt.rows.active)
        with self.assertRaisesRegex(
                checkpointing.CheckpointMismatch,
                "does not have the same component structure"):
            checkpointing.restore_rebuild_model_state(
                opt, {"scenario": snapshot})

        rebuilt.rows[2].deactivate()
        checkpointing.restore_rebuild_model_state(
            opt, {"scenario": snapshot})


class TestRebuildProbeFailureAgreement(unittest.TestCase):
    def test_extension_state_hook_failure_reaches_rank_agreement(self):
        class RecordingComm:
            agreed = False

            def Allreduce(self, send, receive, op=None):
                self.agreed = True
                receive[:] = send

        comm = RecordingComm()
        opt = types.SimpleNamespace(
            options={}, cylinder_rank=0, n_proc=2, mpicomm=comm,
            local_scenarios={}, extobject=None)
        with mock.patch.object(checkpointing, "require_backend_dependency"):
            with mock.patch.object(
                    checkpointing, "gather_extension_model_state",
                    side_effect=ValueError("injected hook failure")):
                with self.assertRaisesRegex(
                        RuntimeError,
                        "cannot gather extension model state.*injected hook failure"):
                    checkpointing.probe_model_checkpointable(
                        opt, checkpointing.PYFORY_REBUILD_BACKEND)
        self.assertTrue(comm.agreed)


@unittest.skipUnless(checkpointing.pyfory_available,
                     "pyfory is an optional checkpoint dependency")
@unittest.skipUnless(solver_available,
                     "no solver is available for checkpoint resume tests")
class TestPyforyRebuildBackend(unittest.TestCase):
    """The pyfory codec round trips plain state and rebuilds PH models."""

    N = 4
    STOP = 2

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _options(self, max_iters, **overrides):
        return _options(
            max_iters,
            ckpt_dir=self.ckpt_dir,
            checkpoint_backend=checkpointing.PYFORY_REBUILD_BACKEND,
            **overrides,
        )

    def test_round_trip_and_resume_in_a_fresh_process(self):
        reference = _make_ph(_options(self.N))
        reference.ph_main()

        stopped = _make_ph(self._options(self.STOP))
        stopped.ph_main()
        snapshot = checkpointing._snapshot_rebuild_model(
            next(iter(stopped.local_scenarios.values())))
        self.assertIsInstance(snapshot["variable_values"]["reals"],
                              np.ndarray)
        self.assertIsInstance(snapshot["mutable_param_values"]["reals"],
                              np.ndarray)
        decoded = checkpointing._new_fory().deserialize(
            checkpointing._new_fory().serialize(snapshot))
        self.assertTrue(checkpointing._checkpoint_state_equal(
            snapshot, decoded))
        with open(os.path.join(self.ckpt_dir, "manifest.json")) as stream:
            manifest = json.load(stream)
        self.assertEqual(manifest["backend"],
                         checkpointing.PYFORY_REBUILD_BACKEND)
        self.assertEqual(manifest["codec_version"],
                         checkpointing._pyfory_version())
        self.assertTrue(manifest["write_id"])
        generation_dir = os.path.join(
            self.ckpt_dir, "hub", f"gen_{self.STOP:04d}")
        with open(os.path.join(generation_dir, "hub_rank_0000.pkl"),
                  "rb") as stream:
            leaf = pickle.load(stream)
        self.assertEqual(leaf["write_id"], manifest["write_id"])
        self.assertNotIn("nl_fingerprint_files", leaf)
        self.assertTrue(all(name.endswith(".fory")
                            for name in leaf["model_files"].values()))

        out_path = os.path.join(self._tmp.name, "resumed.json")
        call_args = json.dumps([
            self.ckpt_dir, self.N - self.STOP, out_path,
            checkpointing.PYFORY_REBUILD_BACKEND,
        ])
        result = subprocess.run(
            [sys.executable, "-c",
             "import json, mpisppy.tests.test_checkpoint as t; "
             f"t._resume_and_dump(*json.loads({call_args!r}))"],
            capture_output=True, text=True, timeout=900, check=False,
            env=subprocess_env(),
        )
        self.assertEqual(
            result.returncode, 0,
            msg=f"fresh-process pyfory resume failed:\n{result.stderr}")
        with open(out_path) as stream:
            resumed = json.load(stream)
        self.assertTrue(resumed["resumed"])
        self.assertEqual(resumed["resume_iteration"], self.STOP)
        self.assertEqual(resumed["state"], _flat_snapshot(reference))

    def test_cross_backend_writes_decode_from_the_manifest(self):
        dill_run = _make_ph(_options(
            2, ckpt_dir=self.ckpt_dir,
            checkpoint_backend=checkpointing.DILL_RELOAD_BACKEND))
        dill_run.ph_main()

        # The command-line backend selects the next write. The manifest still
        # selects the existing checkpoint's decoder.
        pyfory_run = _make_ph(_options(
            1, ckpt_dir=self.ckpt_dir, resume_from=self.ckpt_dir,
            checkpoint_backend=checkpointing.PYFORY_REBUILD_BACKEND))
        pyfory_run.ph_main()
        with open(os.path.join(self.ckpt_dir, "manifest.json")) as stream:
            manifest = json.load(stream)
        self.assertEqual(manifest["backend"],
                         checkpointing.PYFORY_REBUILD_BACKEND)

        dill_again = _make_ph(_options(
            1, ckpt_dir=self.ckpt_dir, resume_from=self.ckpt_dir,
            checkpoint_backend=checkpointing.DILL_RELOAD_BACKEND))
        dill_again.ph_main()
        with open(os.path.join(self.ckpt_dir, "manifest.json")) as stream:
            manifest = json.load(stream)
        self.assertEqual(manifest["backend"],
                         checkpointing.DILL_RELOAD_BACKEND)
        self.assertEqual(set(manifest), {
            "format_version", "backend", "generation", "n_proc",
            "structural_fingerprint",
        })
        generation_dir = os.path.join(
            self.ckpt_dir, "hub", f"gen_{manifest['generation']:04d}")
        with open(os.path.join(generation_dir, "hub_rank_0000.pkl"),
                  "rb") as stream:
            leaf = pickle.load(stream)
        self.assertNotIn("write_id", leaf)
        self.assertEqual(set(leaf), {
            "format_version", "backend", "generation", "geometry",
            "structural_fingerprint", "model_files",
            "initially_fixed_nonants", "extension_state", "trivial_bound",
            "best_bound_obj_val", "best_solution_obj_val",
            "best_outer_bound",
        })
        self.assertTrue(all(name.endswith(".dill")
                            for name in leaf["model_files"].values()))

    def test_interrupted_cross_backend_replacements_keep_the_manifest_copy(self):
        opt = _make_ph(_options(
            1, ckpt_dir=self.ckpt_dir,
            checkpoint_backend=checkpointing.DILL_RELOAD_BACKEND))
        opt.ph_main()
        generation = 1

        def interrupt_before_flip(new_backend):
            with mock.patch.object(
                    checkpointing, "_publish_manifest",
                    side_effect=OSError("injected before manifest flip")), \
                    mock.patch.object(
                        checkpointing, "_sweep_uncommitted_generations"):
                with self.assertRaisesRegex(OSError,
                                            "before manifest flip"):
                    checkpointing.write_checkpoint(
                        opt, self.ckpt_dir, generation,
                        backend=new_backend)

        def interrupt_after_flip(new_backend):
            with mock.patch.object(
                    checkpointing, "_sweep_uncommitted_generations"):
                checkpointing.write_checkpoint(
                    opt, self.ckpt_dir, generation, backend=new_backend)

        # Old dill remains selected while the new pyfory directory is already
        # in place but the manifest still points at dill.
        interrupt_before_flip(checkpointing.PYFORY_REBUILD_BACKEND)
        leaf, _ = checkpointing.load_checkpoint(opt, self.ckpt_dir)
        self.assertEqual(leaf["backend"], checkpointing.DILL_RELOAD_BACKEND)
        checkpointing._sweep_uncommitted_generations(
            self.ckpt_dir, os.path.join(self.ckpt_dir, "hub"))
        self.assertEqual(os.listdir(os.path.join(self.ckpt_dir, "hub")),
                         ["gen_0001.retiring"])

        # The pyfory write commits but cleanup has not run. Its write_id picks
        # the final directory while the retired dill copy is still present.
        interrupt_after_flip(checkpointing.PYFORY_REBUILD_BACKEND)
        leaf, _ = checkpointing.load_checkpoint(opt, self.ckpt_dir)
        self.assertEqual(leaf["backend"],
                         checkpointing.PYFORY_REBUILD_BACKEND)
        checkpointing._sweep_uncommitted_generations(
            self.ckpt_dir, os.path.join(self.ckpt_dir, "hub"))
        self.assertEqual(os.listdir(os.path.join(self.ckpt_dir, "hub")),
                         ["gen_0001"])

        # Reverse direction before the manifest flip: the pyfory copy is
        # selected by its write identity from .retiring.
        interrupt_before_flip(checkpointing.DILL_RELOAD_BACKEND)
        leaf, _ = checkpointing.load_checkpoint(opt, self.ckpt_dir)
        self.assertEqual(leaf["backend"],
                         checkpointing.PYFORY_REBUILD_BACKEND)
        checkpointing._sweep_uncommitted_generations(
            self.ckpt_dir, os.path.join(self.ckpt_dir, "hub"))
        self.assertEqual(os.listdir(os.path.join(self.ckpt_dir, "hub")),
                         ["gen_0001.retiring"])

        # And after the flip: the new dill copy is selected and the retired
        # pyfory generation is safe to reclaim.
        interrupt_after_flip(checkpointing.DILL_RELOAD_BACKEND)
        leaf, _ = checkpointing.load_checkpoint(opt, self.ckpt_dir)
        self.assertEqual(leaf["backend"], checkpointing.DILL_RELOAD_BACKEND)
        checkpointing._sweep_uncommitted_generations(
            self.ckpt_dir, os.path.join(self.ckpt_dir, "hub"))
        self.assertEqual(os.listdir(os.path.join(self.ckpt_dir, "hub")),
                         ["gen_0001"])

    def test_linearized_prox_cuts_and_manager_state_are_replayed(self):
        options = self._options(2, linearize_proximal_terms=True)
        writer = _make_ph(options)
        writer.ph_main()

        def cut_state(ph):
            result = {}
            for sname, scenario in ph.local_scenarios.items():
                managers = scenario._mpisppy_data.xsqvar_prox_approx
                result[sname] = {}
                for manager in managers.values():
                    keys = []
                    for key in scenario._mpisppy_model.xsqvar_cuts.keys():
                        if isinstance(key, tuple) and (
                                key[0] == manager.var_index
                                or tuple(key[:-1])
                                == tuple(manager.var_index)):
                            keys.append(key)
                    result[sname][manager.xvar.name] = (
                        tuple(float(value) for value in manager.cut_values),
                        tuple(sorted(map(repr, keys))),
                    )
            return result

        before = cut_state(writer)
        self.assertTrue(any(
            len(keys) > 0
            for scenario in before.values() for keys in scenario.values()),
            "the test run did not generate any proximal cut managers")

        resumed = _make_ph(_options(
            1, ckpt_dir=self.ckpt_dir, resume_from=self.ckpt_dir,
            checkpoint_backend=checkpointing.PYFORY_REBUILD_BACKEND,
            linearize_proximal_terms=True))
        resumed.PH_Prep()
        resumed.Iter0()
        self.assertEqual(cut_state(resumed), before)

    def test_discrete_prox_cuts_are_replayed(self):
        creator_kwargs = {"use_integer": True, "crops_multiplier": 1}
        structural_cfg = {
            "model": "test-farmer",
            "creator_kwargs": dict(creator_kwargs),
        }
        writer = _make_ph(
            self._options(
                2, linearize_proximal_terms=True,
                checkpoint_structural_cfg=structural_cfg),
            scenario_creator_kwargs=creator_kwargs)
        writer.ph_main()

        before = {
            sname: checkpointing._prox_approx_snapshot(scenario)
            for sname, scenario in writer.local_scenarios.items()
        }
        self.assertTrue(any(
            state["kind"] == "discrete" and state["cut_indices"]
            for scenario in before.values()
            for state in scenario.values()),
            "the integer test run did not generate any discrete proximal cuts")

        resumed = _make_ph(_options(
            0, resume_from=self.ckpt_dir,
            checkpoint_backend=checkpointing.PYFORY_REBUILD_BACKEND,
            linearize_proximal_terms=True,
            checkpoint_structural_cfg=structural_cfg),
            scenario_creator_kwargs=creator_kwargs)
        resumed.PH_Prep()
        resumed.Iter0()
        after = {
            sname: checkpointing._prox_approx_snapshot(scenario)
            for sname, scenario in resumed.local_scenarios.items()
        }
        self.assertEqual(after, before)

    def test_extension_must_opt_in_and_restore_model_state(self):
        unsafe = _make_ph(
            self._options(1), extension_name="recorder")
        with self.assertRaisesRegex(RuntimeError,
                                    "has not declared support"):
            unsafe.ph_main()

        writer = _make_ph(self._options(1),
                          extension_name="rebuild_safe")
        writer.ph_main()
        resumed = _make_ph(_options(
            1, ckpt_dir=self.ckpt_dir, resume_from=self.ckpt_dir,
            checkpoint_backend=checkpointing.PYFORY_REBUILD_BACKEND),
            extension_name="rebuild_safe")
        resumed.PH_Prep()
        resumed.Iter0()
        extension = dict(checkpointing._extension_objects(resumed))[
            "RebuildSafeStateExtension"]
        self.assertEqual(
            extension.restored_model_state["scenario_names"],
            SCENARIO_NAMES)
        self.assertEqual(
            extension.restored_model_state["variable_name"],
            "rebuild_extension_marker")

    def test_extension_rebuild_opt_in_requires_both_model_state_hooks(self):
        unsafe = _make_ph(
            self._options(1),
            extension_name="rebuild_missing_model_state_hooks")
        with self.assertRaisesRegex(
                RuntimeError,
                "does not implement both checkpoint model-state hooks"):
            unsafe.ph_main()

    def test_extension_model_state_must_be_encodable_by_pyfory(self):
        unencodable = _make_ph(
            self._options(1),
            extension_name="rebuild_unencodable_model_state")
        with self.assertRaisesRegex(
                RuntimeError,
                "cannot encode model state from extension '"):
            unencodable.ph_main()

    def test_late_extension_structure_and_values_restore_in_a_fresh_process(
            self):
        writer = _make_ph(self._options(1), extension_name="rebuild_safe")
        writer.ph_main()
        expected_values = {
            sname: float(scenario.rebuild_extension_marker.value)
            for sname, scenario in writer.local_scenarios.items()
        }
        expected_layouts = {
            sname: json.loads(json.dumps(
                checkpointing._component_layout(scenario)))
            for sname, scenario in writer.local_scenarios.items()
        }

        out_path = os.path.join(self._tmp.name, "rebuild-extension.json")
        call_args = json.dumps([self.ckpt_dir, out_path])
        result = subprocess.run(
            [sys.executable, "-c",
             "import json, mpisppy.tests.test_checkpoint_pyfory_rebuild as t; "
             f"t._resume_rebuild_extension_and_dump(*json.loads({call_args!r}))"],
            capture_output=True, text=True, timeout=900, check=False,
            env=subprocess_env(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(out_path) as stream:
            resumed = json.load(stream)
        self.assertTrue(resumed["resumed"])
        self.assertEqual(resumed["resume_iteration"], 1)
        for sname, saved_value in expected_values.items():
            self.assertEqual(
                resumed["scenarios"][sname]["marker_value"], saved_value)
            self.assertTrue(
                resumed["scenarios"][sname]["constraint_present"])
            self.assertEqual(
                resumed["scenarios"][sname]["component_layout"],
                expected_layouts[sname])

    def test_decimal_snapshot_values_keep_type_and_report_context(self):
        from decimal import Decimal

        import pyomo.environ as pyo

        ph = _make_ph(_options(1))
        ph.PH_Prep()
        scenario = next(iter(ph.local_scenarios.values()))
        scenario.checkpoint_decimal = pyo.Param(
            mutable=True, initialize=Decimal("1.234500"))
        snapshot = checkpointing._snapshot_rebuild_model(scenario)
        encoded = checkpointing._new_fory().serialize(snapshot)
        decoded = checkpointing._new_fory().deserialize(encoded)
        restored = scenario.clone()
        checkpointing._restore_rebuild_model_values(restored, decoded)
        self.assertIsInstance(restored.checkpoint_decimal.value, Decimal)
        self.assertEqual(restored.checkpoint_decimal.value,
                         Decimal("1.234500"))
        self.assertTrue(checkpointing._checkpoint_state_equal(
            checkpointing._snapshot_rebuild_model(restored), decoded))

        scenario.checkpoint_unsupported = pyo.Param(
            mutable=True, within=pyo.Any, initialize=object())
        with self.assertRaisesRegex(
                TypeError,
                f"scenario '{scenario.name}'.*mutable Param "
                "'checkpoint_unsupported' index None.*builtins.object"):
            checkpointing._snapshot_rebuild_model(scenario)

    def test_malformed_rebuild_snapshots_are_rejected(self):
        ph = _make_ph(self._options(1))
        ph.PH_Prep()
        scenario = next(iter(ph.local_scenarios.values()))
        snapshot = checkpointing._snapshot_rebuild_model(scenario)

        bad_version = copy.deepcopy(snapshot)
        bad_version["state_version"] += 1
        with self.assertRaisesRegex(
                checkpointing.CheckpointMismatch, "state version"):
            checkpointing._restore_rebuild_model_values(
                scenario, bad_version)

        bad_values = copy.deepcopy(snapshot)
        bad_values["variable_values"]["kinds"] = []
        with self.assertRaisesRegex(
                checkpointing.CheckpointMismatch,
                "inconsistent variable state"):
            checkpointing._restore_rebuild_model_values(
                scenario, bad_values)

    def test_extension_object_payload_is_restored(self):
        writer = _make_ph(
            self._options(1), extension_name="checkpoint_payload")
        writer.ph_main()
        expected = CheckpointPayloadExtension(writer).payload

        resumed = _make_ph(_options(
            0, ckpt_dir=self.ckpt_dir, resume_from=self.ckpt_dir,
            checkpoint_backend=checkpointing.PYFORY_REBUILD_BACKEND),
            extension_name="checkpoint_payload")
        resumed.PH_Prep()
        resumed.Iter0()
        extension = dict(checkpointing._extension_objects(resumed))[
            "CheckpointPayloadExtension"]
        self.assertEqual(extension.restored_payload, expected)

    def test_shipped_snapshot_safe_extensions_opt_in(self):
        from mpisppy.extensions.coeff_rho import CoeffRho
        from mpisppy.extensions.dyn_rho_base import Dyn_Rho_extension_base
        from mpisppy.extensions.fixer import Fixer
        from mpisppy.extensions.grad_rho import GradRho
        from mpisppy.extensions.integer_relax_then_enforce import (
            IntegerRelaxThenEnforce,
        )
        from mpisppy.extensions.mult_rho_updater import MultRhoUpdater
        from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
        from mpisppy.extensions.primal_dual_rho import PrimalDualRho
        from mpisppy.extensions.reduced_costs_fixer import ReducedCostsFixer
        from mpisppy.extensions.relaxed_ph_fixer import RelaxedPHFixer
        from mpisppy.extensions.sensi_rho import SensiRho
        from mpisppy.extensions.sep_rho import SepRho
        from mpisppy.extensions.slammer import Slammer

        for extension_cls in (
                CoeffRho, Dyn_Rho_extension_base, Fixer, GradRho,
                IntegerRelaxThenEnforce, MultRhoUpdater, NormRhoUpdater,
                PrimalDualRho, ReducedCostsFixer, RelaxedPHFixer, SepRho,
                SensiRho, Slammer):
            with self.subTest(extension=extension_cls.__name__):
                self.assertTrue(extension_cls.checkpoint_rebuild_safe)

        scenario = types.SimpleNamespace(
            _mpisppy_data=types.SimpleNamespace(
                conv_iter_count={("ROOT", 0): 3, ("ROOT", 1): 1}))
        fixer = Fixer.__new__(Fixer)
        fixer.ph = types.SimpleNamespace(local_scenarios={"scen0": scenario})
        saved_counts = fixer.checkpoint_model_state()
        scenario._mpisppy_data.conv_iter_count = {}
        fixer.restore_checkpoint_model_state(saved_counts)
        self.assertEqual(
            scenario._mpisppy_data.conv_iter_count,
            {("ROOT", 0): 3, ("ROOT", 1): 1})

        scenario = types.SimpleNamespace(_relaxed_integer_vars=object())
        opt = types.SimpleNamespace(local_scenarios={"scen0": scenario})
        relaxer = IntegerRelaxThenEnforce.__new__(IntegerRelaxThenEnforce)
        relaxer.opt = opt
        saved_relaxation = relaxer.checkpoint_model_state()
        relaxer.restore_checkpoint_model_state(saved_relaxation)
        self.assertTrue(relaxer._integers_relaxed)


class TestRebuildNLFingerprint(unittest.TestCase):
    def test_streamed_digest_matches_pinned_nl_writer_output(self):
        import hashlib
        import io

        import pyomo.environ as pyo
        from pyomo.repn.plugins.nl_writer import NLWriter
        from pyomo.repn.util import FileDeterminism

        model = pyo.ConcreteModel(name="nl-fingerprint-model")
        model.x = pyo.Var(bounds=(0, 10))
        model.c = pyo.Constraint(expr=model.x >= 1)
        model.obj = pyo.Objective(expr=2 * model.x)
        writer_options = {
            "file_determinism": FileDeterminism.SORT_SYMBOLS,
            "symbolic_solver_labels": True,
            "linear_presolve": False,
            "skip_trivial_constraints": False,
            "show_section_timing": False,
            "scale_model": True,
            "export_nonlinear_variables": None,
            "row_order": None,
            "column_order": None,
            "export_defined_variables": True,
        }

        stream = io.StringIO()
        NLWriter().write(model, stream, **writer_options)
        expected = hashlib.sha256(stream.getvalue().encode("utf-8")).hexdigest()
        digest = checkpointing._rebuild_model_nl_fingerprint(model)
        self.assertEqual(digest, expected)
        self.assertEqual(
            digest,
            checkpointing._rebuild_model_nl_fingerprint(model.clone()))

        model.x.setub(9)
        self.assertNotEqual(
            digest, checkpointing._rebuild_model_nl_fingerprint(model))
        model.x.setub(10)
        model.c.set_value(2 * model.x >= 2)
        self.assertNotEqual(
            digest, checkpointing._rebuild_model_nl_fingerprint(model))
        model.c.set_value(model.x >= 1)
        model.obj.set_value(3 * model.x)
        self.assertNotEqual(
            digest, checkpointing._rebuild_model_nl_fingerprint(model))

    def test_option_requires_rebuild_and_rejects_component_cache(self):
        options = {
            "checkpoint_structural_cfg": {},
            "checkpoint_backend": checkpointing.PYFORY_REBUILD_BACKEND,
            "checkpoint_nl_fingerprint": True,
        }
        checkpointing.require_checkpoint_structural_cfg(options)

        options["checkpoint_backend"] = checkpointing.DILL_RELOAD_BACKEND
        with self.assertRaisesRegex(RuntimeError, "requires the pyfory-rebuild"):
            checkpointing.require_checkpoint_structural_cfg(options)



@unittest.skipIf(not solver_available,
                 "no solver is available for checkpoint resume tests")
@unittest.skipUnless(checkpointing.pyfory_available,
                     "pyfory is an optional checkpoint dependency")
class TestPyforyRebuildNLFingerprint(unittest.TestCase):
    """Exercise NL digest publication and validation through a PH resume."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _options(self, max_iters, **overrides):
        return _options(
            max_iters,
            ckpt_dir=self.ckpt_dir,
            checkpoint_backend=checkpointing.PYFORY_REBUILD_BACKEND,
            checkpoint_nl_fingerprint=True,
            **overrides,
        )

    def _write_checkpoint(self, include_unused_state=False):
        writer = _make_ph(self._options(1))
        if include_unused_state:
            import pyomo.environ as pyo

            for scenario in writer.local_scenarios.values():
                scenario.nl_unused_checkpoint_state = pyo.Param(
                    mutable=True, initialize=1.0)
        writer.ph_main()
        return writer

    def _read_leaf(self, generation=1):
        generation_dir = os.path.join(
            self.ckpt_dir, "hub", f"gen_{generation:04d}")
        with open(os.path.join(generation_dir, "hub_rank_0000.pkl"),
                  "rb") as stream:
            return generation_dir, pickle.load(stream)

    def test_hash_files_are_published_and_saved_method_drives_fresh_resume(self):
        writer = self._write_checkpoint()
        generation_dir, leaf = self._read_leaf()
        expected_files = {
            sname: checkpointing._nl_fingerprint_filename(0, sname)
            for sname in writer.local_scenarios
        }
        self.assertEqual(leaf["model_fingerprint_method"],
                         checkpointing.NL_FINGERPRINT_METHOD)
        self.assertEqual(leaf["nl_fingerprint_files"], expected_files)
        self.assertEqual(
            sum(name.endswith(".nl.sha256")
                for name in os.listdir(generation_dir)),
            len(writer.local_scenarios))
        for sname, filename in expected_files.items():
            with open(os.path.join(generation_dir, filename), "rb") as stream:
                contents = stream.read()
            self.assertRegex(contents.decode("ascii"), r"^[0-9a-f]{64}\n$")

        loaded_leaf, snapshots = checkpointing.load_checkpoint(
            writer, self.ckpt_dir)
        expected_digests = {}
        for sname, filename in expected_files.items():
            with open(os.path.join(generation_dir, filename), "rb") as stream:
                expected_digests[sname] = stream.read()[:-1].decode("ascii")
        self.assertEqual(loaded_leaf["nl_fingerprints"], expected_digests)
        self.assertEqual(set(snapshots), set(writer.local_scenarios))

        # The runtime flag is off in the child. Restore must still validate by
        # the method saved in the checkpoint leaf.
        out_path = os.path.join(self._tmp.name, "nl-resumed.json")
        call_args = json.dumps([
            self.ckpt_dir, 0, out_path,
            checkpointing.PYFORY_REBUILD_BACKEND,
            {"checkpoint_nl_fingerprint": False},
        ])
        result = subprocess.run(
            [sys.executable, "-c",
             "import json, mpisppy.tests.test_checkpoint as t; "
             f"t._resume_and_dump(*json.loads({call_args!r}))"],
            capture_output=True, text=True, timeout=900, check=False,
            env=subprocess_env(),
        )
        self.assertEqual(result.returncode, 0,
                         msg=f"fresh-process NL resume failed:\n{result.stderr}")
        with open(out_path) as stream:
            resumed = json.load(stream)
        self.assertTrue(resumed["resumed"])
        self.assertEqual(resumed["resume_iteration"], 1)

    def test_missing_malformed_and_misnamed_hash_files_are_refused(self):
        writer = self._write_checkpoint()
        generation_dir, leaf = self._read_leaf()
        filename = leaf["nl_fingerprint_files"][SCENARIO_NAMES[0]]
        path = os.path.join(generation_dir, filename)
        with open(path, "rb") as stream:
            valid_contents = stream.read()

        os.unlink(path)
        with self.assertRaisesRegex(checkpointing.CheckpointMismatch,
                                    "NL fingerprint file"):
            checkpointing.load_checkpoint(writer, self.ckpt_dir)

        with open(path, "wb") as stream:
            stream.write(b"not-a-digest\n")
        with self.assertRaisesRegex(checkpointing.CheckpointMismatch,
                                    "exactly one lowercase"):
            checkpointing.load_checkpoint(writer, self.ckpt_dir)

        with open(path, "wb") as stream:
            stream.write(valid_contents)
        leaf["nl_fingerprint_files"][SCENARIO_NAMES[0]] = "unexpected.sha256"
        with open(os.path.join(generation_dir, "hub_rank_0000.pkl"),
                  "wb") as stream:
            pickle.dump(leaf, stream)
        with self.assertRaisesRegex(checkpointing.CheckpointMismatch,
                                    "exactly the NL fingerprint files"):
            checkpointing.load_checkpoint(writer, self.ckpt_dir)

    def test_final_validation_checks_nl_formulation_and_snapshot_values(self):
        import pyomo.environ as pyo

        writer = self._write_checkpoint(include_unused_state=True)
        leaf, snapshots = checkpointing.load_checkpoint(
            writer, self.ckpt_dir)
        scenario = writer.local_scenarios[SCENARIO_NAMES[0]]
        objective = next(scenario.component_data_objects(
            pyo.Objective, active=True))
        objective_expr = objective.expr
        objective.set_value(objective_expr + 1)
        with self.assertRaisesRegex(checkpointing.CheckpointMismatch,
                                    "NL formulation.*does not match"):
            checkpointing.validate_rebuild_model_state(
                writer, snapshots,
                fingerprint_method=leaf["model_fingerprint_method"],
                nl_fingerprints=leaf["nl_fingerprints"])

        objective.set_value(objective_expr)
        unused_param = scenario.nl_unused_checkpoint_state
        saved_digest = leaf["nl_fingerprints"][SCENARIO_NAMES[0]]
        unused_param.set_value(2.0)
        self.assertEqual(
            checkpointing._rebuild_model_nl_fingerprint(scenario),
            saved_digest,
            "an unused mutable Param should not affect the NL formulation")
        with self.assertRaisesRegex(
                checkpointing.CheckpointMismatch,
                r"checkpoint snapshot \(mutable_param_values\)"):
            checkpointing.validate_rebuild_model_state(
                writer, snapshots,
                fingerprint_method=leaf["model_fingerprint_method"],
                nl_fingerprints=leaf["nl_fingerprints"])

    def test_failed_hash_write_keeps_the_previous_generation(self):
        writer = self._write_checkpoint()
        original_manifest = checkpointing._read_manifest(self.ckpt_dir)
        original_atomic_write = checkpointing._atomic_write_bytes

        def fail_nl_hash(path, callback, timing=None):
            if path.endswith(".nl.sha256"):
                raise OSError("injected NL hash write failure")
            return original_atomic_write(path, callback, timing=timing)

        with mock.patch.object(checkpointing, "_atomic_write_bytes",
                               side_effect=fail_nl_hash):
            with self.assertRaisesRegex(
                    (OSError, RuntimeError), "injected NL hash write failure"):
                checkpointing.write_checkpoint(
                    writer, self.ckpt_dir, 2,
                    backend=checkpointing.PYFORY_REBUILD_BACKEND)

        self.assertEqual(checkpointing._read_manifest(self.ckpt_dir),
                         original_manifest)
        self.assertTrue(os.path.isdir(os.path.join(
            self.ckpt_dir, "hub", "gen_0001")))
        self.assertFalse(os.path.exists(os.path.join(
            self.ckpt_dir, "hub", "gen_0002.tmp")))


class TestPyforyRebuildDependency(unittest.TestCase):
    def test_pyfory_minimum_version_is_accepted(self):
        with mock.patch.object(checkpointing, "pyfory_available", True):
            for version in ("1.7.7", "1.7.8", "2.0.0"):
                with self.subTest(version=version), mock.patch.object(
                        checkpointing, "_pyfory_version", return_value=version):
                    checkpointing.require_backend_dependency(
                        checkpointing.PYFORY_REBUILD_BACKEND)

    def test_older_or_unrecognized_pyfory_version_is_refused(self):
        with mock.patch.object(checkpointing, "pyfory_available", True):
            for version in ("1.7.6", "1.7.7rc1", "unknown"):
                with self.subTest(version=version), mock.patch.object(
                        checkpointing, "_pyfory_version", return_value=version):
                    with self.assertRaises(RuntimeError):
                        checkpointing.require_backend_dependency(
                            checkpointing.PYFORY_REBUILD_BACKEND)

    def test_pyfory_backend_is_refused_at_setup_when_dependency_is_missing(self):
        opt = _make_ph(_options(1))
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            opt.options["checkpoint_dir"] = checkpoint_dir
            opt.options["checkpoint_backend"] = (
                checkpointing.PYFORY_REBUILD_BACKEND)
            with mock.patch.object(checkpointing, "pyfory_available", False):
                with self.assertRaisesRegex(RuntimeError, "requires pyfory"):
                    Checkpointer(opt)

    def test_missing_pyfory_is_a_setup_error_for_its_backend(self):
        with mock.patch.object(checkpointing, "pyfory_available", False):
            with self.assertRaisesRegex(RuntimeError, "requires pyfory"):
                checkpointing.require_backend_dependency(
                    checkpointing.PYFORY_REBUILD_BACKEND)
