"""Serial checks for the development full-model codec's PH index handling."""

import io
import unittest
from unittest import mock

import pyomo.environ as pyo

import pyfory_reload_probe as probe


def legacy_model():
    model = pyo.ConcreteModel()
    model.x = pyo.Var([0, 1], initialize=2.0)
    model._mpisppy_data = pyo.Block()
    model._mpisppy_data.nonant_indices = {
        ('ROOT', i): model.x[i] for i in model.x
    }
    model._mpisppy_model = pyo.Block()
    for name in probe.PH_INDEXED_PARAMS:
        model._mpisppy_model.add_component(name, pyo.Param(
            model._mpisppy_data.nonant_indices.keys(),
            initialize=3.0, mutable=True))
    return model


def index_reference(param):
    return param.index_set()._init_values._init.val


class TestLegacyPHIndexNormalization(unittest.TestCase):
    def test_roundtrip_preserves_params_without_mutating_live_views(self):
        model = legacy_model()
        originals = [index_reference(model._mpisppy_model.component(name))
                     for name in probe.PH_INDEXED_PARAMS]
        views = [ref._ref for ref in originals]
        writer, _ = probe.codec('cycle', model)
        stream = io.BytesIO()
        writer.dump(model, stream)
        reader, _ = probe.codec('cycle', legacy_model())
        restored = reader.deserialize(stream.getvalue())
        for name in probe.PH_INDEXED_PARAMS:
            param = restored._mpisppy_model.component(name)
            self.assertEqual(tuple(param), (('ROOT', 0), ('ROOT', 1)))
            self.assertEqual(tuple(index_reference(param)._ref), tuple(param))
            self.assertIsInstance(index_reference(param)._ref, tuple)
            self.assertEqual(pyo.value(param['ROOT', 0]), 3.0)
            param['ROOT', 0] = 4.0
            self.assertEqual(pyo.value(param['ROOT', 0]), 4.0)
        for ref, view in zip(originals, views):
            self.assertIs(ref._ref, view)
            self.assertIsInstance(view, type({}.keys()))
        self.assertIs(restored._mpisppy_data.nonant_indices['ROOT', 0],
                      restored.x[0])

    def test_failed_write_does_not_mutate_the_initializer(self):
        model = legacy_model()
        ref = index_reference(model._mpisppy_model.W)
        original = ref._ref
        codec, _ = probe.codec('cycle', model)
        serializer = probe.CycleStateSerializer(
            codec.type_resolver, type(ref), probe.ph_index_normalizations(model))
        context = mock.Mock()
        context.write_ref.side_effect = RuntimeError('injected encode failure')
        with self.assertRaisesRegex(RuntimeError, 'injected encode failure'):
            serializer.write(context, ref)
        self.assertIs(ref._ref, original)
        self.assertIsInstance(context.write_ref.call_args.args[0][-1]['_ref'],
                              tuple)

    def test_unrelated_views_are_not_normalized_even_with_the_same_keys(self):
        model = legacy_model()
        model.user_param = pyo.Param(
            model._mpisppy_data.nonant_indices.keys(),
            initialize=1, mutable=True)
        ref = index_reference(model.user_param)
        normalizations = probe.ph_index_normalizations(model)
        self.assertNotIn(id(ref), normalizations)
        state = probe.checkpoint_object_state(ref, normalizations)
        self.assertIs(state[-1]['_ref'], ref._ref)
        writer, _ = probe.codec('cycle', model)
        payload = writer.serialize(model)
        reader, _ = probe.codec('cycle', model)
        with self.assertRaisesRegex(ValueError, 'dict_keys is not registered'):
            reader.deserialize(payload)

    def test_changed_nonant_membership_is_refused(self):
        model = legacy_model()
        model._mpisppy_data.nonant_indices['ROOT', 2] = model.x[0]
        with self.assertRaisesRegex(ValueError, 'indices differ'):
            probe.codec('cycle', model)


class TestPHParameterIndices(unittest.TestCase):
    def test_all_ph_parameter_indices_are_snapshots(self):
        from types import SimpleNamespace
        from mpisppy.phbase import PHBase

        model = legacy_model()
        for name in probe.PH_INDEXED_PARAMS:
            model._mpisppy_model.del_component(name)
        opt = SimpleNamespace(local_scenarios={'scenario': model}, Ag=None,
                              options={'defaultPHrho': 2, 'defaultPHp': 0.1,
                                       'defaultPHbeta': 0.1})
        PHBase.attach_Ws_and_prox(opt)
        PHBase.attach_smoothing(opt)
        PHBase.attach_xbars(opt)
        keys = tuple(model._mpisppy_data.nonant_indices)
        for name in probe.PH_INDEXED_PARAMS:
            param = model._mpisppy_model.component(name)
            self.assertEqual(tuple(param), keys)
            self.assertEqual(index_reference(param)._ref, keys)
            self.assertIsInstance(index_reference(param)._ref, tuple)
        self.assertEqual(probe.ph_index_normalizations(model), {})


if __name__ == '__main__':
    unittest.main()
