"""Graph invariants specific to the full-model pyfory reload codec."""

import datetime
import decimal
import io
import importlib.util
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from collections import OrderedDict, defaultdict, namedtuple
from fractions import Fraction

import pyomo.environ as pyo
from pyomo.common.collections import Bunch
from pyomo.core.base.PyomoModel import PyomoConfig

if importlib.util.find_spec("pyfory") is None:
    codec = None
else:
    from mpisppy.utils import pyfory_model_codec as codec

reload_runtime_available = False
if codec is not None:
    try:
        codec.require_supported_runtime()
    except (RuntimeError, TypeError):
        pass
    else:
        reload_runtime_available = True


class _OwnDictionaryView:
    def __init__(self):
        self.view = self.__dict__.keys()


class _CustomDict(dict):
    pass


class _SlottedPayload:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class _ListPayload(list):
    pass


class _SetPayload(set):
    pass


class _FrozenPayload(frozenset):
    pass


_TuplePayload = namedtuple("_TuplePayload", "left right")


@unittest.skipUnless(reload_runtime_available,
                     "pyfory-reload's tested runtime is unavailable")
class TestPyforyModelCodec(unittest.TestCase):
    def _roundtrip(self, graph):
        model = pyo.ConcreteModel()
        model.graph = graph
        stream = io.BytesIO()
        codec.dump(model, stream)
        return codec.load(io.BytesIO(stream.getvalue())).graph

    def test_views_preserve_owner_cycles_aliases_and_mutations(self):
        left = {("tuple", 1): [1, 2]}
        right = {("tuple", 1): [1, 2]}
        keys = left.keys()
        values = left.values()
        items = left.items()
        distinct_keys = left.keys()
        left["self"] = left
        left["view"] = keys
        right["left"] = left
        left["right"] = right
        restored = self._roundtrip([
            left, keys, keys, distinct_keys, values, items, right,
            (keys, values), right.keys(),
        ])
        owner, saved_keys, alias, another, saved_values, saved_items, peer = \
            restored[:7]
        self.assertIs(owner, codec.backing_mapping(saved_keys))
        self.assertIs(saved_keys, alias)
        self.assertIsNot(saved_keys, another)
        self.assertIs(codec.backing_mapping(another), owner)
        self.assertIs(codec.backing_mapping(saved_values), owner)
        self.assertIs(codec.backing_mapping(saved_items), owner)
        self.assertIs(owner["self"], owner)
        self.assertIs(owner["view"], saved_keys)
        self.assertIs(peer["left"], owner)
        self.assertIs(owner["right"], peer)
        self.assertIs(restored[7][0], saved_keys)
        self.assertIs(restored[7][1], saved_values)
        self.assertIs(codec.backing_mapping(restored[8]), peer)
        self.assertIsNot(owner, peer)
        owner["new"] = 7
        self.assertIn("new", saved_keys)
        self.assertIn(7, saved_values)
        self.assertIn(("new", 7), saved_items)
        del owner["new"]
        self.assertNotIn("new", saved_keys)
        owner["new"] = 9
        self.assertEqual(list(owner)[-1], "new")

    def test_view_first_and_view_only_owners(self):
        for kind in ("keys", "values", "items"):
            with self.subTest(kind=kind):
                source = {}
                view = getattr(source, kind)()
                source["view"] = view
                restored_view, alias = self._roundtrip((view, view))
                self.assertIs(restored_view, alias)
                owner = codec.backing_mapping(restored_view)
                self.assertIs(owner["view"], restored_view)
                owner["fresh"] = 4
                self.assertIn({"keys": "fresh", "values": 4,
                               "items": ("fresh", 4)}[kind], restored_view)
                empty_owner = {}
                empty_view = self._roundtrip(getattr(empty_owner, kind)())
                restored_owner = codec.backing_mapping(empty_view)
                self.assertEqual(len(restored_owner), 0)
                restored_owner["later"] = 5
                self.assertIn({"keys": "later", "values": 5,
                               "items": ("later", 5)}[kind], empty_view)

    def test_bunch_and_pyomo_config_keep_concrete_owner(self):
        for cls in (Bunch, PyomoConfig):
            with self.subTest(cls=cls):
                owner = cls()
                dict.__setitem__(owner, ("ROOT", 0), {"raw": 1})
                view = owner.items()
                dict.__setitem__(owner, "view", view)
                attrs = owner.__dict__
                attrs["self_dict"] = attrs
                attrs["own_view"] = attrs.keys()
                restored, saved_view, saved_attrs = self._roundtrip(
                    [owner, view, attrs])
                self.assertIs(type(restored), cls)
                self.assertIs(codec.backing_mapping(saved_view), restored)
                self.assertIs(restored["view"], saved_view)
                self.assertIs(restored.__dict__, saved_attrs)
                self.assertIs(saved_attrs["self_dict"], saved_attrs)
                self.assertIs(codec.backing_mapping(saved_attrs["own_view"]),
                              restored.__dict__)
                self.assertIsInstance(dict.__getitem__(restored, ("ROOT", 0)),
                                      dict)
                restored["extra"] = 3
                self.assertIn(("extra", 3), saved_view)

    def test_local_recursive_callback_and_shared_cell(self):
        shared = {"calls": 0}

        def callback(value):
            shared["calls"] += 1
            return callback if value == 0 else shared["calls"]

        def observer():
            return shared["calls"]

        restored_callback, restored_observer = self._roundtrip(
            (callback, observer))
        self.assertIs(restored_callback(0), restored_callback)
        self.assertEqual(restored_callback(1), 2)
        self.assertEqual(restored_observer(), 2)
        self.assertIs(restored_callback.__closure__[1],
                      restored_observer.__closure__[0])

    def test_fresh_process_reader_and_file_identity(self):
        owner = {"existing": 1}
        model = pyo.ConcreteModel()
        model.graph = (owner.keys(), owner)
        model.ordered = OrderedDict([("second", 2), ("first", 1)])
        model.defaults = defaultdict(list)
        def callback(value):
            return value + 1
        callback.__name__ = "renamed_callback"
        callback.__doc__ = "saved callback documentation"
        model.callback = callback
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.fory")
            with open(path, "wb") as stream:
                codec.dump(model, stream, scenario_name="scen0",
                           write_id="test-write")
            script = (
                "from mpisppy.utils import pyfory_model_codec as c\n"
                "import sys\n"
                "with open(sys.argv[1], 'rb') as f:\n"
                "    m = c.load(f, scenario_name='scen0', "
                "write_id='test-write')\n"
                "v, d = m.graph\n"
                "assert c.backing_mapping(v) is d\n"
                "d['new'] = 2\n"
                "assert 'new' in v\n"
                "assert m.callback.__name__ == 'renamed_callback'\n"
                "assert m.callback.__doc__ == 'saved callback documentation'\n"
                "assert m.callback(2) == 3\n"
                "assert list(m.ordered) == ['second', 'first']\n"
                "m.defaults['new'].append(4)\n"
                "assert m.defaults['new'] == [4]\n"
            )
            result = subprocess.run([sys.executable, "-c", script, path],
                                    capture_output=True, text=True,
                                    timeout=30, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(path, "rb") as stream:
                with self.assertRaisesRegex(ValueError, "write identity"):
                    codec.load(stream, write_id="wrong")

    def test_common_metadata_and_container_subclasses_in_fresh_process(self):
        model = pyo.ConcreteModel()
        timezone = datetime.timezone(datetime.timedelta(hours=-7))
        timestamp = datetime.datetime(2026, 11, 6, 12, 30)
        wall_time = datetime.time(12, 30, 45, 123456, fold=1)
        wrapped = _ListPayload([1, 2])
        wrapped.note = "preserved"
        wrapped.append(wrapped)
        marked_set = _SetPayload(["one", "two"])
        marked_set.note = "set state"
        model.graph = {
            "timestamp": timestamp,
            "timestamp_alias": timestamp,
            "aware_timestamp": datetime.datetime(
                2026, 11, 1, 1, 30, tzinfo=timezone, fold=1),
            "wall_time": wall_time,
            "wall_time_alias": wall_time,
            "aware_wall_time": datetime.time(
                1, 30, 45, 123456, tzinfo=timezone, fold=1),
            "timezone": timezone,
            "decimal": decimal.Decimal("1.25"),
            "path": pathlib.Path("/tmp/model.dat"),
            "fraction": Fraction(2**100, 3),
            "uuid": uuid.UUID("01234567-89ab-cdef-0123-456789abcdef"),
            "slots": _SlottedPayload(17),
            "list": wrapped,
            "tuple": _TuplePayload(3, "four"),
            "set": marked_set,
            "frozenset": _FrozenPayload(["alpha", "beta"]),
            "struct_time": time.struct_time((2026, 11, 6, 12, 30, 0, 4, 310, 0)),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.fory")
            with open(path, "wb") as stream:
                codec.dump(model, stream)
            script = (
                "import datetime, decimal, pathlib, sys, time, uuid\n"
                "from fractions import Fraction\n"
                "from mpisppy.utils import pyfory_model_codec as c\n"
                "with open(sys.argv[1], 'rb') as stream:\n"
                "    graph = c.load(stream).graph\n"
                "assert graph['timestamp'] == datetime.datetime(2026, 11, 6, 12, 30)\n"
                "assert graph['timestamp_alias'] is graph['timestamp']\n"
                "assert graph['timestamp'].tzinfo is None\n"
                "assert graph['aware_timestamp'].fold == 1\n"
                "assert graph['aware_timestamp'].tzinfo is graph['timezone']\n"
                "assert graph['aware_timestamp'].utcoffset() == datetime.timedelta(hours=-7)\n"
                "assert graph['wall_time'] == datetime.time(12, 30, 45, 123456)\n"
                "assert graph['wall_time'].fold == 1\n"
                "assert graph['wall_time_alias'] is graph['wall_time']\n"
                "assert graph['aware_wall_time'].hour == 1\n"
                "assert graph['aware_wall_time'].minute == 30\n"
                "assert graph['aware_wall_time'].second == 45\n"
                "assert graph['aware_wall_time'].microsecond == 123456\n"
                "assert graph['aware_wall_time'].fold == 1\n"
                "assert graph['aware_wall_time'].tzinfo is graph['timezone']\n"
                "assert graph['decimal'] == decimal.Decimal('1.25')\n"
                "assert graph['path'] == pathlib.Path('/tmp/model.dat')\n"
                "assert graph['fraction'] == Fraction(2**100, 3)\n"
                "assert graph['uuid'] == uuid.UUID('01234567-89ab-cdef-0123-456789abcdef')\n"
                "assert graph['slots'].value == 17\n"
                "assert type(graph['list']).__name__ == '_ListPayload'\n"
                "assert graph['list'][:2] == [1, 2]\n"
                "assert graph['list'][2] is graph['list']\n"
                "assert graph['list'].note == 'preserved'\n"
                "assert tuple(graph['tuple']) == (3, 'four')\n"
                "assert graph['set'] == {'one', 'two'}\n"
                "assert graph['set'].note == 'set state'\n"
                "assert type(graph['frozenset']).__name__ == '_FrozenPayload'\n"
                "assert graph['frozenset'] == {'alpha', 'beta'}\n"
                "assert graph['struct_time'].tm_year == 2026\n"
            )
            result = subprocess.run([sys.executable, "-X", "faulthandler",
                                     "-c", script, path],
                                    capture_output=True, text=True,
                                    timeout=30, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_main_module_callback_in_fresh_process(self):
        writer = (
            "import sys\n"
            "import pyomo.environ as pyo\n"
            "from mpisppy.utils import pyfory_model_codec as c\n"
            "SCALE = 2\n"
            "def callback(value):\n"
            "    return value + SCALE\n"
            "model = pyo.ConcreteModel()\n"
            "model.callback = callback\n"
            "with open(sys.argv[1], 'wb') as stream:\n"
            "    c.dump(model, stream)\n"
        )
        reader = (
            "import sys\n"
            "from mpisppy.utils import pyfory_model_codec as c\n"
            "with open(sys.argv[1], 'rb') as stream:\n"
            "    model = c.load(stream)\n"
            "assert model.callback(3) == 5\n"
            "assert model.callback.__globals__['SCALE'] == 2\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.fory")
            for script in (writer, reader):
                result = subprocess.run(
                    [sys.executable, "-c", script, path],
                    capture_output=True, text=True, timeout=30,
                    check=False)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_main_module_class_is_refused_before_checkpoint_write(self):
        cls = type("DriverPayload", (), {"__module__": "__main__"})
        model = pyo.ConcreteModel()
        model.graph = cls()
        stream = io.BytesIO()
        with self.assertRaisesRegex(TypeError, "stable import path"):
            codec.dump(model, stream)
        self.assertEqual(stream.getvalue(), b"")

    def test_cyclic_immutable_container_subclass_is_refused(self):
        child = []
        wrapped = _TuplePayload(child, 1)
        child.append(wrapped)
        model = pyo.ConcreteModel()
        model.graph = wrapped
        with self.assertRaisesRegex(TypeError,
                                    "cyclic immutable container subclass"):
            codec.dump(model, io.BytesIO())

    def test_unsupported_instance_dictionary_view_is_refused(self):
        model = pyo.ConcreteModel()
        model.graph = _OwnDictionaryView()
        with self.assertRaisesRegex(TypeError, "own instance dictionary"):
            codec.dump(model, io.BytesIO())

    def test_ordered_and_default_mappings_preserve_behavior(self):
        ordered = OrderedDict([("second", 2), ("first", 1)])
        view = ordered.keys()
        values_view = ordered.values()
        items_view = ordered.items()
        ordered["view"] = view
        ordered.note = {"owner": ordered}
        attributes_view = ordered.__dict__.keys()
        defaults = defaultdict(lambda: [])
        defaults["owner"] = defaults
        defaults_view = defaults.items()
        restored_view, restored_values_view, restored_items_view, \
            restored_ordered, alias, restored_attributes_view, \
            restored_defaults, restored_defaults_view = self._roundtrip(
                (view, values_view, items_view, ordered, ordered,
                 attributes_view, defaults, defaults_view))
        self.assertIs(type(restored_ordered), OrderedDict)
        self.assertIs(restored_view, restored_ordered["view"])
        self.assertIs(codec.backing_mapping(restored_view), restored_ordered)
        self.assertIs(codec.backing_mapping(restored_values_view),
                      restored_ordered)
        self.assertIs(codec.backing_mapping(restored_items_view),
                      restored_ordered)
        self.assertIs(restored_ordered, alias)
        self.assertEqual(list(restored_ordered),
                         ["second", "first", "view"])
        self.assertIs(restored_ordered.note["owner"], restored_ordered)
        self.assertIs(codec.backing_mapping(restored_attributes_view),
                      restored_ordered.__dict__)
        restored_ordered["last"] = 3
        self.assertEqual(list(restored_ordered)[-1], "last")
        self.assertIn("last", restored_view)
        self.assertIn(3, restored_values_view)
        self.assertIn(("last", 3), restored_items_view)
        self.assertIs(type(restored_defaults), defaultdict)
        self.assertIs(restored_defaults["owner"], restored_defaults)
        self.assertIs(codec.backing_mapping(restored_defaults_view),
                      restored_defaults)
        restored_defaults["new"].append(4)
        self.assertEqual(restored_defaults["new"], [4])
        self.assertIn(("new", [4]), restored_defaults_view)

    def test_unsupported_mapping_has_a_graph_path(self):
        model = pyo.ConcreteModel()
        model.graph = [_CustomDict(value=1)]
        with self.assertRaisesRegex(TypeError,
                                    r"unknown.*Unsupported mapping subclass"):
            codec.dump(model, io.BytesIO())

    def test_corrupt_type_table_and_truncated_header_are_refused(self):
        model = pyo.ConcreteModel()
        stream = io.BytesIO()
        codec.dump(model, stream)
        data = stream.getvalue()
        with self.assertRaisesRegex(ValueError, "Truncated"):
            codec.load(io.BytesIO(data[:len(codec.MAGIC) + 3]))
        start = len(codec.MAGIC)
        size = struct.unpack(">I", data[start:start + 4])[0]
        header = json.loads(data[start + 4:start + 4 + size])
        header["types"].append(header["types"][0])
        raw = json.dumps(header).encode("utf-8")
        corrupted = (codec.MAGIC + struct.pack(">I", len(raw)) + raw
                     + data[start + 4 + size:])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            codec.load(io.BytesIO(corrupted))

    def test_malformed_owner_and_cell_tokens_are_refused(self):
        tables = codec._Tables()
        with self.assertRaisesRegex(ValueError, "Nonconsecutive"):
            tables.shell(1, 0)
        with self.assertRaisesRegex(ValueError, "Nonconsecutive"):
            tables.cell_shell(1)
        tables.shell(0, 0)
        with self.assertRaisesRegex(ValueError, "Undefined dictionary owner"):
            tables.finish()


def _uc_snapshot(model):
    """Compare model state by component identity rather than traversal order."""
    return {
        "variables": {
            var.name: (var.value, var.fixed, str(var.domain), var.lb,
                       var.ub, var.stale)
            for var in model.component_data_objects(pyo.Var)
        },
        "parameters": {
            component.name + "[" + repr(key) + "]":
                pyo.value(value, exception=False)
            for component in model.component_objects(pyo.Param)
            for key, value in component.items()
        },
        "constraints": {
            con.name: (str(con.expr), con.active)
            for con in model.component_data_objects(pyo.Constraint)
        },
        "objectives": {
            obj.name: (str(obj.expr), obj.active, str(obj.sense))
            for obj in model.component_data_objects(pyo.Objective)
        },
    }


@unittest.skipUnless(reload_runtime_available,
                     "pyfory-reload's tested runtime is unavailable")
class TestPyforyUcGraph(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "_egret-int", "egret", "__init__.py")),
        "local Egret checkout is unavailable")
    def test_constructed_uc_graph_in_fresh_process(self):
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        sys.path.insert(0, os.path.join(root, "_egret-int"))
        try:
            spec = importlib.util.spec_from_file_location(
                "pyfory_uc_test_creator",
                os.path.join(root, "examples", "uc", "uc_funcs.py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            model = module.scenario_creator(
                "Scenario1", num_scens=25,
                path=os.path.join(root, "examples", "uc", "25scenarios_r1"))
            owner = {"existing": 1}
            view = owner.keys()
            owner["view"] = view
            model._reload_view_test = (view, owner, view)
            expected = _uc_snapshot(model)
            with tempfile.TemporaryDirectory() as folder:
                model_path = os.path.join(folder, "model.fory")
                expected_path = os.path.join(folder, "expected.json")
                with open(model_path, "wb") as stream:
                    codec.dump(model, stream, scenario_name="Scenario1",
                               write_id="uc-test")
                with open(expected_path, "w") as stream:
                    json.dump(expected, stream)
                script = (
                    "import json, sys\n"
                    "sys.path.insert(0, sys.argv[3])\n"
                    "from mpisppy.utils import pyfory_model_codec as c\n"
                    "from mpisppy.tests.test_pyfory_model_codec "
                    "import _uc_snapshot\n"
                    "with open(sys.argv[1], 'rb') as f:\n"
                    "    m = c.load(f, scenario_name='Scenario1', "
                    "write_id='uc-test')\n"
                    "with open(sys.argv[2]) as f:\n"
                    "    expected = json.load(f)\n"
                    "assert json.loads(json.dumps(_uc_snapshot(m))) "
                    "== expected\n"
                    "v, d, alias = m._reload_view_test\n"
                    "assert v is alias and c.backing_mapping(v) is d\n"
                    "d['new'] = 2\n"
                    "assert 'new' in v\n"
                    "callback = m.Impedence._validate\n"
                    "assert any(cell.cell_contents is callback "
                    "for cell in callback.__closure__)\n"
                    "callback.print_negative_reactance_warning = True\n"
                    "assert callback(m, -1.0, 'test') is True\n"
                    "assert callback.print_negative_reactance_warning "
                    "is False\n"
                    "ref = m.Contingencies._init_values._init.val\n"
                    "if type(ref) not in c.VIEW_KINDS: ref = ref._ref\n"
                    "source = c.backing_mapping(ref)\n"
                    "count = len(m.Contingencies)\n"
                    "source['new'] = 1\n"
                    "assert 'new' in ref and len(m.Contingencies) == count\n"
                    "for node in m._mpisppy_node_list:\n"
                    "    for var in node.nonant_vardata_list:\n"
                    "        assert var.model() is m\n"
                )
                result = subprocess.run(
                    [sys.executable, "-c", script, model_path,
                     expected_path, os.path.join(root, "_egret-int")],
                    capture_output=True, text=True, timeout=180,
                    check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
        finally:
            sys.path.remove(os.path.join(root, "_egret-int"))


if __name__ == "__main__":
    unittest.main()
