"""Serial development probe for full-model pyfory compatibility.

This is not a checkpoint backend. Readers construct a matching seed model to
register types; a production codec needs a stable, independent registry.
The custom adapters use public pyfory APIs and Pyomo state hooks, plus Pyomo's
private global-set lookup. Payloads are temporary diagnostic files.

Example (two separate processes):
    .venv/bin/python pyfory_reload_probe.py write cycle indexed /tmp/model.fory
    .venv/bin/python pyfory_reload_probe.py read cycle indexed /tmp/model.fory

The farmer and ph cases import mpi-sppy and use singleton MPI. The ph case
runs two PH iterations with linearized proximal terms using the test solver.
No MPI launcher is used. A reported compatibility failure exits with code 1.
"""

import argparse
import io
import json
import types
import enum
import sys
from pyomo.common.collections import Bunch
from pyomo.core.base.global_set import GlobalSetBase
import pyfory
import pyomo.environ as pyo
from pyomo.core.base.initializer import ConstantInitializer
from pyomo.core.base.set import FiniteSetOf, TuplizeValuesInitializer

OPTIONS = dict(xlang=False, ref=True, strict=False,
               max_depth=10000, max_graph_memory_bytes=1 << 30)

PH_INDEXED_PARAMS = ('W', 'rho', 'z', 'p', 'beta', 'xbars', 'xsqbars')


def ph_index_normalizations(model):
    """Identify legacy views retained only as PH parameter index initializers.

    PH's nonant membership is fixed after setup. Do not normalize arbitrary
    views: dill preserves their connection to the original dictionary on
    supported CPython versions. Keep strong references for identity checks.
    This deliberately recognizes the Pyomo 6.10 initializer shape only.
    """
    block = getattr(model, '_mpisppy_model', None)
    data = getattr(model, '_mpisppy_data', None)
    indices = getattr(data, 'nonant_indices', None)
    if block is None or indices is None:
        return {}
    keys = tuple(indices)
    normalizations = {}
    for name in PH_INDEXED_PARAMS:
        param = block.component(name)
        if param is None or param.ctype is not pyo.Param:
            continue
        initializer = getattr(param.index_set(), '_init_values', None)
        if type(initializer) is not TuplizeValuesInitializer:
            continue
        initializer = initializer._init
        if type(initializer) is not ConstantInitializer:
            continue
        reference = initializer.val
        if (type(reference) is not FiniteSetOf
                or type(reference._ref) is not type({}.keys())):
            continue
        if tuple(reference._ref) != keys or tuple(param.index_set()) != keys:
            raise ValueError(f'{param.name} indices differ from PH nonant indices')
        normalizations[id(reference)] = (reference, keys)
    return normalizations


def checkpoint_object_state(obj, normalizations):
    """Normalize a copy of the serialized state without editing the model."""
    state = obj.__getstate__()
    entry = normalizations.get(id(obj))
    if entry is not None and entry[0] is obj:
        # AutoSlots places the copied instance dictionary at the end. Copy
        # again explicitly so neither the live object nor shared state changes.
        state = list(state)
        fields = dict(state[-1])
        if fields.get('_ref') is not obj._ref:
            raise ValueError('Unexpected Pyomo SetOf state layout')
        fields['_ref'] = entry[1]
        state[-1] = fields
    return state


class CycleStateSerializer(pyfory.Serializer):
    def __init__(self, resolver, cls, normalizations):
        super().__init__(resolver, cls)
        self.normalizations = normalizations

    def write(self, ctx, obj):
        ctx.write_ref(checkpoint_object_state(obj, self.normalizations))

    def read(self, ctx):
        obj = self.type_.__new__(self.type_)
        ctx.reference(obj)
        state = ctx.read_ref()
        if hasattr(self.type_, '__setstate__'):
            obj.__setstate__(state)
        else:
            obj.__dict__.update(state)
        return obj


class GlobalSerializer(pyfory.Serializer):
    def write(self, ctx, obj):
        ctx.write_string(obj.name)

    def read(self, ctx):
        from pyomo.core.base.global_set import _get_global_set
        return _get_global_set(ctx.read_string())


class BunchSerializer(pyfory.Serializer):
    def write(self, ctx, obj):
        ctx.write_ref(dict(obj))
        ctx.write_ref(obj.__dict__)

    def read(self, ctx):
        obj = self.type_()
        ctx.reference(obj)
        obj.update(ctx.read_ref())
        obj.__dict__.update(ctx.read_ref())
        return obj


def make_model(case):
    if case == 'farmer':
        from mpisppy.tests.examples import farmer
        return farmer.scenario_creator('scen0', use_integer=False, crops_multiplier=1)
    if case == 'ph':
        from mpisppy.tests.test_checkpoint import _make_ph, _options
        ph = _make_ph(_options(2, linearize_proximal_terms=True))
        ph.ph_main()
        m = ph.local_scenarios['scen0']
        del m._solver_plugin
        return m
    m = pyo.ConcreteModel(name='probe')
    if case == 'empty':
        return m
    m.x = pyo.Var(initialize=2.0)
    if case == 'indexed':
        from pyomo.common.collections import ComponentMap, ComponentSet
        m.y = pyo.Var([0, 1], initialize=3.0)
        m.c = pyo.Constraint(expr=m.x + m.y[0] >= 1)
        m.obj = pyo.Objective(expr=m.x + m.y[0] + m.y[1])
        m.alias = [m.x, m.x]
        m.cmap = ComponentMap([(m.x, 'x')])
        m.cset = ComponentSet([m.x])
    return m


def graph_types(root, normalizations=None):
    normalizations = normalizations or {}
    seen = {}
    classes = {}
    def walk(obj):
        if id(obj) in seen:
            return
        seen[id(obj)] = obj
        if isinstance(obj, (type, types.FunctionType, types.ModuleType)):
            return
        if isinstance(obj, (Bunch, enum.Enum)):
            classes[type(obj)] = ''
        if type(obj).__module__.startswith(('pyomo.', 'mpisppy.')) and not isinstance(obj, (Bunch, enum.Enum)):
            classes[type(obj)] = obj.name if isinstance(obj, GlobalSetBase) else ''
            walk(checkpoint_object_state(obj, normalizations))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(k)
                walk(v)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                walk(v)
    walk(root)
    return sorted(classes, key=lambda c: (c.__module__, c.__qualname__, classes[c]))


def codec(mode, model):
    f = pyfory.Fory(**OPTIONS)
    normalizations = ph_index_normalizations(model) if mode == 'cycle' else {}
    classes = [] if mode == 'native' else ([type(model)] if mode == 'registered' else graph_types(model, normalizations))
    for i, cls in enumerate(classes):
        kw = {}
        if mode == 'stateful' and hasattr(cls, '__setstate__'):
            kw['serializer'] = pyfory.StatefulSerializer(f.type_resolver, cls)
        elif mode == 'cycle':
            if issubclass(cls, GlobalSetBase):
                kw['serializer'] = GlobalSerializer(f.type_resolver, cls)
            elif issubclass(cls, Bunch):
                kw['serializer'] = BunchSerializer(f.type_resolver, cls)
            elif not issubclass(cls, enum.Enum):
                kw['serializer'] = CycleStateSerializer(
                    f.type_resolver, cls, normalizations)
        f.register_type(cls, type_id=9000+i, **kw)
    return f, [c.__module__+'.'+c.__qualname__ for c in classes]


def validate(m, case, expected):
    if case in ('farmer', 'ph'):
        assert m.name == expected.name, 'scenario name changed'
        original_vars = list(expected.component_data_objects(pyo.Var))
        restored_vars = list(m.component_data_objects(pyo.Var))
        assert len(original_vars) == len(restored_vars)
        for old, new in zip(original_vars, restored_vars):
            assert new.parent_component().model() is m, 'variable owner changed'
            assert (new.name, new.value, new.fixed) == (old.name, old.value, old.fixed)
        for node in m._mpisppy_node_list:
            for var in node.nonant_vardata_list:
                assert var is m.find_component(var.name), 'nonant identity lost'
        old_expr = [str(o.expr) for o in expected.component_data_objects(pyo.Objective)]
        new_expr = [str(o.expr) for o in m.component_data_objects(pyo.Objective)]
        assert old_expr == new_expr, 'objective changed'
        if case == 'ph':
            from mpisppy.utils import checkpointing
            assert checkpointing._checkpoint_state_equal(
                checkpointing._snapshot_rebuild_model(m),
                checkpointing._snapshot_rebuild_model(expected),
            ), 'PH parameter, variable, solve, or proximal state changed'
            for key, manager in m._mpisppy_data.xsqvar_prox_approx.items():
                assert manager.xvar is m.find_component(manager.xvar.name)
                assert manager.xvarsqrd is m._mpisppy_model.xsqvar[key]
                assert manager.cuts is m._mpisppy_model.xsqvar_cuts
                assert manager.W is m._mpisppy_model.W[key]
                assert manager.rho is m._mpisppy_model.rho[key]
                assert manager.xbar is m._mpisppy_model.xbars[key]
        return
    assert m.name == 'probe', 'model name lost'
    if case == 'empty':
        return
    assert pyo.value(m.x) == 2
    assert m.x.parent_block() is m
    assert m.x[None] is m.x
    assert m.x.domain is pyo.Reals, 'global Reals identity lost'
    if case == 'indexed':
        assert m.y[0].parent_component() is m.y
        assert m.y.parent_block() is m
        assert m.alias[0] is m.x and m.alias[1] is m.x
        assert m.cmap[m.x] == 'x' and m.x in m.cset
        m.x.set_value(7)
        assert pyo.value(m.obj) == 13
        assert pyo.value(m.c.body) == 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['write', 'read', 'roundtrip'])
    ap.add_argument('mode', choices=['native', 'registered', 'stateful', 'cycle'])
    ap.add_argument('case', choices=['empty', 'scalar', 'indexed', 'farmer', 'ph'])
    ap.add_argument('path')
    a = ap.parse_args()
    m = make_model(a.case)
    result = dict(action=a.action, mode=a.mode, case=a.case)
    try:
        f, classes = codec(a.mode, m)
        result['classes'] = classes
        if a.action in ('write', 'roundtrip'):
            stream = io.BytesIO()
            f.dump(m, stream)
            data = stream.getvalue()
            result['bytes'] = len(data)
            if a.action == 'write':
                with open(a.path, 'wb') as out:
                    out.write(data)
        if a.action in ('read', 'roundtrip'):
            if a.action == 'read':
                with open(a.path, 'rb') as src:
                    data = src.read()
            f, _ = codec(a.mode, m)
            restored = f.deserialize(data)
            validate(restored, a.case, m)
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = type(exc).__name__ + ': ' + str(exc)
    print(json.dumps(result))
    return 0 if result['status'] == 'ok' else 1

if __name__ == '__main__':
    sys.exit(main())
