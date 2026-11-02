"""Versioned, self-contained pyfory codec for full scenario-model reloads.

The file's type table configures a fresh reader.  Dictionary tokens supplement
pyfory references so a view can be attached to its real owner before either
object's contents have been read.
"""

import builtins
import dis
import functools
import gc
import importlib
import json
import logging
import marshal
import struct
import sys
import types
from collections import OrderedDict, defaultdict
from enum import Enum

import pyfory
import pyomo
import pyomo.environ as pyo
from pyomo.common.collections import Bunch
from pyomo.core.base.PyomoModel import PyomoConfig
from pyomo.core.base.global_set import GlobalSetBase, _get_global_set
from pyomo.core.expr.relational_expr import TrivialRelationalExpression


MAGIC = b"MSPFORYR2"
SCHEMA_VERSION = 2
ADAPTER_VERSION = 1
SUPPORTED_PYTHON_VERSION = (3, 13, 2)
SUPPORTED_PYOMO_VERSION = "6.10.1"
SUPPORTED_PYFORY_VERSION = "1.7.7"
MAX_HEADER_BYTES = 1 << 20
MAX_TYPES = 10000
MAX_OWNERS = 1000000
MAX_CELLS = 1000000
MAX_ENTRIES = 10000000
MAX_PAYLOAD_BYTES = 1 << 30
FORY_OPTIONS = dict(xlang=False, ref=True, strict=False, max_depth=10000,
                    max_graph_memory_bytes=MAX_PAYLOAD_BYTES)
DICT_VIEW_KINDS = {type(getattr({}, kind)()): kind
                   for kind in ("keys", "values", "items")}
ORDERED_VIEW_KINDS = {type(getattr(OrderedDict(), kind)()): kind
                      for kind in ("keys", "values", "items")}
VIEW_KINDS = {**DICT_VIEW_KINDS, **ORDERED_VIEW_KINDS}
SUPPORTED_MAPPINGS = (dict, Bunch, PyomoConfig, OrderedDict, defaultdict)
RESERVED_TYPE_IDS = {"function": 9040, "cell": 9050,
                     "keys": 9060, "values": 9061, "items": 9062,
                     "odict_keys": 9063, "odict_values": 9064,
                     "odict_items": 9065}
FIRST_DYNAMIC_TYPE_ID = 9100


def _mapping_setitem(owner, key, value):
    if type(owner) is OrderedDict:
        OrderedDict.__setitem__(owner, key, value)
    else:
        dict.__setitem__(owner, key, value)


def backing_mapping(view):
    """Return the sole CPython referent of an exact built-in dictionary view."""
    if type(view) not in VIEW_KINDS:
        raise TypeError(f"Unsupported dictionary view type {type(view)!r}")
    referents = gc.get_referents(view)
    if len(referents) != 1 or type(referents[0]) not in SUPPORTED_MAPPINGS:
        raise TypeError("Cannot identify a supported backing dictionary for "
                        f"{type(view).__name__}")
    return referents[0]


def require_supported_runtime():
    """Fail at setup when the documented view-owner mechanism is unavailable."""
    if sys.implementation.name != "cpython" or \
            tuple(sys.version_info[:3]) != SUPPORTED_PYTHON_VERSION or \
            pyomo.version.version != SUPPORTED_PYOMO_VERSION or \
            pyfory.__version__ != SUPPORTED_PYFORY_VERSION:
        raise RuntimeError(
            "pyfory-reload supports CPython 3.13.2, Pyomo 6.10.1, "
            "and pyfory 1.7.7")
    for cls in SUPPORTED_MAPPINGS:
        for filled in (False, True):
            owner = cls()
            if filled:
                _mapping_setitem(owner, "before", 1)
            for kind in ("keys", "values", "items"):
                view = getattr(owner, kind)()
                if backing_mapping(view) is not owner:
                    raise RuntimeError("Dictionary view owner extraction failed")
                _mapping_setitem(owner, "after", 2)
                if kind == "keys" and "after" not in view or \
                        kind == "values" and 2 not in view or \
                        kind == "items" and ("after", 2) not in view:
                    raise RuntimeError("Dictionary views are not live")


def _importable(obj):
    try:
        target = importlib.import_module(obj.__module__)
        for part in obj.__qualname__.split("."):
            target = getattr(target, part)
        return target is obj
    except (AttributeError, ImportError, TypeError):
        return False


def _resolve(module, name):
    obj = importlib.import_module(module)
    for part in name.split("."):
        if part == "<locals>" or not part:
            raise ValueError(f"Invalid importable name {module}:{name}")
        obj = getattr(obj, part)
    return obj


def _global_names(code):
    names = set()
    for instruction in dis.get_instructions(code):
        if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            if instruction.argval in ("globals", "locals", "eval", "exec"):
                raise TypeError("pyfory-reload cannot encode a callback with "
                                "dynamic global lookup")
            names.add(instruction.argval)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names.update(_global_names(constant))
    return names


class _Tables:
    def __init__(self):
        self.kept = []
        self.tokens = {}
        self.shells = {}
        self.status = {}
        self.cells_kept = []
        self.cell_tokens = {}
        self.cells = {}
        self.cell_status = {}
        self.namespaces = {}
        self.states = {}
        self.mapping_codes = {dict: 0}
        self.code_mappings = {0: dict}

    def token(self, obj):
        key = id(obj)
        if key not in self.tokens:
            if len(self.kept) >= MAX_OWNERS:
                raise ValueError("Too many dictionary owners in checkpoint")
            self.tokens[key] = len(self.kept)
            self.kept.append(obj)
        return self.tokens[key]

    def cell_token(self, cell):
        key = id(cell)
        if key not in self.cell_tokens:
            if len(self.cells_kept) >= MAX_CELLS:
                raise ValueError("Too many closure cells in checkpoint")
            self.cell_tokens[key] = len(self.cells_kept)
            self.cells_kept.append(cell)
        return self.cell_tokens[key]

    def shell(self, token, code):
        if token >= MAX_OWNERS or code not in self.code_mappings:
            raise ValueError("Invalid dictionary owner token or type")
        cls = self.code_mappings[code]
        if token not in self.shells:
            if token != len(self.shells):
                raise ValueError("Nonconsecutive dictionary owner token")
            if cls is OrderedDict:
                self.shells[token] = OrderedDict()
            elif cls is defaultdict:
                self.shells[token] = defaultdict()
            else:
                self.shells[token] = dict.__new__(cls)
            self.status[token] = "declared"
        obj = self.shells[token]
        if type(obj) is not cls:
            raise ValueError("Conflicting dictionary owner types")
        return obj

    def cell_shell(self, token):
        if token >= MAX_CELLS:
            raise ValueError("Invalid closure cell token")
        if token not in self.cells:
            if token != len(self.cells):
                raise ValueError("Nonconsecutive closure cell token")
            self.cells[token] = types.CellType()
            self.cell_status[token] = "declared"
        return self.cells[token]

    def state(self, obj):
        key = id(obj)
        if key not in self.states:
            self.states[key] = (obj, obj.__getstate__())
        return self.states[key][1]

    def namespace(self, func):
        source = func.__globals__
        key = id(source)
        if key not in self.namespaces:
            self.namespaces[key] = (source, {"__name__": func.__module__})
        namespace = self.namespaces[key][1]
        for name in _global_names(func.__code__):
            if name in source:
                namespace[name] = source[name]
        return namespace

    def finish(self):
        missing = [token for token, status in self.status.items()
                   if status != "complete"]
        if missing:
            raise ValueError(f"Undefined dictionary owner token {missing[0]}")
        missing = [token for token, status in self.cell_status.items()
                   if status != "complete"]
        if missing:
            raise ValueError(f"Undefined closure cell token {missing[0]}")


class _MappingSerializer(pyfory.Serializer):
    def __init__(self, resolver, cls, tables):
        super().__init__(resolver, cls)
        self.tables = tables

    def write(self, context, obj):
        tables = self.tables
        context.write_var_uint32(tables.token(obj))
        context.write_var_uint32(tables.mapping_codes[type(obj)])
        if len(obj) > MAX_ENTRIES:
            raise ValueError("Too many dictionary entries in checkpoint")
        context.write_var_uint32(len(obj))
        for key, value in obj.items():
            context.write_ref(key)
            context.write_ref(value)
        if type(obj) is defaultdict:
            context.write_ref(obj.default_factory)
        if isinstance(obj, Bunch) or type(obj) is OrderedDict:
            context.write_ref(obj.__dict__)

    def read(self, context):
        tables = self.tables
        token = context.read_var_uint32()
        code = context.read_var_uint32()
        obj = tables.shell(token, code)
        if type(obj) is not self.type_ or tables.status[token] != "declared":
            raise ValueError("Duplicate or conflicting dictionary definition")
        tables.status[token] = "populating"
        context.reference(obj)
        count = context.read_var_uint32()
        if count > MAX_ENTRIES:
            raise ValueError("Too many dictionary entries in checkpoint")
        for _ in range(count):
            key = context.read_ref()
            _mapping_setitem(obj, key, context.read_ref())
        if type(obj) is defaultdict:
            obj.default_factory = context.read_ref()
        if isinstance(obj, Bunch) or type(obj) is OrderedDict:
            attributes = context.read_ref()
            if type(attributes) is not dict:
                raise ValueError("Invalid mapping instance attributes")
            # Bind the decoded dictionary itself so saved aliases and views
            # of the mapping's __dict__ point to its live instance state.
            object.__setattr__(obj, "__dict__", attributes)
        tables.status[token] = "complete"
        return obj


class _ViewSerializer(pyfory.Serializer):
    def __init__(self, resolver, cls, tables):
        super().__init__(resolver, cls)
        self.tables = tables

    def write(self, context, obj):
        owner = backing_mapping(obj)
        cls = type(owner)
        if cls not in self.tables.mapping_codes:
            raise TypeError(f"Unsupported dictionary view owner {cls!r}")
        context.write_var_uint32(self.tables.token(owner))
        context.write_var_uint32(self.tables.mapping_codes[cls])
        context.write_ref(owner)

    def read(self, context):
        token = context.read_var_uint32()
        code = context.read_var_uint32()
        owner = self.tables.shell(token, code)
        view = getattr(owner, VIEW_KINDS[self.type_])()
        if type(view) is not self.type_:
            raise ValueError("Dictionary view type does not match its owner")
        context.reference(view)
        if context.read_ref() is not owner:
            raise ValueError("Dictionary view decoded with a different owner")
        return view


class _CellSerializer(pyfory.Serializer):
    def __init__(self, resolver, tables):
        super().__init__(resolver, types.CellType)
        self.tables = tables

    def write(self, context, obj):
        context.write_var_uint32(self.tables.cell_token(obj))
        try:
            value = obj.cell_contents
        except ValueError:
            context.write_bool(False)
        else:
            context.write_bool(True)
            context.write_ref(value)

    def read(self, context):
        token = context.read_var_uint32()
        cell = self.tables.cell_shell(token)
        if self.tables.cell_status[token] != "declared":
            raise ValueError("Duplicate closure cell definition")
        self.tables.cell_status[token] = "populating"
        context.reference(cell)
        if context.read_bool():
            cell.cell_contents = context.read_ref()
        self.tables.cell_status[token] = "complete"
        return cell


class _FunctionSerializer(pyfory.Serializer):
    def __init__(self, resolver, tables):
        super().__init__(resolver, types.FunctionType)
        self.tables = tables

    def write(self, context, obj):
        by_reference = _importable(obj)
        context.write_bool(by_reference)
        context.write_string(obj.__module__)
        context.write_string(obj.__qualname__)
        if by_reference:
            return
        context.write_bytes_and_size(marshal.dumps(obj.__code__))
        context.write_string(obj.__name__)
        namespace = self.tables.namespace(obj)
        context.write_var_uint32(self.tables.token(namespace))
        cells = obj.__closure__ or ()
        context.write_var_uint32(len(cells))
        for cell in cells:
            context.write_var_uint32(self.tables.cell_token(cell))
        context.write_ref(namespace)
        for cell in cells:
            context.write_ref(cell)
        context.write_ref(obj.__defaults__)
        context.write_ref(obj.__kwdefaults__)
        context.write_ref(obj.__dict__)
        context.write_ref(obj.__annotations__)
        context.write_ref(obj.__doc__)

    def read(self, context):
        by_reference = context.read_bool()
        module = context.read_string()
        qualname = context.read_string()
        if by_reference:
            obj = _resolve(module, qualname)
            if not isinstance(obj, types.FunctionType):
                raise ValueError("Function reference does not resolve to a function")
            context.reference(obj)
            return obj
        code = marshal.loads(context.read_bytes_and_size())
        if not isinstance(code, types.CodeType):
            raise ValueError("Invalid callback code in checkpoint")
        name = context.read_string()
        namespace = self.tables.shell(context.read_var_uint32(), 0)
        namespace["__builtins__"] = builtins.__dict__
        count = context.read_var_uint32()
        if count > MAX_CELLS or count != len(code.co_freevars):
            raise ValueError("Invalid callback closure size")
        cells = tuple(self.tables.cell_shell(context.read_var_uint32())
                      for _ in range(count))
        obj = types.FunctionType(code, namespace, name, None,
                                 cells or None)
        context.reference(obj)
        obj.__module__ = module
        obj.__qualname__ = qualname
        if context.read_ref() is not namespace:
            raise ValueError("Callback namespace identity mismatch")
        for cell in cells:
            if context.read_ref() is not cell:
                raise ValueError("Callback closure identity mismatch")
        obj.__defaults__ = context.read_ref()
        obj.__kwdefaults__ = context.read_ref()
        obj.__dict__.update(context.read_ref())
        obj.__annotations__ = context.read_ref()
        obj.__doc__ = context.read_ref()
        return obj


class _StateSerializer(pyfory.Serializer):
    def __init__(self, resolver, cls, tables):
        super().__init__(resolver, cls)
        self.tables = tables

    def write(self, context, obj):
        context.write_ref(self.tables.state(obj))

    def read(self, context):
        try:
            obj = self.type_.__new__(self.type_)
        except Exception as exc:
            raise TypeError(f"Cannot allocate checkpoint type {self.type_!r}") from exc
        context.reference(obj)
        state = context.read_ref()
        if hasattr(obj, "__setstate__"):
            obj.__setstate__(state)
        elif isinstance(state, dict) and hasattr(obj, "__dict__"):
            obj.__dict__.update(state)
        else:
            raise TypeError(f"Cannot restore state of {self.type_!r}")
        return obj


class _GlobalSetSerializer(pyfory.Serializer):
    def write(self, context, obj):
        context.write_string(obj.name)

    def read(self, context):
        return _get_global_set(context.read_string())


class _TrivialSerializer(pyfory.Serializer):
    def write(self, context, obj):
        if obj is not pyo.Constraint.Feasible and obj is not pyo.Constraint.Infeasible:
            raise TypeError("Unknown Pyomo trivial relational expression")
        context.write_bool(obj is pyo.Constraint.Feasible)

    def read(self, context):
        return (pyo.Constraint.Feasible if context.read_bool()
                else pyo.Constraint.Infeasible)


class _LoggerSerializer(pyfory.Serializer):
    def write(self, context, obj):
        context.write_string(obj.name)

    def read(self, context):
        return logging.getLogger(context.read_string())


class _ImportableCallableSerializer(pyfory.Serializer):
    def write(self, context, obj):
        if not _importable(obj):
            raise TypeError(f"Cached callable {obj!r} is not importable")
        context.write_string(obj.__module__)
        context.write_string(obj.__qualname__)

    def read(self, context):
        obj = _resolve(context.read_string(), context.read_string())
        if type(obj) is not self.type_:
            raise ValueError("Cached callable reference has a different type")
        context.reference(obj)
        return obj


def _descriptor(cls, example):
    if isinstance(example, GlobalSetBase):
        return {"kind": "global_set", "name": example.name}
    if cls is TrivialRelationalExpression:
        kind = "trivial"
    elif cls in SUPPORTED_MAPPINGS:
        kind = "mapping"
    elif issubclass(cls, Enum):
        kind = "enum"
    elif issubclass(cls, logging.Logger):
        kind = "logger"
    elif cls is functools._lru_cache_wrapper:
        kind = "importable_callable"
    else:
        kind = "state"
    if "<locals>" in cls.__qualname__:
        raise TypeError(f"Checkpoint type {cls!r} is not importable")
    return {"kind": kind, "name": f"{cls.__module__}:{cls.__qualname__}"}


def _discover(root, tables):
    seen = {}
    classes = {}
    view_owners = {}
    instance_dicts = {}

    def visit(obj, path):
        key = id(obj)
        if key in seen:
            return
        seen[key] = obj
        cls = type(obj)
        try:
            if cls in VIEW_KINDS:
                owner = backing_mapping(obj)
                view_owners[id(owner)] = path
                visit(owner, path + ".owner")
            elif cls is types.FunctionType:
                if not _importable(obj):
                    instance_dicts[id(obj.__dict__)] = path
                    namespace = tables.namespace(obj)
                    visit(namespace, path + ".globals")
                    # A later function sharing this source namespace can add
                    # names after the namespace itself was first visited.
                    for name in _global_names(obj.__code__):
                        if name in obj.__globals__:
                            visit(obj.__globals__[name],
                                  path + ".global." + name)
                    for cell in obj.__closure__ or ():
                        try:
                            visit(cell.cell_contents, path + ".closure")
                        except ValueError:
                            pass
                    visit(obj.__defaults__, path + ".defaults")
                    visit(obj.__kwdefaults__, path + ".kwdefaults")
                    visit(obj.__dict__, path + ".attributes")
                    visit(obj.__annotations__, path + ".annotations")
                    visit(obj.__doc__, path + ".doc")
            elif cls is types.MethodType:
                visit(obj.__self__, path + ".self")
                visit(obj.__func__, path + ".function")
            elif isinstance(obj, (type, types.ModuleType)):
                return
            elif isinstance(obj, dict):
                if cls is not dict:
                    if cls not in SUPPORTED_MAPPINGS:
                        raise TypeError(f"Unsupported mapping subclass {cls!r}")
                    classes.setdefault(cls, obj)
                for entry_key, value in obj.items():
                    visit(entry_key, path + ".key")
                    visit(value, path + f"[{str(entry_key)[:64]}]")
                if cls is defaultdict:
                    visit(obj.default_factory, path + ".default_factory")
                if isinstance(obj, Bunch) or cls is OrderedDict:
                    visit(obj.__dict__, path + ".__dict__")
            elif isinstance(obj, (tuple, list, set, frozenset)):
                for index, value in enumerate(obj):
                    visit(value, path + f"[{index}]")
            elif isinstance(obj, (Enum, logging.Logger,
                                  functools._lru_cache_wrapper,
                                  TrivialRelationalExpression)):
                if cls is functools._lru_cache_wrapper and not _importable(obj):
                    raise TypeError("Local cached callable has no supported "
                                    "checkpoint adapter")
                if isinstance(obj, logging.Logger) and cls is not logging.Logger:
                    raise TypeError("Unsupported logger subclass")
                if isinstance(obj, TrivialRelationalExpression) and \
                        obj is not pyo.Constraint.Feasible and \
                        obj is not pyo.Constraint.Infeasible:
                    raise TypeError("Unknown Pyomo trivial expression")
                classes.setdefault(cls, obj)
            elif cls.__module__ not in ("builtins", "array") and not \
                    cls.__module__.startswith("numpy"):
                classes.setdefault(cls, obj)
                if hasattr(obj, "__dict__"):
                    instance_dicts[id(obj.__dict__)] = path
                visit(tables.state(obj), path + ".state")
        except Exception as exc:
            if str(exc).startswith(getattr(root, "name", "model") + "."):
                raise
            raise type(exc)(f"{path}: {exc}") from exc

    visit(root, getattr(root, "name", "model"))
    for key, path in view_owners.items():
        if key in instance_dicts:
            raise TypeError(f"{path}: a view of {instance_dicts[key]}'s own "
                            "instance dictionary is not supported")
    entries = [(_descriptor(cls, example), cls)
               for cls, example in classes.items()]
    entries.sort(key=lambda item: json.dumps(item[0], sort_keys=True))
    if len(entries) > MAX_TYPES:
        raise ValueError("Too many checkpoint types")
    return entries


def _resolve_entries(descriptors):
    if not isinstance(descriptors, list) or len(descriptors) > MAX_TYPES:
        raise ValueError("Invalid pyfory checkpoint type table")
    seen_names = set()
    seen_classes = set()
    entries = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or set(descriptor) != {"kind", "name"}:
            raise ValueError("Invalid pyfory checkpoint type descriptor")
        kind, name = descriptor["kind"], descriptor["name"]
        if not isinstance(name, str) or not isinstance(kind, str) or \
                (kind, name) in seen_names:
            raise ValueError("Duplicate or invalid checkpoint type descriptor")
        seen_names.add((kind, name))
        if kind == "global_set":
            example = _get_global_set(name)
            cls = type(example)
        else:
            if ":" not in name:
                raise ValueError("Invalid checkpoint class name")
            cls = _resolve(*name.split(":", 1))
            if not isinstance(cls, type):
                raise ValueError("Checkpoint type is not a class")
            example = None
        if cls in seen_classes:
            raise ValueError("Duplicate checkpoint class registration")
        seen_classes.add(cls)
        if kind == "mapping" and cls not in SUPPORTED_MAPPINGS:
            raise ValueError("Unsupported checkpoint mapping adapter")
        if kind != "global_set" and _descriptor(cls, example)["kind"] != kind:
            raise ValueError("Checkpoint adapter kind mismatch")
        entries.append((descriptor, cls))
    if entries != sorted(entries, key=lambda item: json.dumps(item[0], sort_keys=True)):
        raise ValueError("Unsorted checkpoint type table")
    return entries


def _new_codec(entries, tables):
    codec = pyfory.Fory(**FORY_OPTIONS)
    codec.register_serializer(dict, _MappingSerializer(codec.type_resolver,
                                                       dict, tables))
    codec.register_type(types.FunctionType,
                        type_id=RESERVED_TYPE_IDS["function"],
                        serializer=_FunctionSerializer(codec.type_resolver,
                                                       tables))
    codec.register_type(types.CellType, type_id=RESERVED_TYPE_IDS["cell"],
                        serializer=_CellSerializer(codec.type_resolver, tables))
    for cls, kind in VIEW_KINDS.items():
        type_name = kind if cls in DICT_VIEW_KINDS else "odict_" + kind
        codec.register_type(cls, type_id=RESERVED_TYPE_IDS[type_name],
                            serializer=_ViewSerializer(codec.type_resolver,
                                                       cls, tables))
    for offset, (descriptor, cls) in enumerate(entries):
        kind = descriptor["kind"]
        if kind == "mapping":
            code = offset + 1
            tables.mapping_codes[cls] = code
            tables.code_mappings[code] = cls
            serializer = _MappingSerializer(codec.type_resolver, cls, tables)
        elif kind == "global_set":
            serializer = _GlobalSetSerializer(codec.type_resolver, cls)
        elif kind == "trivial":
            serializer = _TrivialSerializer(codec.type_resolver, cls)
        elif kind == "logger":
            serializer = _LoggerSerializer(codec.type_resolver, cls)
        elif kind == "importable_callable":
            serializer = _ImportableCallableSerializer(codec.type_resolver, cls)
        elif kind == "enum":
            serializer = None
        else:
            serializer = _StateSerializer(codec.type_resolver, cls, tables)
        codec.register_type(cls, type_id=FIRST_DYNAMIC_TYPE_ID + offset,
                            serializer=serializer)
    return codec


def _header(model, entries, scenario_name, write_id):
    return {
        "backend": "pyfory-reload",
        "schema_version": SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "python_implementation": sys.implementation.name,
        "python_version": list(sys.version_info[:3]),
        "pyfory_version": pyfory.__version__,
        "pyomo_version": pyomo.version.version,
        "scenario_name": scenario_name or getattr(model, "name", None),
        "model_name": getattr(model, "name", None),
        "write_id": write_id,
        "types": [descriptor for descriptor, _ in entries],
    }


def dump(model, stream, *, scenario_name=None, write_id=None):
    """Write one complete scenario with a reader-independent type registry."""
    require_supported_runtime()
    tables = _Tables()
    entries = _discover(model, tables)
    header = json.dumps(_header(model, entries, scenario_name, write_id),
                        separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(header) > MAX_HEADER_BYTES:
        raise ValueError("Pyfory reload checkpoint header is too large")
    # Buffer the payload so a failed custom serializer cannot leave a partial
    # stand-alone file. Atomic checkpoint publication also protects generations.
    payload = _new_codec(entries, tables).serialize(model)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("Pyfory reload checkpoint payload is too large")
    stream.write(MAGIC)
    stream.write(struct.pack(">I", len(header)))
    stream.write(header)
    stream.write(payload)


def load(stream, *, scenario_name=None, write_id=None):
    """Read a full model using only registrations named in its file header."""
    require_supported_runtime()
    if stream.read(len(MAGIC)) != MAGIC:
        raise ValueError("Invalid pyfory model checkpoint header")
    size_blob = stream.read(4)
    if len(size_blob) != 4:
        raise ValueError("Truncated pyfory reload checkpoint header")
    size = struct.unpack(">I", size_blob)[0]
    if size > MAX_HEADER_BYTES:
        raise ValueError("Pyfory reload checkpoint header is too large")
    raw = stream.read(size)
    if len(raw) != size:
        raise ValueError("Truncated pyfory reload checkpoint header")
    header = json.loads(raw)
    if not isinstance(header, dict) or header.get("backend") != "pyfory-reload" or \
            header.get("schema_version") != SCHEMA_VERSION or \
            header.get("adapter_version") != ADAPTER_VERSION or \
            header.get("python_implementation") != sys.implementation.name or \
            header.get("python_version") != list(sys.version_info[:3]) or \
            header.get("pyfory_version") != pyfory.__version__ or \
            header.get("pyomo_version") != pyomo.version.version:
        raise ValueError("Unsupported pyfory reload checkpoint runtime or schema")
    if scenario_name is not None and header.get("scenario_name") != scenario_name:
        raise ValueError("Pyfory reload scenario name mismatch")
    if write_id is not None and header.get("write_id") != write_id:
        raise ValueError("Pyfory reload write identity mismatch")
    entries = _resolve_entries(header.get("types"))
    payload = stream.read(MAX_PAYLOAD_BYTES + 1)
    if not payload or len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("Missing or oversized pyfory reload payload")
    tables = _Tables()
    model = _new_codec(entries, tables).deserialize(payload)
    tables.finish()
    if getattr(model, "name", None) != header.get("model_name"):
        raise ValueError("Pyfory reload model name mismatch")
    return model
