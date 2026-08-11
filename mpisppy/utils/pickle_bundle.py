###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Utilities to support pickling and unpickling "proper" bundles
# This file also provides support for pickled scenarios

# NOTE: if/because we require the bundles to consume entire
#       second stage tree nodes, the resulting problem is two stage.
#  This adds complications when working with multi-stage problems.

# BTW: ssn (not in this repo) uses this as of March 2022

import inspect
import os
from pyomo.common.dependencies import attempt_import
dill, dill_available = attempt_import("dill")

# Bound the search in _closures_reachable_from so a large model cannot turn a
# failed pickle into a long walk. Pyomo nests a rule a few levels down (a
# Constraint keeps IndexedCallInitializer._fcn; a Var's bounds rule sits under
# BoundInitializer._initializer._fcn), so a shallow depth suffices.
_DIAGNOSTIC_MAX_DEPTH = 4
_DIAGNOSTIC_MAX_VISITS = 20000


def _attributes_of(obj):
    """Yield (name, value) over both __dict__ and __slots__ attributes.

    Pyomo's Initializer classes use __slots__, so vars() alone misses the rule
    functions this diagnostic is looking for.
    """
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        yield from list(d.items())
    for cls in type(obj).__mro__:
        for name in getattr(cls, "__slots__", ()) or ():
            try:
                yield name, getattr(obj, name)
            except AttributeError:
                continue


def _closures_reachable_from(root):
    """Yield functions that carry a closure, reachable from root.

    Walks a bounded neighborhood rather than the whole object graph: we only
    descend into plain helper objects (Pyomo initializers and the like), never
    into other model components, and stop at a fixed depth and visit count.
    """
    seen = set()
    visits = 0
    stack = [(root, 0)]
    while stack:
        obj, depth = stack.pop()
        if depth > _DIAGNOSTIC_MAX_DEPTH:
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        visits += 1
        if visits > _DIAGNOSTIC_MAX_VISITS:
            return
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            if getattr(obj, "__closure__", None):
                yield obj
            continue
        if isinstance(obj, (str, bytes, int, float, bool, type(None))):
            continue
        for _, value in _attributes_of(obj):
            stack.append((value, depth + 1))


def _unserializable_closure_cells(func):
    """Yield (freevar_name, value) for closure cells dill cannot serialize."""
    cells = getattr(func, "__closure__", None) or ()
    names = getattr(getattr(func, "__code__", None), "co_freevars", ()) or ()
    for name, cell in zip(names, cells):
        try:
            value = cell.cell_contents
        except ValueError:
            continue  # empty cell
        try:
            dill.dumps(value)
        except Exception:
            yield name, value


def find_undillable_closures(model):
    """Locate closure variables on `model` that dill cannot serialize.

    Returns a list of (rule_name, freevar_name, value_type_name). Pyomo keeps a
    reference to each rule function on the component it built, so anything a
    rule closed over is reachable from the model and has to be serializable
    too -- which is how an otherwise ordinary model becomes unpicklable.
    """
    found = []
    seen_rules = set()
    try:
        components = list(model.component_objects(descend_into=True))
    except Exception:
        components = []
    for comp in [model] + components:
        for func in _closures_reachable_from(comp):
            key = (getattr(func, "__qualname__", repr(func)), id(func))
            if key in seen_rules:
                continue
            seen_rules.add(key)
            for varname, value in _unserializable_closure_cells(func):
                found.append((getattr(func, "__name__", "<unknown>"),
                              varname, type(value).__name__))
    return found


def describe_dill_failure(model, exc, what="model"):
    """Explain, as concretely as possible, why `model` could not be dilled.

    dill's own error is typically opaque (e.g. "args[0] from __newobj__ args
    has the wrong class"), naming neither the component nor the offending
    object. When the cause is something captured in a rule's closure -- the
    common case -- this pins it down by name.
    """
    lines = [f"Could not serialize the {what} with dill: "
             f"{type(exc).__name__}: {exc}"]
    try:
        culprits = find_undillable_closures(model)
    except Exception:
        culprits = []

    if culprits:
        lines.append("")
        lines.append(
            "The cause appears to be a value captured in the closure of a "
            "Pyomo rule. Pyomo keeps the rule function on the component it "
            "built, so whatever the rule closed over must be serializable "
            "too. Found:")
        for rule, varname, typename in culprits:
            lines.append(f"  - rule {rule}() closes over '{varname}' "
                         f"of type {typename}")
        lines.append("")
        lines.append(
            "Fix this in the model: read the values the rule needs into plain "
            "local variables before defining it, so the closure captures those "
            "values rather than the object holding them. For example, "
            "'seed = cfg.initial_seed' outside the rule, using 'seed' inside "
            "it. (An mpi-sppy Config is a Pyomo ConfigDict, which dill cannot "
            "serialize even though the standard pickle module can.)")
    else:
        lines.append("")
        lines.append(
            "No unserializable value was found in a Pyomo rule closure, so "
            "something else reachable from the model cannot be serialized: an "
            "object stored on the model itself, or on one of its components. "
            "The exception above often names the offending type; dill handles "
            "more than the standard pickle module does, so what fails here is "
            "usually a live handle on something outside the process rather "
            "than ordinary data. Note that mpi-sppy strips the solver plugin "
            "it attaches before serializing, so an attached solver is only an "
            "issue if the model holds one of its own.")
    return "\n".join(lines)


def dill_pickle(model, fname):
    """ serialize model using dill to file name"""
    # global_toc(f"about to pickle to {fname}")
    try:
        with open(fname, "wb") as f:
            dill.dump(model, f)
    except Exception as exc:
        # Do not leave a truncated file that a later run would try to load.
        try:
            os.remove(fname)
        except OSError:
            pass
        raise RuntimeError(
            describe_dill_failure(model, exc,
                                  what=f"model destined for '{fname}'")
        ) from exc
    # global_toc(f"done with pickle {fname}")


def dill_unpickle(fname):
    """ load a model from fname"""

    # global_toc(f"about to unpickle {fname}")
    try:
        with open(fname, "rb") as f:
            m = dill.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the dilled model in '{fname}' "
            f"({type(exc).__name__}: {exc}). A dilled model is only readable "
            f"by the same environment that wrote it -- the same Python, "
            f"Pyomo, dill, and model code -- so this usually means the file "
            f"is from a different environment, or is truncated."
        ) from exc
    # global_toc(f"done with unpickle {fname}")
    return m


def check_args(cfg):
    """ Make sure the pickle bundle args make sense; this assumes the config
    has all the appropriate fields."""
    assert cfg.get("pickle_bundles_dir") is None or cfg.get("unpickle_bundles_dir") is None
    assert cfg.get("pickle_scenarios_dir") is None or cfg.get("unpickle_scenarios_dir") is None
    if cfg.get("unpickle_scenarios_dir") is not None and have_proper_bundles(cfg):
        raise RuntimeError("Unpickled scenarios in proper bundles are not supported")
    if cfg.get("unpickle_bundles_dir") is not None and not os.path.isdir(cfg.unpickle_bundles_dir):
        raise RuntimeError(f"Directory to load pickled bundle files from not found: {cfg.unpickle_bundles_dir}")
    if cfg.get("unpickle_scenarios_dir") is not None and not os.path.isdir(cfg.unpickle_scenarios_dir):
        raise RuntimeError(f"Directory to load pickled scenarios files from not found: {cfg.unpickle_scenarios_dir}")
    

def have_proper_bundles(cfg):
    """ boolean to indicate we have pickled bundles"""
    return cfg.get("scenarios_per_bundle") is not None\
        and cfg.scenarios_per_bundle > 0
        

