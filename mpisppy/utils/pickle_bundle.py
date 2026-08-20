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
# Bound on how many attributes of a bound method's owner get reported.
_DIAGNOSTIC_MAX_SELF_ATTRS = 5


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

    Walks a bounded neighborhood of `root` -- following both __dict__ and
    __slots__ attributes, whatever they hold -- and stops at a fixed depth and
    visit count. It does descend into whatever it finds, model components
    included; the caps, not the object kind, are what bound it. The visit cap
    can truncate the walk on a very large subproblem, which is harmless here
    because find_undillable_closures re-roots at every component, so a rule
    missed from one starting point is reached from its own.
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
            # Bound methods are yielded whether or not they close over
            # anything: for a class-based model builder the culprit is
            # typically an attribute of __self__, not a closure cell.
            if inspect.ismethod(obj) or getattr(obj, "__closure__", None):
                yield obj
            continue
        if isinstance(obj, (str, bytes, int, float, bool, type(None))):
            continue
        for _, value in _attributes_of(obj):
            stack.append((value, depth + 1))


def _is_dillable(value, memo):
    """Test whether dill can serialize `value`, memoized by object identity.

    A proper bundle is an EF over many scenarios, each carrying its own copy of
    the rule functions, so the same large closure value is reached again and
    again; without the memo a big dillable dataset gets re-serialized once per
    scenario. The memo maps id -> (value, result) and holds the value, both to
    keep the answer and to keep the object alive so its id cannot be recycled.
    """
    key = id(value)
    if key in memo:
        return memo[key][1]
    try:
        dill.dumps(value)
        result = True
    except Exception:
        result = False
    memo[key] = (value, result)
    return result


def _unserializable_captures(func, memo):
    """Yield (kind, description, value) for what `func` carries that dill rejects.

    `kind` is "closure" or "self", so the caller can give advice that matches
    how the value was actually captured.

    Covers closure cells and, for a bound method, the attributes of the
    instance it is bound to -- a rule written as a method on a model-builder
    class reaches its configuration through ``self``, not through a closure.
    """
    cells = getattr(func, "__closure__", None) or ()
    names = getattr(getattr(func, "__code__", None), "co_freevars", ()) or ()
    for name, cell in zip(names, cells):
        try:
            value = cell.cell_contents
        except ValueError:
            continue  # empty cell
        if not _is_dillable(value, memo):
            yield "closure", name, value

    owner = getattr(func, "__self__", None)
    if owner is None or _is_owner_uninteresting(owner):
        return
    if _is_dillable(owner, memo):
        return
    # Expanding a large owner is both slow and misleading: every attribute
    # that merely *reaches* the real culprit tests as unserializable too, so
    # a model-sized owner buries the answer in red herrings. Report a bounded
    # number of specific attributes, and fall back to naming the instance.
    reported = 0
    for attr, value in _attributes_of(owner):
        if reported >= _DIAGNOSTIC_MAX_SELF_ATTRS:
            break
        if not _is_dillable(value, memo):
            yield "self", f"self.{attr}", value
            reported += 1
    if not reported:
        yield "self", "self", owner


def _is_owner_uninteresting(owner):
    """True when a bound method's owner is not a model-builder object.

    The case this diagnostic wants is a small class that builds models and
    holds configuration. A Pyomo component -- most often the model itself,
    reached through something like ``model._helper = model.clone`` -- is not
    that, and expanding it produces pages of attributes that are unserializable
    only because they transitively reach the same real culprit.
    """
    try:
        from pyomo.core.base.component import ComponentBase
    except ImportError:      # pragma: no cover - very old Pyomo
        return False
    return isinstance(owner, ComponentBase)


def find_undillable_closures(model):
    """Locate values a rule on `model` carries that dill cannot serialize.

    Returns a deduplicated list of
    (rule_name, capture_name, value_type_name, kind), where `kind` is
    "closure" for a value in a rule's closure cell and "self" for one reached
    through the instance a bound-method rule belongs to. Deduplication matters
    because a bundle holds one copy of each rule per scenario.

    Pyomo keeps a reference to each rule function on the component it built, so
    anything a rule captured is reachable from the model and has to be
    serializable too -- which is how an otherwise ordinary model becomes
    unpicklable.

    Known limitation: only what a *rule* carries is inspected. An
    unserializable object stored directly on the model
    (``model._handle = ...``) or inside a plain container is not reported,
    though it fails just the same; the caller says so when nothing is found.

    Not idempotent, by nature: probing a value serializes it, and serializing
    a Pyomo ConfigValue resolves its default and changes its class. Calling
    this repeatedly on one model can therefore report fewer culprits each
    time. It runs on a failure path where the run is ending anyway, and
    resolving a default is semantically a no-op.
    """
    if not dill_available:
        # Without dill every serialization test below would raise, and every
        # closure cell would be misreported as the culprit.
        return []
    found = []
    seen_funcs = set()
    memo = {}
    try:
        components = list(model.component_objects(descend_into=True))
    except Exception:
        components = []
    for comp in [model] + components:
        for func in _closures_reachable_from(comp):
            if id(func) in seen_funcs:
                continue
            seen_funcs.add(id(func))
            for kind, varname, value in _unserializable_captures(func, memo):
                entry = (getattr(func, "__name__", "<unknown>"),
                         varname, type(value).__name__, kind)
                if entry not in found:
                    found.append(entry)
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

    if not dill_available:
        lines.append("")
        lines.append(
            "dill is not installed. It is an optional dependency; install it "
            "with 'pip install mpi-sppy[extras]' (or 'pip install dill').")
        return "\n".join(lines)

    diagnostic_error = None
    try:
        culprits = find_undillable_closures(model)
    except Exception as diag_exc:
        culprits = []
        diagnostic_error = f"{type(diag_exc).__name__}: {diag_exc}"

    if culprits:
        lines.append("")
        lines.append(
            "The cause appears to be a value captured in the closure of a "
            "Pyomo rule. Pyomo keeps the rule function on the component it "
            "built, so whatever the rule closed over must be serializable "
            "too. Found:")
        for rule, varname, typename, kind in culprits:
            verb = "closes over" if kind == "closure" else "reaches"
            lines.append(f"  - rule {rule}() {verb} '{varname}' "
                         f"of type {typename}")
        lines.append("")
        if any(kind == "closure" for *_, kind in culprits):
            lines.append(
                "For a rule defined as a nested function, fix this in the "
                "model: read the values the rule needs into plain local "
                "variables before defining it, so the closure captures those "
                "values rather than the object holding them. For example, "
                "'seed = cfg.initial_seed' outside the rule, using 'seed' "
                "inside it.")
        if any(kind == "self" for *_, kind in culprits):
            lines.append(
                "For a rule defined as a method, the value is reached through "
                "the instance the method is bound to, not through a closure. "
                "Either keep the unserializable attribute off that instance, "
                "or have the method read from plain data copied onto it "
                "during construction.")
        lines.append(
            "(An mpi-sppy Config is a Pyomo ConfigDict. A ConfigDict whose "
            "entries still hold unresolved defaults fails the first time it "
            "is serialized -- a Pyomo lazy-initialization quirk, not "
            "something dill does differently: reading an entry flips its "
            "class, and that happens while the pickler is already reading "
            "its state.)")
    else:
        lines.append("")
        lines.append(
            "No unserializable value was found in a Pyomo rule closure, so "
            "something else reachable from the model cannot be serialized: an "
            "object stored on the model itself, or on one of its components, "
            "or held inside a plain container. The exception above often names "
            "the offending type. Note that a live handle on something outside "
            "the process -- a solver interface, a connection, an open "
            "resource -- is the usual culprit; keep such things off the model "
            "if it has to be serialized.")
        if diagnostic_error is not None:
            lines.append("")
            lines.append(
                f"(The closure diagnostic itself failed with "
                f"{diagnostic_error}, so it could not rule that cause in or "
                f"out. This is a bug in mpi-sppy, not in your model.)")
    return "\n".join(lines)


def _remove_quietly(fname):
    try:
        os.remove(fname)
    except OSError:
        pass


def dill_pickle(model, fname):
    """ serialize model using dill to file name"""
    # global_toc(f"about to pickle to {fname}")
    if not dill_available:
        # Check before opening: otherwise we truncate the target on the way to
        # discovering we cannot write it.
        raise RuntimeError(
            "dill is required to pickle scenarios and bundles but is not "
            "installed. It is an optional dependency; install it with "
            "'pip install mpi-sppy[extras]' (or 'pip install dill')."
        )
    # Opening is deliberately outside the try below: an OSError here (missing
    # directory, unwritable path) is self-explanatory, must not be dressed up
    # as a serialization failure, and must not trigger the cleanup -- that
    # would delete a pre-existing file this call never wrote to.
    f = open(fname, "wb")
    try:
        try:
            dill.dump(model, f)
        finally:
            f.close()
    except OSError:
        # An OSError from here is a write failure (disk full, quota, NFS),
        # not an open failure -- the file exists and holds our own truncated
        # write. Clean it up, then let the error speak for itself.
        _remove_quietly(fname)
        raise
    except Exception as exc:
        _remove_quietly(fname)
        raise RuntimeError(
            describe_dill_failure(model, exc,
                                  what=f"model destined for '{fname}'")
        ) from exc
    # global_toc(f"done with pickle {fname}")


def dill_unpickle(fname):
    """ load a model from fname"""

    # global_toc(f"about to unpickle {fname}")
    # As in dill_pickle: a missing or unreadable file speaks for itself, and
    # saying "this is from a different environment" about a path that does not
    # exist would be actively misleading. Bundle file names are constructed by
    # name (see proper_bundler), so a naming mismatch lands here routinely.
    f = open(fname, "rb")
    try:
        try:
            m = dill.load(f)
        finally:
            f.close()
    except OSError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the dilled model in '{fname}' "
            f"({type(exc).__name__}: {exc}). A dilled model is only readable "
            f"by the same environment that wrote it -- the same Python, "
            f"Pyomo, dill, and model code -- so this usually means the file "
            f"was written by a different environment, or is truncated."
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
        

