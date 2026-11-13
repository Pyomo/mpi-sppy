###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Read/write primitives for checkpointing a run so it can be resumed later.

See ``doc/designs/checkpointing_design.md``. The division of labor is:

- This module owns the *file format*: what a checkpoint generation looks like
  on disk, how it is published atomically, and the fingerprints that decide
  whether a checkpoint may be resumed into the current run.
- ``mpisppy/extensions/checkpointer.py`` owns *when* a checkpoint is written.
- ``PHBase.Iter0`` owns the resume branch itself, because restoring has to
  happen in the middle of startup (the reloaded models must be in place before
  solvers are created).

The ``dill-reload`` and ``pyfory-reload`` backends save each mid-run scenario
model, which brings back
the dual weights, rho, nonant values and fixedness, the recourse values that
serve as a MIP warm start, and the proximal-approximation cuts, all mutually
consistent. Everything that does *not* live on a scenario model -- the global
iteration counter, bounds, and the initially-fixed-nonant baseline -- is written
alongside as a small pickle of plain data.
"""

import hashlib
import io
import json
import math
import os
import pickle
import re
import shutil
import struct
import time
import uuid
import numbers
from array import array
from collections.abc import KeysView
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import numpy as np
from pyomo.common.dependencies import attempt_import
from pyomo.core.base.set import RangeSet, Set, UnknownSetDimen
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.core.expr.numeric_expr import (
    ExternalFunctionExpression, UnaryFunctionExpression)

import mpisppy.MPI as MPI
import mpisppy.utils.pickle_bundle as pickle_bundle

dill, dill_available = attempt_import("dill")
pyfory, pyfory_available = attempt_import("pyfory")

# Bump when the on-disk layout changes in a way older readers cannot handle.
#: Bumped when the on-disk shape changes. Version 2 replaced the spoke
#: incumbent file's per-scenario ``inner_bound`` -- read live at write time,
#: so it could describe a later solve than the values beside it -- with the
#: objective cached when the incumbent itself was.
FORMAT_VERSION = 2

DILL_RELOAD_BACKEND = "dill-reload"
PYFORY_REBUILD_BACKEND = "pyfory-rebuild"
PYFORY_RELOAD_BACKEND = "pyfory-reload"
LEAF_BACKEND = "leaf"
GRAPH_FINGERPRINT_METHOD = "graph"
NL_FINGERPRINT_METHOD = "nl-sha256"


def _is_pyfory_backend(backend):
    return backend in (PYFORY_REBUILD_BACKEND, PYFORY_RELOAD_BACKEND)

# The rebuild codec carries a plain-data state snapshot rather than a Pyomo
# graph. Its explicit limits bound malformed or unexpectedly deep payloads.
PYFORY_MAX_DEPTH = 10000
PYFORY_MAX_GRAPH_MEMORY_BYTES = 1 << 30
PYFORY_SUPPORTED_VERSION = "1.7.7"
PYFORY_STATE_VERSION = 2

MANIFEST_NAME = "manifest.json"
HUB_SUBDIR = "hub"
SPOKES_SUBDIR = "spokes"

# Option keys that must match for a checkpoint to be resumable. Deliberately a
# named subset rather than the whole configuration: these are the entries that
# change the *structure* of the scenario models or the meaning of the state
# riding in them, so a mismatch means the checkpoint cannot be restored into
# this run. Everything else is free to change between a stop and a resume --
# notably the iteration limit and the time limit, which a user legitimately
# adjusts when picking a run back up the next morning, and the display/verbosity
# options, which have no bearing on the state at all.
STRUCTURAL_OPTION_KEYS = (
    "defaultPHrho",
    "linearize_proximal_terms",
    "linearize_binary_proximal_terms",
    "proximal_linearization_tolerance",
    "smoothed",
    "defaultPHp",
    "defaultPHbeta",
)

# Configuration entries a resume may legitimately differ on. Everything else
# in the cfg is folded into the fingerprint, so this is a *denylist*: a new
# option is checked by default, and only becomes exempt when someone decides it
# cannot make a restored checkpoint describe a different problem. The opposite
# policy -- naming the structural options -- silently missed everything a
# model's own inparser_adder registers, so a farmer checkpoint could be resumed
# with --farmer-with-integers and quietly answer the LP.
NON_STRUCTURAL_CFG_KEYS = frozenset({
    # How long to run. Resuming with a different budget is the point, and
    # that goes for both bounds: --max-iterations sizes the run being started
    # and --stop-at-iteration-number sizes the study it belongs to. The Slurm
    # buffer is likewise a budget for the current job, not saved model state.
    "max_iterations", "stop_at_iteration_number", "time_limit",
    "slurm_buffer_seconds",
    "intra_hub_conv_thresh", "rel_gap", "abs_gap", "max_stalled_iters",
    # Checkpoint plumbing itself. How often a checkpoint is written is a
    # cadence knob like the iteration limit: it changes what a stop costs, not
    # what problem the checkpoint describes.
    "checkpoint_dir", "checkpoint_backend", "checkpoint_every_iterations",
    "checkpoint_before_seconds", "checkpoint_component_fingerprint_cache",
    "checkpoint_nl_fingerprint",
    "resume_from",
    # Display, logging and output destinations.
    "verbose", "display_progress", "display_timing",
    "display_convergence_detail", "tee_rank0_solves", "trace_prefix",
    "solution_base_name", "write_xhat_file", "xhat_from_file",
    "solver_log_dir", "incumbent_on_improvement_filename_prefix",
    "W_fname", "Xbar_fname", "init_W_fname", "init_Xbar_fname",
    "separate_W_files", "init_separate_W_files",
    "wtracker", "wtracker_file_prefix", "wtracker_wlen",
    "wtracker_reportlen", "wtracker_stdevthresh",
    # Which solver and how it is driven: a different solver continues the same
    # problem, it does not redefine it.
    "solver_name", "solver_options", "max_solver_threads",
    "presolve", "user_warmstart", "warmstart_subproblems",
    # Every per-cylinder *_solver_options_file is exempt via the suffix rule
    # below; the global one has no prefix to match, and the same setting
    # should not become structural merely by being written in a file.
    "solver_options_file",
    # Per-cylinder solver selection and gap control. Tightening a mipgap on day
    # two of a multi-day study is the most ordinary adjustment there is, and it
    # continues the same problem rather than redefining it.
    "starting_mipgap", "mipgap_ratio", "mipgaps_json",
    # Diagnostics, tracing and IIS output: they observe a run, never shape it.
    "track_convergence", "track_duals", "track_nonants", "track_xbars",
    "track_reduced_costs", "tracking_folder", "ph_track_progress",
    "track_scen_gaps",
    "xhatter_write_iis", "xhatter_iis_method", "xhatter_iis_dir",
    "rc_debug", "rc_verbose", "tee_EF", "hub_only_solver_logs",
    "inspect_buffers_on_shutdown", "fwph_save_file",
    "write_scenario_lp_mps_files_dir", "config_file",
    # Which cylinders run. A different spoke set can change where the hub
    # goes (with --ph-primal-hub the dual spokes feed it W), but it leaves
    # nothing in the hub's models, so the checkpoint still describes the same
    # problem. reduced_costs and cross_scenario_cuts are not here: each also
    # gives the hub an extension that changes the hub's own models (fixing
    # variables, adding cuts), and a resume without it would keep them.
    "lagrangian", "xhatshuffle", "xhatxbar", "xhatlshaped", "fwph",
    "subgradient", "ph_primal_hub", "ph_dual", "relaxed_ph",
    "ph_xfeas_spoke", "lagranger", "xhatlooper", "xhatspecific",
    "slammax", "slammin", "certified_outer_bound",
    # This hub heuristic runs only after the checkpointed PH iterations.
    "final_xhat",
    # The certified spoke's cushion only loosens the bound that spoke
    # reports; the hub keeps its best outer bound across a resume either way.
    "certified_outer_bound_cushion",
    # How the run was set up, not what problem it is. out-of-the-box prints an
    # equivalent command line and invites the user to reuse it, so leaving
    # these structural refuses a resume of exactly the line it printed.
    "out_of_the_box", "out_of_the_box_minus", "out_of_the_box_plus",
    "inspect_only",
})


def _is_non_structural(key):
    """True when a cfg entry may differ between the write and the resume.

    Beyond the explicit names above, every per-cylinder solver knob follows a
    naming convention (``<cylinder>_solver_name``, ``_solver_options``,
    ``_solver_options_file``, ``_mipgap``, ``_rank_ratio``), and enumerating
    them by hand would go stale the moment a cylinder is added.
    """
    if key in NON_STRUCTURAL_CFG_KEYS:
        return True
    return key.endswith((
        "_solver_name", "_solver_options", "_solver_options_file",
        "_mipgap", "_mipgaps_json", "_mipgap_ratio",
        "_rank_ratio", "_solver_log_dir",
    ))


class CheckpointMismatch(RuntimeError):
    """A checkpoint exists but cannot be resumed into the current run."""


def _canonical(value):
    """Render an option value as something JSON can hash reproducibly."""
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def structural_fingerprint(options):
    """Hash the structural subset of ``options`` (see STRUCTURAL_OPTION_KEYS)."""
    payload = {k: _canonical(options.get(k)) for k in STRUCTURAL_OPTION_KEYS}
    extras = options.get("checkpoint_structural_cfg") or {}
    for k, v in sorted(extras.items()):
        payload[f"cfg:{k}"] = _canonical(v)
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def require_checkpoint_structural_cfg(options):
    """Require direct callers to state the model configuration they own.

    The generic-cylinder path builds this from every structural Config entry,
    but direct PH/Xhat callers hand us only ``opt.options``. Their scenario
    creator and its arguments otherwise never reach the fingerprint, so a
    checkpoint for one model can be resumed as another with the same geometry.
    An empty dict is valid for a creator with no structural inputs; omitting
    the key is not.
    """
    if "checkpoint_structural_cfg" not in options:
        raise RuntimeError(
            "Checkpointing through the direct API requires "
            "options['checkpoint_structural_cfg'] to describe the model "
            "creator and its structural inputs. The generic-cylinder path "
            "sets it automatically.")
    if options.get("checkpoint_component_fingerprint_cache", False) and \
            options.get("checkpoint_backend", DILL_RELOAD_BACKEND) != \
            PYFORY_REBUILD_BACKEND:
        raise RuntimeError(
            "checkpoint_component_fingerprint_cache requires the "
            "pyfory-rebuild checkpoint backend.")
    if options.get("checkpoint_nl_fingerprint", False):
        if options.get("checkpoint_backend", DILL_RELOAD_BACKEND) != \
                PYFORY_REBUILD_BACKEND:
            raise RuntimeError(
                "checkpoint_nl_fingerprint requires the pyfory-rebuild "
                "checkpoint backend.")
        if options.get("checkpoint_component_fingerprint_cache", False):
            raise RuntimeError(
                "checkpoint_nl_fingerprint cannot be combined with "
                "checkpoint_component_fingerprint_cache.")
    if not isinstance(options["checkpoint_structural_cfg"], dict):
        raise RuntimeError(
            "options['checkpoint_structural_cfg'] must be a dict of stable "
            "model configuration values.")


def sanitize_for_filename(name):
    """Make a scenario name safe to embed in a file name.

    Never use ``sputils.extract_num`` here: it scrapes trailing digits, which
    are not unique for ADMM's wrapped scenario names.
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))


def check_filename_collisions(scenario_names):
    """Refuse scenario names whose sanitized file names collide.

    Two distinct names that sanitize to the same fragment (``scen 1`` and
    ``scen_1``, say) would write to the same model file -- the second
    silently overwriting the first -- and a resume would then restore one
    scenario's model for both, with no error anywhere. The comparison ignores
    case, because on a case-insensitive filesystem (the macOS and Windows
    defaults) ``Scenario1`` and ``scenario1`` are the same file too.
    """
    seen = {}
    for sname in scenario_names:
        fragment = sanitize_for_filename(sname)
        key = fragment.casefold()
        other = seen.get(key)
        if other is not None:
            if sanitize_for_filename(other) == fragment:
                why = ("once unsafe characters are replaced, so their "
                       "checkpoints would overwrite each other")
            else:
                why = ("apart from case, so their checkpoints would "
                       "overwrite each other on a filesystem that ignores "
                       "case")
            raise RuntimeError(
                f"Scenario names '{other}' and '{sname}' both map to "
                f"'{fragment}' in checkpoint file names {why}. Rename the "
                f"scenarios so the names stay distinct."
            )
        seen[key] = sname


def _generation_dirname(generation):
    return f"gen_{generation:04d}"


def _leaf_filename(rank):
    return f"hub_rank_{rank:04d}.pkl"


def _model_filename(rank, sname, backend=DILL_RELOAD_BACKEND):
    suffix = {
        DILL_RELOAD_BACKEND: "dill",
        PYFORY_REBUILD_BACKEND: "fory",
        PYFORY_RELOAD_BACKEND: "fory",
    }.get(backend)
    if suffix is None:
        raise ValueError(f"Unknown checkpoint backend '{backend}'.")
    return (f"hub_rank_{rank:04d}_scen_"
            f"{sanitize_for_filename(sname)}.{suffix}")


def _nl_fingerprint_filename(rank, sname):
    return (f"hub_rank_{rank:04d}_scen_"
            f"{sanitize_for_filename(sname)}.nl.sha256")


def require_backend_dependency(backend):
    """Check the optional codec needed by a hub checkpoint backend."""
    if backend == DILL_RELOAD_BACKEND:
        if not dill_available:
            raise RuntimeError(
                "The '{}' checkpoint backend requires dill, which is not "
                "installed. Install the optional dependencies with "
                "'pip install mpi-sppy[extras]' (or 'pip install dill'), or "
                "choose a different --checkpoint-backend.".format(backend)
            )
    elif _is_pyfory_backend(backend):
        if not pyfory_available:
            raise RuntimeError(
                "The '{}' checkpoint backend requires pyfory, which is not "
                "installed. Install it with 'pip install mpi-sppy[pyfory]' "
                "or choose a different --checkpoint-backend.".format(backend)
            )
        installed_version = _pyfory_version()
        if installed_version != PYFORY_SUPPORTED_VERSION:
            raise RuntimeError(
                f"The '{backend}' checkpoint backend supports pyfory "
                f"{PYFORY_SUPPORTED_VERSION}, but this run has pyfory "
                f"{installed_version}. Install the supported version with "
                "'pip install mpi-sppy[pyfory]'."
            )
        if backend == PYFORY_RELOAD_BACKEND:
            from mpisppy.utils import pyfory_model_codec
            pyfory_model_codec.require_supported_runtime()
    else:
        require_implemented_backend(backend)


def _new_fory():
    """Return an independently usable Python-native codec for state data."""
    require_backend_dependency(PYFORY_REBUILD_BACKEND)
    return pyfory.Fory(
        xlang=False,
        ref=True,
        strict=False,
        max_depth=PYFORY_MAX_DEPTH,
        max_graph_memory_bytes=PYFORY_MAX_GRAPH_MEMORY_BYTES,
    )


_CHECKPOINT_SCALAR_TAG = "__mpisppy_checkpoint_scalar__"


def _plain_checkpoint_value(value, context=None):
    """Convert a Pyomo/NumPy scalar to a bounded pyfory-native value.

    ``context`` identifies the scenario/component/index at the call site so
    unsupported model values can be fixed without tracing a generic type
    error back through a large scenario creator.
    """
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, Decimal):
        # Preserve exact decimal values without relying on codec-specific
        # support for Decimal itself.
        return {_CHECKPOINT_SCALAR_TAG: "decimal", "value": str(value)}
    if (isinstance(value, dict)
            and value.get(_CHECKPOINT_SCALAR_TAG) == "decimal"
            and set(value) == {_CHECKPOINT_SCALAR_TAG, "value"}
            and isinstance(value["value"], str)):
        # _snapshot_rebuild_model prepares scalar metadata before the packed
        # value arrays call this helper a second time.
        return value
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, numbers.Real):
        return float(value)
    location = f" at {context}" if context else ""
    raise TypeError(
        f"Checkpoint state{location} contains unsupported value type "
        f"'{type(value).__module__}.{type(value).__qualname__}'."
    )


def _restore_plain_checkpoint_value(value):
    """Decode the explicit scalar tags emitted by _plain_checkpoint_value."""
    if (isinstance(value, dict)
            and _CHECKPOINT_SCALAR_TAG in value):
        if (value.get(_CHECKPOINT_SCALAR_TAG) != "decimal"
                or set(value) != {_CHECKPOINT_SCALAR_TAG, "value"}
                or not isinstance(value["value"], str)):
            raise CheckpointMismatch(
                "A pyfory checkpoint has an invalid tagged scalar value.")
        try:
            return Decimal(value["value"])
        except (TypeError, ValueError, InvalidOperation) as exc:
            raise CheckpointMismatch(
                "A pyfory checkpoint has an invalid tagged Decimal value."
            ) from exc
    return value


def _pack_checkpoint_values(values):
    """Pack homogeneous numeric snapshot data into bounded NumPy arrays.

    Most scenario state is numeric. Keeping those values in arrays avoids a
    Python object and pyfory graph entry for every variable or mutable Param,
    while the kind and overflow side tables retain None, integer, and less
    common nonnumeric values without silently coercing them.
    """
    values = [_plain_checkpoint_value(value) for value in values]
    kinds = np.zeros(len(values), dtype=np.uint8)
    integers = np.zeros(len(values), dtype=np.int64)
    reals = np.zeros(len(values), dtype=np.float64)
    others = {}
    int_info = np.iinfo(np.int64)
    for index, value in enumerate(values):
        if value is None:
            continue
        if isinstance(value, bool):
            # Keep the Python type: a mutable Param may use a Boolean value
            # in model logic, where 1 and True are not interchangeable.
            kinds[index] = 3
            others[index] = value
        elif isinstance(value, int):
            if int_info.min <= value <= int_info.max:
                kinds[index] = 1
                integers[index] = value
            else:
                kinds[index] = 3
                others[index] = value
        elif isinstance(value, numbers.Real):
            kinds[index] = 2
            reals[index] = float(value)
        else:
            kinds[index] = 3
            others[index] = value
    return {
        "kinds": kinds,
        "integers": integers,
        "reals": reals,
        "others": others,
    }


def _unpack_checkpoint_value(packed, index):
    """Read one value from :func:`_pack_checkpoint_values`."""
    try:
        kind = int(packed["kinds"][index])
        if kind == 0:
            return None
        if kind == 1:
            return int(packed["integers"][index])
        if kind == 2:
            return float(packed["reals"][index])
        if kind == 3 and index in packed["others"]:
            return _restore_plain_checkpoint_value(packed["others"][index])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise CheckpointMismatch(
            "A pyfory checkpoint has a malformed packed value array."
        ) from exc
    raise CheckpointMismatch(
        "A pyfory checkpoint has an unknown packed value kind."
    )


def _checkpoint_state_equal(left, right):
    """Compare a plain snapshot recursively, including NumPy arrays."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray)
                and isinstance(right, np.ndarray)
                and left.dtype == right.dtype
                and left.shape == right.shape
                and np.array_equal(left, right, equal_nan=True))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(_checkpoint_state_equal(left[key], right[key])
                        for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_checkpoint_state_equal(a, b)
                        for a, b in zip(left, right)))
    return left == right


def _safe_index(index, context=None):
    """Convert a Pyomo index into plain data without losing tuple geometry."""
    if index is None:
        return None
    if isinstance(index, tuple):
        return tuple(_safe_index(part, context) for part in index)
    if isinstance(index, (bool, int, float, str, bytes)):
        return index
    if hasattr(index, "item") and callable(index.item):
        return _safe_index(index.item(), context)
    location = f" at {context}" if context else ""
    raise TypeError(
        f"Checkpoint index {index!r}{location} has unsupported type "
        f"'{type(index).__module__}.{type(index).__qualname__}'."
    )


def _component_layout(scenario):
    """Describe model structure that scenario_creator/extensions rebuild."""
    layout = []
    for component in scenario.component_objects(descend_into=True,
                                                active=None):
        try:
            size = len(component)
        except TypeError:
            size = 1
        layout.append((
            component.name,
            type(component).__module__,
            type(component).__qualname__,
            int(size),
            bool(component.active),
        ))
    return sorted(layout)


def _prox_cut_owner_and_index(key):
    """Split either Pyomo tuple-key form into a manager key and cut index."""
    if not isinstance(key, tuple) or len(key) < 2:
        return None
    if len(key) == 2 and isinstance(key[0], tuple):
        return tuple(key[0]), int(key[1])
    return tuple(key[:-1]), int(key[-1])


def _prox_approx_snapshot(scenario):
    """Store PH's proximal cuts and their manager bookkeeping."""
    managers = getattr(scenario._mpisppy_data, "xsqvar_prox_approx", False)
    if not managers:
        return None
    cuts = scenario._mpisppy_model.xsqvar_cuts
    cuts_by_manager = {}
    for key, cut in cuts.items():
        owner_and_index = _prox_cut_owner_and_index(key)
        if owner_and_index is not None:
            owner, index = owner_and_index
            cuts_by_manager.setdefault(owner, []).append((index, cut))
    result = {}
    for manager in managers.values():
        var_name = manager.xvar.name
        manager_cuts = cuts_by_manager.get(tuple(manager.var_index), ())
        if manager.xvar.is_integer():
            result[var_name] = {
                "kind": "discrete",
                "cut_indices": sorted(index for index, _ in manager_cuts),
            }
        else:
            from pyomo.core.expr.numeric_expr import LinearExpression
            from pyomo.environ import value as pyo_value

            cut_records = []
            for index, cut in manager_cuts:
                body = cut.body
                if not isinstance(body, LinearExpression):
                    raise TypeError(
                        f"Proximal cut '{cut.name}' is not linear and cannot "
                        "be saved by pyfory-rebuild."
                    )
                variables = body.linear_vars
                if (len(variables) != 2
                        or variables[0] is not manager.xvarsqrd
                        or variables[1] is not manager.xvar):
                    raise TypeError(
                        f"Proximal cut '{cut.name}' has unexpected variables "
                        "and cannot be saved by pyfory-rebuild."
                    )
                coefs = body.linear_coefs
                cut_records.append((
                    index,
                    _plain_checkpoint_value(coefs[0], cut.name),
                    _plain_checkpoint_value(coefs[1], cut.name),
                    _plain_checkpoint_value(body.constant, cut.name),
                    _plain_checkpoint_value(
                        pyo_value(cut.lower, exception=False),
                        f"scenario '{scenario.name}', proximal cut "
                        f"'{cut.name}' lower bound"),
                    _plain_checkpoint_value(
                        pyo_value(cut.upper, exception=False),
                        f"scenario '{scenario.name}', proximal cut "
                        f"'{cut.name}' upper bound"),
                    bool(cut.active),
                ))
            result[var_name] = {
                "kind": "continuous",
                # 0.0 is the manager's initial duplicate-suppression marker;
                # the corresponding epigraph lower bound is xsqvar >= 0.
                "cut_values": [float(value)
                               for value in manager.cut_values
                               if float(value) != 0.0],
                "cut_index": int(manager.cut_index),
                "cut_records": sorted(cut_records),
            }
    return result


def _snapshot_rebuild_model(scenario):
    """Capture plain PH/scenario state for the pyfory-rebuild backend.

    The scenario structure is rebuilt from the creator and approved extension
    hooks. Variable values/fixedness, mutable Params, and proximal cuts are
    overlaid by stable component names and indices. Solver result flags and
    bounds are saved separately so resume hooks see the completed iterate.
    """
    from pyomo.environ import Param, Var, value as pyo_value

    variables = []
    variable_values = []
    for var in scenario.component_data_objects(Var, descend_into=True,
                                               active=None):
        value = _plain_checkpoint_value(
            var.value,
            f"scenario '{scenario.name}', variable '{var.name}' value")
        lb_value = pyo_value(var.lb, exception=False)
        ub_value = pyo_value(var.ub, exception=False)
        lb = (None if lb_value is None else _plain_checkpoint_value(
            lb_value,
            f"scenario '{scenario.name}', variable '{var.name}' lower bound"))
        ub = (None if ub_value is None else _plain_checkpoint_value(
            ub_value,
            f"scenario '{scenario.name}', variable '{var.name}' upper bound"))
        domain_name = getattr(var.domain, "name", None)
        if domain_name is None:
            raise TypeError(
                f"Variable '{var.name}' has a domain without a stable name; "
                "pyfory-rebuild cannot save it."
            )
        variables.append((
            var.name, bool(var.fixed), bool(var.stale), lb, ub,
            str(domain_name),
        ))
        variable_values.append(value)

    parameters = []
    parameter_values = []
    for component in scenario.component_objects(Param, descend_into=True,
                                                active=None):
        if not component.mutable:
            continue
        for index, param in component.items():
            safe_index = _safe_index(
                index,
                f"scenario '{scenario.name}', mutable Param "
                f"'{component.name}' index")
            parameters.append((component.name, safe_index))
            parameter_values.append(_plain_checkpoint_value(
                param.value,
                f"scenario '{scenario.name}', mutable Param "
                f"'{component.name}' index {safe_index!r}"))

    return {
        "state_version": PYFORY_STATE_VERSION,
        "scenario_name": str(scenario.name),
        "component_layout": _component_layout(scenario),
        "variables": variables,
        "variable_values": _pack_checkpoint_values(variable_values),
        "mutable_params": parameters,
        "mutable_param_values": _pack_checkpoint_values(parameter_values),
        "prox_approx": _prox_approx_snapshot(scenario),
        # The startup probe runs before the first solve, when this flag has
        # not yet been created on a fresh scenario.
        "solution_available": bool(getattr(
            scenario._mpisppy_data, "solution_available", False)),
        "inner_bound": _plain_checkpoint_value(
            scenario._mpisppy_data.inner_bound,
            f"scenario '{scenario.name}' inner bound"),
        "outer_bound": _plain_checkpoint_value(
            scenario._mpisppy_data.outer_bound,
            f"scenario '{scenario.name}' outer bound"),
    }


def _fingerprint_field(digest, tag, payload=b""):
    """Add one length-delimited field to a canonical state fingerprint."""
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _fingerprint_snapshot_value(digest, value):
    """Hash the plain-data snapshot without relying on dict insertion order."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot be fingerprinted safely")
        array_value = np.ascontiguousarray(value)
        metadata = (value.dtype.str + ":" +
                    ",".join(str(size) for size in value.shape)).encode(
                        "ascii")
        _fingerprint_field(digest, b"A", metadata)
        _fingerprint_field(digest, b"a", array_value.tobytes(order="C"))
    elif isinstance(value, np.generic):
        _fingerprint_snapshot_value(digest, value.item())
    elif value is None:
        _fingerprint_field(digest, b"N")
    elif isinstance(value, bool):
        _fingerprint_field(digest, b"B", b"1" if value else b"0")
    elif isinstance(value, int):
        _fingerprint_field(digest, b"I", str(value).encode("ascii"))
    elif isinstance(value, float):
        _fingerprint_field(digest, b"F", struct.pack("!d", value))
    elif isinstance(value, Decimal):
        _fingerprint_field(digest, b"D", str(value).encode("ascii"))
    elif isinstance(value, str):
        _fingerprint_field(digest, b"S", value.encode("utf-8"))
    elif isinstance(value, bytes):
        _fingerprint_field(digest, b"Y", value)
    elif isinstance(value, (tuple, list)):
        _fingerprint_field(
            digest, b"T" if isinstance(value, tuple) else b"L",
            len(value).to_bytes(8, "big"))
        for item in value:
            _fingerprint_snapshot_value(digest, item)
    elif isinstance(value, dict):
        _fingerprint_field(digest, b"M", len(value).to_bytes(8, "big"))
        # Snapshot mapping keys are strings, integers, or safe index tuples.
        # Their repr is stable and avoids comparing unlike key types.
        for key in sorted(
                value,
                key=lambda item: (type(item).__module__,
                                  type(item).__qualname__, repr(item))):
            _fingerprint_snapshot_value(digest, key)
            _fingerprint_snapshot_value(digest, value[key])
    else:
        raise TypeError(
            "Unsupported value in pyfory-rebuild model fingerprint: "
            f"{type(value).__module__}.{type(value).__qualname__}")


def _fingerprint_graph_value(digest, value):
    """Hash an expression tree without evaluating mutable references."""
    if isinstance(value, NumericConstant):
        _fingerprint_field(digest, b"constant")
        _fingerprint_snapshot_value(digest, value.value)
    elif getattr(value, "is_component_type", lambda: False)():
        _fingerprint_field(digest, b"reference", value.name.encode("utf-8"))
    elif getattr(value, "is_expression_type", lambda: False)():
        cls = type(value)
        if isinstance(value, ExternalFunctionExpression):
            raise TypeError(
                "External functions cannot be fingerprinted by "
                "pyfory-rebuild")
        if (isinstance(value, UnaryFunctionExpression)
                and getattr(value._fcn, "__module__", None)
                not in ("math", "builtins")):
            raise TypeError(
                "Custom expression functions cannot be fingerprinted by "
                "pyfory-rebuild")
        if cls.__module__.startswith("pyomo.core.expr."):
            _fingerprint_field(
                digest, b"expression",
                f"{cls.__module__}.{cls.__qualname__}".encode("utf-8"))
            # UnaryFunctionExpression stores the operation outside its args.
            if isinstance(value, UnaryFunctionExpression):
                _fingerprint_field(
                    digest, b"operation", value.getname().encode("utf-8"))
            args = value.args
            _fingerprint_field(digest, b"arity", len(args).to_bytes(8, "big"))
            for arg in args:
                _fingerprint_graph_value(digest, arg)
        else:
            raise TypeError(
                "Unsupported expression in pyfory-rebuild fingerprint: "
                f"{cls.__module__}.{cls.__qualname__}")
    elif isinstance(value, (tuple, list)):
        _fingerprint_field(digest, b"tuple" if isinstance(value, tuple)
                           else b"list", len(value).to_bytes(8, "big"))
        for item in value:
            _fingerprint_graph_value(digest, item)
    else:
        _fingerprint_snapshot_value(digest, value)


def _fingerprint_domain(digest, domain):
    """Include custom finite domain members, not just the domain name."""
    cls = type(domain)
    _fingerprint_graph_value(digest, (
        f"{cls.__module__}.{cls.__qualname__}", domain.name))
    if domain.isfinite():
        members = [_safe_index(item, domain.name) for item in domain]
        if not domain.isordered():
            members.sort(key=repr)
        _fingerprint_graph_value(digest, tuple(members))


def _add_pyomo_component_fingerprint(scenario, component, digest, pyo=None):
    """Hash one supported Pyomo component, including inactive expressions."""
    if pyo is None:
        import pyomo.environ as pyo

    name = component.name
    _fingerprint_field(digest, b"component", name.encode("utf-8"))
    cls = type(component)
    _fingerprint_field(digest, b"type", f"{cls.__module__}."
                       f"{cls.__qualname__}".encode("utf-8"))
    _fingerprint_graph_value(digest, bool(component.active))
    if isinstance(component, (pyo.BuildAction, pyo.BuildCheck)):
        # These rules run during construction and store no model data.
        # Fingerprint their effects through the components they created.
        return
    if isinstance(component, pyo.Suffix):
        _fingerprint_graph_value(
            digest, (int(component.direction),
                     None if component.datatype is None else
                     int(component.datatype)))
        if name == "_relaxed_integer_vars" or name.endswith(
                "._relaxed_integer_vars"):
            # Pyomo stores the undo map at key None. Its inner keys are
            # id(var), which change across a rebuild or model clone.
            suffix_entries = list(component.items())
            if (len(suffix_entries) != 1 or
                    suffix_entries[0][0] is not None or
                    not isinstance(suffix_entries[0][1], dict)):
                raise TypeError(
                    f"Unexpected integer-relaxation suffix '{name}'")
            undo_map = suffix_entries[0][1]
            relaxed = []
            for entry in undo_map.values():
                if not isinstance(entry, tuple) or len(entry) != 2:
                    raise TypeError(
                        f"Unexpected integer-relaxation entry in '{name}'")
                var, original_domain = entry
                if (not getattr(var, "is_variable_type", lambda: False)()
                        or scenario.find_component(var.name) is not var):
                    raise TypeError(
                        f"Integer-relaxation suffix '{name}' refers to "
                        "a variable outside its scenario")
                relaxed.append((var.name, original_domain))
            relaxed.sort(key=lambda entry: entry[0])
            _fingerprint_field(digest, b"entries",
                               len(relaxed).to_bytes(8, "big"))
            for var_name, original_domain in relaxed:
                _fingerprint_graph_value(digest, var_name)
                _fingerprint_domain(digest, original_domain)
        elif component.export_enabled():
            entries = sorted(component.items(), key=lambda pair:
                             pair[0].name)
            _fingerprint_field(digest, b"entries",
                               len(entries).to_bytes(8, "big"))
            for key, value in entries:
                _fingerprint_graph_value(digest, key)
                _fingerprint_graph_value(digest, value)
        else:
            # IMPORT values are solver results, not part of the problem.
            raise TypeError(
                f"Non-export Suffix '{name}' may acquire state "
                "that pyfory-rebuild does not restore")
        return
    if isinstance(component, (pyo.Set, pyo.RangeSet)):
        # RangeSet is a scalar SetData component, not a pyo.Set, and
        # does not expose the indexed component items() interface.
        entries = ([(None, component)] if isinstance(component, pyo.RangeSet)
                   else sorted(component.items(), key=lambda pair:
                               repr(_safe_index(pair[0], name))))
        for index, data in entries:
            if not data.isfinite():
                raise TypeError(f"Cannot fingerprint infinite Set '{name}'")
            _fingerprint_graph_value(digest, _safe_index(index, name))
            dimen = data.dimen
            if dimen is UnknownSetDimen:
                _fingerprint_field(digest, b"set-dimen-unknown")
            else:
                _fingerprint_graph_value(digest, dimen)
            ordered = data.isordered()
            _fingerprint_graph_value(digest, ordered)
            members = [_safe_index(item, name) for item in data]
            if not ordered:
                members.sort(key=repr)
            _fingerprint_graph_value(digest, tuple(members))
        return
    if not isinstance(component, (pyo.Block, pyo.Var, pyo.Param,
                                  pyo.Constraint, pyo.Objective,
                                  pyo.Expression, pyo.SOSConstraint)):
        raise TypeError(
            f"Unsupported Pyomo component in pyfory-rebuild "
            f"fingerprint: '{name}' ({type(component).__name__})")
    if isinstance(component, pyo.Param):
        _fingerprint_graph_value(digest, bool(component.mutable))
        _fingerprint_domain(digest, component.domain)
    entries = sorted(component.items(), key=lambda pair: repr(
        _safe_index(pair[0], name)))
    _fingerprint_field(digest, b"entries",
                       len(entries).to_bytes(8, "big"))
    for index, data in entries:
        _fingerprint_graph_value(digest, _safe_index(index, name))
        if isinstance(component, pyo.Param):
            # Immutable indexed Params yield native scalars, not ParamData.
            if component.mutable:
                _fingerprint_graph_value(digest, data.name)
            else:
                _fingerprint_snapshot_value(digest, pyo.value(data))
            continue
        _fingerprint_graph_value(digest, data.name)
        _fingerprint_graph_value(
            digest, bool(getattr(data, "active", True)))
        if isinstance(component, pyo.Var):
            _fingerprint_domain(digest, data.domain)
            _fingerprint_graph_value(digest, data.lower)
            _fingerprint_graph_value(digest, data.upper)
        elif isinstance(component, pyo.Constraint):
            _fingerprint_graph_value(digest, data.expr)
        elif isinstance(component, pyo.Objective):
            _fingerprint_graph_value(digest, int(data.sense))
            _fingerprint_graph_value(digest, data.expr)
        elif isinstance(component, pyo.Expression):
            _fingerprint_graph_value(digest, data.expr)
        elif isinstance(component, pyo.SOSConstraint):
            _fingerprint_graph_value(digest, int(data.level))
            _fingerprint_graph_value(digest, tuple(data.get_items()))


def _add_pyomo_graph_fingerprint(scenario, digest):
    """Hash supported Pyomo components, including inactive expressions."""
    import pyomo.environ as pyo

    components = sorted(
        scenario.component_objects(descend_into=True, active=None),
        key=lambda component: component.name)
    for component in components:
        _add_pyomo_component_fingerprint(scenario, component, digest, pyo)


def _rebuild_model_fingerprint(scenario):
    """Fingerprint the supported Pyomo graph."""
    digest = hashlib.sha256()
    digest.update(b"mpi-sppy pyfory-rebuild graph v1\0")
    _add_pyomo_graph_fingerprint(scenario, digest)
    return digest.hexdigest()


class _HashingTextStream(io.TextIOBase):
    """Hash UTF-8 text as an NL writer emits it, without retaining the file."""

    def __init__(self, timing=None):
        self._hasher = hashlib.sha256()
        self._timing = timing

    def write(self, text):
        if not isinstance(text, str):
            raise TypeError("NLWriter must write text")
        started = time.perf_counter() if self._timing is not None else None
        cpu_started = time.thread_time() if self._timing is not None else None
        self._hasher.update(text.encode("utf-8"))
        if self._timing is not None:
            self._timing.nl_digest_s += time.perf_counter() - started
            self._timing.nl_digest_cpu_s += (
                time.thread_time() - cpu_started)
        return len(text)

    def tell(self):
        raise io.UnsupportedOperation("tell")

    def writable(self):
        return True

    def flush(self):
        return None

    def digest(self):
        return self._hasher.hexdigest()


def _rebuild_model_nl_fingerprint(scenario, timing=None):
    """Hash Pyomo's deterministic NL text for the active scenario model."""
    from pyomo.repn.plugins.nl_writer import FileDeterminism, NLWriter

    stream = _HashingTextStream(timing)
    started = time.perf_counter() if timing is not None else None
    cpu_started = time.thread_time() if timing is not None else None
    try:
        NLWriter().write(
            scenario,
            stream,
            file_determinism=FileDeterminism.SORT_SYMBOLS,
            symbolic_solver_labels=True,
            linear_presolve=False,
            skip_trivial_constraints=False,
            show_section_timing=False,
            scale_model=True,
            export_nonlinear_variables=None,
            row_order=None,
            column_order=None,
            export_defined_variables=True,
        )
    finally:
        if timing is not None:
            timing.nl_write_s += time.perf_counter() - started
            timing.nl_write_cpu_s += time.thread_time() - cpu_started
    return stream.digest()


def _rebuild_model_fingerprint_with_cache(scenario, component_cache,
                                          dirty_components=None):
    """Fingerprint the graph from cached per-component digests.

    Component digests use the same canonical encoding as the uncached
    fingerprint, then a versioned root hash combines the named component
    digests. Callers that opt into caching must mark a component dirty whenever
    any fingerprinted property of that component changes.
    """
    components = sorted(
        scenario.component_objects(descend_into=True, active=None),
        key=lambda component: component.name)
    import pyomo.environ as pyo

    dirty_components = dirty_components or set()
    recompute_all = None in dirty_components
    current_cache = {}
    digest = hashlib.sha256()
    digest.update(b"mpi-sppy pyfory-rebuild graph v2\0")
    for component in components:
        name = component.name
        cached = component_cache.get(name)
        if (not recompute_all and name not in dirty_components
                and cached is not None and cached[0] is component):
            component_digest = cached[1]
        else:
            component_hasher = hashlib.sha256()
            _add_pyomo_component_fingerprint(
                scenario, component, component_hasher, pyo)
            component_digest = component_hasher.digest()
        current_cache[name] = (component, component_digest)
        _fingerprint_field(digest, b"component", name.encode("utf-8"))
        _fingerprint_field(digest, b"digest", component_digest)
    component_cache.clear()
    component_cache.update(current_cache)
    if isinstance(dirty_components, set):
        dirty_components.clear()
    return digest.hexdigest()


def mark_rebuild_model_fingerprint_dirty(opt, scenario, component_name=None):
    """Invalidate cached graph digests after changing a Pyomo component.

    This API is for callers that explicitly enable
    ``checkpoint_component_fingerprint_cache``. Omitting ``component_name``
    invalidates every component in the named scenario. A changed Set also
    invalidates the full scenario because Var and Param domain fingerprints
    can include its members. Component data invalidates its parent component.
    """
    options = getattr(opt, "options", None) or {}
    if not options.get("checkpoint_component_fingerprint_cache", False):
        return
    local_scenarios = opt.local_scenarios
    if isinstance(scenario, str):
        if scenario not in local_scenarios:
            raise ValueError(f"Unknown local scenario '{scenario}'")
        scenario_name = scenario
        scenario_model = local_scenarios[scenario]
    else:
        scenario_name = next((name for name, model in
                              local_scenarios.items() if model is scenario),
                             None)
        if scenario_name is None:
            raise ValueError("Scenario is not in opt.local_scenarios")
        scenario_model = scenario
    if component_name is not None:
        component = (scenario_model.find_component(component_name)
                     if isinstance(component_name, str) else component_name)
        if component is None or not hasattr(component, "parent_component"):
            raise ValueError(f"Unknown Pyomo component {component_name!r}")
        component = component.parent_component()
        if (component is None or
                scenario_model.find_component(component.name) is not component):
            raise ValueError("Component is not in the scenario")
        component_name = component.name
        if isinstance(component, (Set, RangeSet)):
            component_name = None
    dirty_by_scenario = getattr(
        opt, "_pyfory_rebuild_fingerprint_dirty", None)
    if dirty_by_scenario is None:
        dirty_by_scenario = {}
        opt._pyfory_rebuild_fingerprint_dirty = dirty_by_scenario
    dirty_by_scenario.setdefault(scenario_name, set()).add(component_name)


def validate_rebuild_model_fingerprints(
        opt, snapshots, fingerprint_method=GRAPH_FINGERPRINT_METHOD,
        nl_fingerprints=None):
    """Refuse a rebuild whose saved formulation or restored state differs."""
    timed = bool((getattr(opt, "options", None) or {}).get(
        "display_timing", False))
    rank = int(getattr(opt, "cylinder_rank", 0))
    started = time.perf_counter() if timed else None
    snapshot_s = snapshot_cpu_s = 0.0
    fingerprint_s = fingerprint_cpu_s = 0.0
    nl_timing = _FileWriteTiming() if timed else None
    comparison_s = comparison_cpu_s = 0.0
    if fingerprint_method not in (GRAPH_FINGERPRINT_METHOD,
                                  NL_FINGERPRINT_METHOD):
        raise CheckpointMismatch(
            f"The pyfory-rebuild checkpoint has unsupported model "
            f"fingerprint method {fingerprint_method!r}.")
    try:
        for sname, scenario in opt.local_scenarios.items():
            snapshot = snapshots[sname]
            if fingerprint_method == NL_FINGERPRINT_METHOD:
                expected = (nl_fingerprints or {}).get(sname)
                if not isinstance(expected, str) or not re.fullmatch(
                        r"[0-9a-f]{64}", expected):
                    raise CheckpointMismatch(
                        f"The pyfory-rebuild checkpoint for scenario "
                        f"'{sname}' has no valid NL fingerprint.")
                actual = _rebuild_model_nl_fingerprint(
                    scenario, timing=nl_timing)
            else:
                expected = snapshot.get("model_fingerprint")
                if not isinstance(expected, str) or not re.fullmatch(
                        r"[0-9a-f]{64}", expected):
                    raise CheckpointMismatch(
                        f"The pyfory-rebuild checkpoint for scenario "
                        f"'{sname}' has no valid model fingerprint.")
                fingerprint_version = snapshot.get(
                    "model_fingerprint_version", 1)
                if type(fingerprint_version) is not int or \
                        fingerprint_version not in (1, 2):
                    raise CheckpointMismatch(
                        f"The pyfory-rebuild checkpoint for scenario "
                        f"'{sname}' has unsupported model fingerprint "
                        f"version {fingerprint_version!r}.")
                if timed:
                    phase_started = time.perf_counter()
                    phase_cpu_started = time.thread_time()
                try:
                    if fingerprint_version == 1:
                        actual = _rebuild_model_fingerprint(scenario)
                    else:
                        use_cache = bool((getattr(opt, "options", None) or {}).get(
                            "checkpoint_component_fingerprint_cache", False))
                        caches = getattr(
                            opt, "_pyfory_rebuild_fingerprint_cache", None)
                        if caches is None:
                            caches = {}
                            opt._pyfory_rebuild_fingerprint_cache = caches
                        cache = caches.setdefault(sname, {}) if use_cache else {}
                        dirty_by_scenario = getattr(
                            opt, "_pyfory_rebuild_fingerprint_dirty", {})
                        dirty = dirty_by_scenario.setdefault(sname, set()) \
                            if use_cache else set()
                        if use_cache:
                            # Resume validation always starts from the rebuilt
                            # graph. Seed later writes only with this fresh walk.
                            dirty.add(None)
                        actual = _rebuild_model_fingerprint_with_cache(
                            scenario, cache, dirty)
                finally:
                    if timed:
                        fingerprint_s += time.perf_counter() - phase_started
                        fingerprint_cpu_s += (
                            time.thread_time() - phase_cpu_started)
            if actual != expected:
                if fingerprint_method == NL_FINGERPRINT_METHOD:
                    what = "NL formulation"
                else:
                    what = "pyomo model"
                raise CheckpointMismatch(
                    f"The rebuilt {what} for scenario '{sname}' does not "
                    f"match its checkpoint fingerprint "
                    f"(expected {expected}, got {actual}).")
            saved_state = {
                key: value for key, value in snapshot.items()
                if key not in ("model_fingerprint",
                               "model_fingerprint_version")}
            if timed:
                phase_started = time.perf_counter()
                phase_cpu_started = time.thread_time()
            try:
                current_state = _snapshot_rebuild_model(scenario)
            finally:
                if timed:
                    snapshot_s += time.perf_counter() - phase_started
                    snapshot_cpu_s += (
                        time.thread_time() - phase_cpu_started)
            if timed:
                phase_started = time.perf_counter()
                phase_cpu_started = time.thread_time()
            try:
                state_matches = _checkpoint_state_equal(saved_state,
                                                        current_state)
            finally:
                if timed:
                    comparison_s += time.perf_counter() - phase_started
                    comparison_cpu_s += (
                        time.thread_time() - phase_cpu_started)
            if not state_matches:
                section = next((key for key in sorted(
                    saved_state.keys() | current_state.keys())
                    if key not in saved_state or key not in current_state or
                    not _checkpoint_state_equal(saved_state[key],
                                                current_state[key])), "unknown")
                raise CheckpointMismatch(
                    f"The rebuilt state for scenario '{sname}' does not "
                    f"match its pyfory-rebuild checkpoint snapshot "
                    f"({section}).")
    finally:
        if timed:
            print(
                f"checkpoint_timing event=validate "
                f"backend={PYFORY_REBUILD_BACKEND} rank={rank} "
                f"scenarios={len(opt.local_scenarios)} "
                f"snapshot_s={snapshot_s:.6f} "
                f"snapshot_cpu_s={snapshot_cpu_s:.6f} "
                f"graph_fingerprint_s={fingerprint_s:.6f} "
                f"graph_fingerprint_cpu_s={fingerprint_cpu_s:.6f} "
                f"nl_write_s={nl_timing.nl_write_s if timed else 0.0:.6f} "
                f"nl_write_cpu_s="
                f"{nl_timing.nl_write_cpu_s if timed else 0.0:.6f} "
                f"nl_digest_s={nl_timing.nl_digest_s if timed else 0.0:.6f} "
                f"nl_digest_cpu_s="
                f"{nl_timing.nl_digest_cpu_s if timed else 0.0:.6f} "
                f"comparison_s={comparison_s:.6f} "
                f"comparison_cpu_s={comparison_cpu_s:.6f} "
                f"rank_elapsed_s={time.perf_counter() - started:.6f}",
                flush=True)


def finalize_and_validate_rebuild_model_state(
        opt, snapshots, fingerprint_method=GRAPH_FINGERPRINT_METHOD,
        nl_fingerprints=None):
    """Restore Pyomo's stale flags after setup, then check the rebuilt model."""
    for sname, scenario in opt.local_scenarios.items():
        for name, _, stale, _, _, _ in snapshots[sname]["variables"]:
            var = scenario.find_component(name)
            if var is None:
                raise CheckpointMismatch(
                    f"The rebuilt scenario '{sname}' has no variable '{name}'.")
            var.stale = stale
    validate_rebuild_model_fingerprints(
        opt, snapshots, fingerprint_method=fingerprint_method,
        nl_fingerprints=nl_fingerprints)


def restore_rebuild_solve_state(opt, snapshots):
    """Restore solve results before extensions run their resume callbacks."""
    for sname, scenario in opt.local_scenarios.items():
        snapshot = snapshots[sname]
        if snapshot.get("scenario_name") != str(scenario.name):
            raise CheckpointMismatch(
                f"The pyfory state belongs to scenario "
                f"'{snapshot.get('scenario_name')}', not '{scenario.name}'."
            )
        solve_keys = {"solution_available", "inner_bound", "outer_bound"}
        missing = sorted(solve_keys - snapshot.keys())
        if missing:
            raise CheckpointMismatch(
                f"The pyfory state for scenario '{sname}' has no "
                f"{', '.join(missing)}."
            )
        if not isinstance(snapshot["solution_available"], bool):
            raise CheckpointMismatch(
                f"The pyfory state for scenario '{sname}' has an invalid "
                "solution_available flag."
            )
        for key in ("inner_bound", "outer_bound"):
            bound = _restore_plain_checkpoint_value(snapshot[key])
            if bound is not None and (isinstance(bound, bool)
                                      or not isinstance(
                                          bound, (numbers.Real, Decimal))):
                raise CheckpointMismatch(
                    f"The pyfory state for scenario '{sname}' has an "
                    f"invalid {key}."
                )
        data = scenario._mpisppy_data
        data.solution_available = snapshot["solution_available"]
        data.inner_bound = _restore_plain_checkpoint_value(
            snapshot["inner_bound"])
        data.outer_bound = _restore_plain_checkpoint_value(
            snapshot["outer_bound"])


def _restore_rebuild_model_values(scenario, snapshot):
    """Overlay saved values onto an already rebuilt scenario model."""
    import pyomo.environ as pyo
    from pyomo.environ import Binary, Integers, NonNegativeIntegers, \
        NonNegativeReals, NonPositiveIntegers, NonPositiveReals, Reals, \
        UnitInterval

    if snapshot.get("state_version") != PYFORY_STATE_VERSION:
        raise CheckpointMismatch(
            f"Scenario '{scenario.name}' has pyfory state version "
            f"{snapshot.get('state_version')!r}; this mpi-sppy reads "
            f"version {PYFORY_STATE_VERSION}."
        )
    if snapshot.get("scenario_name") != str(scenario.name):
        raise CheckpointMismatch(
            f"The pyfory state belongs to scenario "
            f"'{snapshot.get('scenario_name')}', not '{scenario.name}'."
        )

    builtin_domains = {
        obj.name: obj for obj in (
            Binary, Integers, NonNegativeIntegers, NonNegativeReals,
            NonPositiveIntegers, NonPositiveReals, Reals, UnitInterval,
        )
    }
    for name in ("Any", "EmptySet", "NegativeIntegers", "NegativeReals",
                 "PositiveIntegers", "PositiveReals", "PercentFraction"):
        domain = getattr(pyo, name, None)
        if domain is not None and getattr(domain, "name", None) == name:
            builtin_domains[name] = domain

    # Restore mutable Params first. Bound expressions that refer to them then
    # regain their checkpoint values without being replaced by constants.
    params = snapshot["mutable_params"]
    packed_params = snapshot["mutable_param_values"]
    if len(packed_params.get("kinds", ())) != len(params):
        raise CheckpointMismatch(
            f"The pyfory checkpoint for scenario '{scenario.name}' has "
            "inconsistent mutable-parameter state."
        )
    for index_in_snapshot, (component_name, index) in enumerate(params):
        value = _unpack_checkpoint_value(packed_params, index_in_snapshot)
        component = scenario.find_component(component_name)
        if component is None:
            raise CheckpointMismatch(
                f"The rebuilt scenario '{scenario.name}' has no mutable "
                f"parameter component '{component_name}' recorded by its "
                "pyfory checkpoint."
            )
        try:
            component[index] = value
        except (KeyError, IndexError, TypeError) as exc:
            raise CheckpointMismatch(
                f"The rebuilt scenario '{scenario.name}' has no parameter "
                f"index {index!r} in '{component_name}'."
            ) from exc

    pending_variables = {}
    variable_records = snapshot["variables"]
    packed_variables = snapshot["variable_values"]
    if len(packed_variables.get("kinds", ())) != len(variable_records):
        raise CheckpointMismatch(
            f"The pyfory checkpoint for scenario '{scenario.name}' has "
            "inconsistent variable state."
        )
    for index_in_snapshot, (name, fixed, stale, lb, ub, domain_name) in \
            enumerate(variable_records):
        value = _unpack_checkpoint_value(packed_variables, index_in_snapshot)
        lb = _restore_plain_checkpoint_value(lb)
        ub = _restore_plain_checkpoint_value(ub)
        var = scenario.find_component(name)
        if var is None:
            # PH's xsqvar is introduced when the deferred PH objective is
            # attached at the end of Iter0. Its value is applied after that.
            if name.startswith("_mpisppy_model.xsqvar["):
                pending_variables[name] = (value, fixed, stale, lb, ub,
                                           domain_name)
                continue
            raise CheckpointMismatch(
                f"The rebuilt scenario '{scenario.name}' has no variable "
                f"'{name}' recorded by its pyfory checkpoint."
            )
        domain = builtin_domains.get(domain_name)
        if domain is None:
            domain = scenario.find_component(domain_name)
        if domain is None:
            raise CheckpointMismatch(
                f"The rebuilt scenario '{scenario.name}' has no variable "
                f"domain '{domain_name}' required by '{name}'."
            )
        var.domain = domain
        current_lb = pyo.value(var.lb, exception=False)
        current_ub = pyo.value(var.ub, exception=False)
        if current_lb != lb:
            var.setlb(lb)
        if current_ub != ub:
            var.setub(ub)
        if fixed:
            var.fix(value)
        else:
            var.unfix()
            var.set_value(value, skip_validation=True)
        var.stale = stale

    return pending_variables


def _replay_prox_approx(opt, scenario, snapshot):
    """Recreate saved proximal cuts and manager bookkeeping."""
    prox = snapshot["prox_approx"]
    if not prox:
        return {}
    from pyomo.core.expr.numeric_expr import LinearExpression
    from mpisppy.utils.sputils import is_persistent

    managers = getattr(scenario._mpisppy_data, "xsqvar_prox_approx", {})
    by_name = {manager.xvar.name: manager
               for manager in managers.values()}
    pending = {}
    for var_name, state in prox.items():
        manager = by_name.get(var_name)
        if manager is None:
            raise CheckpointMismatch(
                f"The rebuilt scenario '{scenario.name}' has no proximal "
                f"manager for '{var_name}'."
            )
        persistent = (scenario._solver_plugin
                      if hasattr(scenario, "_solver_plugin")
                      and is_persistent(scenario._solver_plugin) else None)
        if state["kind"] == "continuous":
            # A saved cut may now lie outside bounds tightened after it was
            # created. add_cut() would clamp it and change the saved model.
            for index, coef_square, coef_x, constant, lower, upper, active \
                    in state["cut_records"]:
                coef_square = _restore_plain_checkpoint_value(coef_square)
                coef_x = _restore_plain_checkpoint_value(coef_x)
                constant = _restore_plain_checkpoint_value(constant)
                lower = _restore_plain_checkpoint_value(lower)
                upper = _restore_plain_checkpoint_value(upper)
                # ProxApproxManagerContinuous creates these cuts with an
                # integer zero. Pyomo's cut.lower presents it as 0.0; use
                # the original expression constant when replaying the cut.
                if lower == 0 and upper is None:
                    lower = 0
                expr = LinearExpression(
                    linear_coefs=[coef_square, coef_x],
                    linear_vars=[manager.xvarsqrd, manager.xvar],
                    constant=constant,
                )
                manager.cuts[manager.var_index, int(index)] = \
                    (lower, expr, upper)
                cut = manager.cuts[manager.var_index, int(index)]
                if not active:
                    cut.deactivate()
                elif persistent is not None:
                    persistent.add_constraint(cut)
            manager.cut_values = array(
                "d", sorted([0.0] + [float(value)
                                   for value in state["cut_values"]]))
            manager.cut_index = int(state["cut_index"])
        elif state["kind"] == "discrete":
            for index in state["cut_indices"]:
                left = int(index) - 1
                slope = 2 * left + 1
                constant = -left * (left + 1)
                expr = LinearExpression(
                    linear_coefs=[1, -slope],
                    linear_vars=[manager.xvarsqrd, manager.xvar],
                    constant=-constant,
                )
                manager.cuts[manager.var_index, int(index)] = \
                    (0, expr, None)
                if persistent is not None:
                    persistent.add_constraint(
                        manager.cuts[manager.var_index, int(index)])
        else:
            raise CheckpointMismatch(
                f"Unknown proximal cut state '{state.get('kind')}'."
            )
    return pending


def restore_rebuild_model_state(opt, snapshots, pending=None):
    """Finish applying model snapshots after deferred PH terms are attached."""
    pending = pending or {}
    for sname, scenario in opt.local_scenarios.items():
        snapshot = snapshots[sname]
        for name, state in pending.get(sname, {}).items():
            var = scenario.find_component(name)
            if var is None:
                raise CheckpointMismatch(
                    f"The rebuilt scenario '{sname}' has no deferred variable "
                    f"'{name}'."
                )
            value, fixed, stale, lb, ub, domain_name = state
            if fixed:
                var.fix(value)
            else:
                var.set_value(value, skip_validation=True)
            var.setlb(lb)
            var.setub(ub)
            var.stale = stale
        _replay_prox_approx(opt, scenario, snapshot)
        if _component_layout(scenario) != snapshot["component_layout"]:
            raise CheckpointMismatch(
                f"The rebuilt scenario '{sname}' does not have the same "
                "component structure as its pyfory checkpoint. An extension "
                "may not have rebuilt model changes made during the saved run."
            )


def gather_extension_model_state(opt):
    """Save model changes owned by extensions that opted into rebuilds."""
    state = {}
    for name, extension in _extension_objects(opt):
        if not getattr(extension, "checkpoint_rebuild_safe", False):
            raise RuntimeError(
                f"Extension '{name}' has not declared support for the "
                "pyfory-rebuild backend. Add a checkpoint model-state "
                "rebuild contract or use dill-reload."
            )
        model_state = extension.checkpoint_model_state()
        # A writer always has a Checkpointer, but a direct PH read-only resume
        # need not attach one. It adds no model structure or model state.
        if name != "Checkpointer":
            state[name] = model_state
    return state


def restore_extension_model_state(opt, state):
    """Let opted-in extensions rebuild model additions before state overlay."""
    require_extension_rebuild_contract(opt)
    if not isinstance(state, dict):
        raise CheckpointMismatch(
            "The pyfory checkpoint has invalid extension model state."
        )
    attached = dict(_extension_objects(opt))
    unexpected = set(state) - set(attached)
    if unexpected:
        raise CheckpointMismatch(
            "The pyfory checkpoint contains model state for unattached "
            f"extension(s): {', '.join(sorted(unexpected))}."
        )
    for name, extension in attached.items():
        if not getattr(extension, "checkpoint_rebuild_safe", False):
            raise CheckpointMismatch(
                f"Extension '{name}' has not declared support for the "
                "pyfory-rebuild backend."
            )
        extension.restore_checkpoint_model_state(state.get(name))


def prepare_rebuild_model_state(opt, snapshots, extension_state):
    """Recreate extension structure and overlay saved values before solvers."""
    restore_extension_model_state(opt, extension_state)
    pending = {}
    for sname, scenario in opt.local_scenarios.items():
        pending[sname] = _restore_rebuild_model_values(
            scenario, snapshots[sname])
    return pending


_SPOKE_FILE_RE = re.compile(r"^spoke_(.+)_ordinal_(\d+)_rank_\d+\.pkl$")


def unclaimed_spoke_files(opt, ckpt_dir, claimed):
    """The (cylinder, ordinal) pairs with a file in the checkpoint that no
    cylinder of this run will read, sorted.

    ``claimed`` is the set of (cylinder class name, ordinal) this run
    carries. A spoke that is dropped on resume leaves its file behind, and
    with it whatever it held -- an incumbent that may be the study's best, or
    a dual cylinder's W -- and the spoke that would have said so is the one
    that is not there. ``opt`` is the hub's, and unused: it is the first
    argument so this reads like every other restore step.
    """
    spokes_dir = os.path.join(ckpt_dir, SPOKES_SUBDIR)
    if not os.path.isdir(spokes_dir):
        return []
    found = set()
    for name in os.listdir(spokes_dir):
        match = _SPOKE_FILE_RE.match(name)
        if match:
            found.add((match.group(1), int(match.group(2))))
    return sorted(found - set(claimed))


def _spoke_filename(cylinder, ordinal, rank):
    """One file per spoke per rank.

    The cylinder class name alone is not unique -- a wheel may carry two
    spokes of the same class configured differently -- so a second number
    disambiguates them. That number is the spoke's *ordinal among cylinders
    of its own class*, not its strata rank.

    The strata rank is the cylinder's index in the whole wheel, and which
    cylinders run is on the list a resume may change (see
    NON_STRUCTURAL_CFG_KEYS: lagrangian, xhatshuffle, fwph and the rest).
    Dropping one renumbers every cylinder after it, so a spoke naming its
    file by strata rank looks for a file that is not there while its own sits
    beside it under the old number -- and, with two spokes of one class,
    lands on the *other* one's file under its own new number and restores an
    incumbent that is not its. The ordinal does not move when an unrelated
    cylinder is added or removed.
    """
    return (f"spoke_{sanitize_for_filename(cylinder)}"
            f"_ordinal_{int(ordinal):02d}_rank_{int(rank):04d}.pkl")


@dataclass
class _FileWriteTiming:
    """Time spent in the parts of a durable file write, summed over files."""

    snapshot_s: float = 0.0
    snapshot_cpu_s: float = 0.0
    graph_fingerprint_s: float = 0.0
    graph_fingerprint_cpu_s: float = 0.0
    nl_write_s: float = 0.0
    nl_write_cpu_s: float = 0.0
    nl_digest_s: float = 0.0
    nl_digest_cpu_s: float = 0.0
    callback_s: float = 0.0
    callback_cpu_s: float = 0.0
    flush_s: float = 0.0
    fsync_s: float = 0.0
    replace_s: float = 0.0
    bytes_written: int = 0


def _atomic_write_bytes(path, write_callback, timing=None):
    """Write via a temp file in the same directory, then rename into place."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        started = time.perf_counter() if timing is not None else None
        cpu_started = time.thread_time() if timing is not None else None
        write_callback(f)
        if timing is not None:
            timing.callback_s += time.perf_counter() - started
            timing.callback_cpu_s += time.thread_time() - cpu_started
            timing.bytes_written += f.tell()
        started = time.perf_counter() if timing is not None else None
        f.flush()
        if timing is not None:
            timing.flush_s += time.perf_counter() - started
        started = time.perf_counter() if timing is not None else None
        os.fsync(f.fileno())
        if timing is not None:
            timing.fsync_s += time.perf_counter() - started
    started = time.perf_counter() if timing is not None else None
    os.replace(tmp, path)
    if timing is not None:
        timing.replace_s += time.perf_counter() - started


def _fsync_dir(path):
    """fsync a directory so the renames recorded in it survive a power loss.

    The per-file fsync in ``_atomic_write_bytes`` makes file *contents*
    durable, but the publish sequence commits by renames, and rename metadata
    lives in the parent directory -- without syncing that too, a power loss
    can leave the manifest naming a generation whose directory did not
    survive. Kill-safety never depends on this, only power-loss safety does,
    so platforms where a directory cannot be opened or synced (e.g. Windows)
    skip it silently.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


###############################################################################
# Multi-rank coordination.
#
# A checkpoint generation spans every rank of the cylinder: each rank holds a
# different slice of the scenarios, so a resumable generation is the *set* of
# per-rank files, and the manifest flip that publishes it must not happen until
# every one of them is on disk. Three things follow, and each is a rule the
# single-rank code did not need.
#
# * **The directory work is rank 0's alone.** Every rank computes the same
#   staging and generation paths, so letting each one create, rename and delete
#   them means ranks destroying each other's files. Rank 0 prepares the staging
#   directory and performs the whole publish; the others only write their own
#   rank-tagged files into it.
# * **Barriers bracket the shared directory.** One before the writes, so no
#   rank writes into a directory rank 0 is about to clear, and one after them
#   (the failure agreement below doubles as it), so rank 0 does not publish a
#   generation that is still missing files.
# * **Failure is agreed on, not discovered.** A mid-run write failure warns and
#   lets the run continue (section 8), which on one rank is a return and on
#   several is a deadlock: the rank that failed skips the barrier the others
#   are waiting at. So every rank reports whether its own write succeeded, all
#   ranks learn the answer together, and either all of them publish or none
#   does. The generation is therefore all-or-nothing, which is what makes the
#   manifest's promise -- that it names a *complete* checkpoint -- true across
#   ranks and not just within one.
#
# * **A refusal is agreed on too.** Setting a checkpoint up and restoring one
#   are per-rank work throughout -- this rank's scenario names, this rank's
#   node, the file named after this rank -- so every refusal on those paths is
#   one a single rank can make while the others walk on into the next
#   collective. ``run_agreed`` runs each such step on every rank and raises on
#   all of them or on none, which is what turns a rank-local refusal into an
#   error message. Anything added to those paths belongs inside it;
#   ``test_checkpoint_multirank.py`` checks that nothing sits beside it.
#
# The iteration-count triggers need no such agreement. They are a pure function
# of the absolute iteration number and the iteration limit (see
# ``Checkpointer._should_write``), both of which are identical on every rank of
# a synchronous PH cylinder, so the ranks arrive at the barrier together
# without being asked. A trigger that is not a pure function of the iteration
# count would reintroduce rank skew and deadlock here, so the elapsed-time one,
# ``--checkpoint-before-seconds``, is put through ``allreduce_or`` first.
###############################################################################

#: Sentinel meaning "no rank failed" in the failure agreement below. Larger
#: than any rank, so MIN over the ranks picks a real failure whenever there is
#: one and this value only when there is none.
_NO_FAILURE = np.iinfo(np.int32).max


def _cylinder_comm(opt):
    """The comm to coordinate a checkpoint over, or None when there is one rank.

    This is the *cylinder's* comm, not COMM_WORLD: a hub and its spokes
    checkpoint independently and must never wait on each other (section 9,
    item 6). Returning None for a single-rank cylinder keeps the serial path
    free of MPI calls entirely, so nothing here depends on an MPI installation
    being present.
    """
    if int(getattr(opt, "n_proc", 1)) <= 1:
        return None
    return opt.mpicomm


def _barrier(comm):
    if comm is not None:
        comm.Barrier()


def _first_failing_rank(comm, rank, failed):
    """Agree across the cylinder on whether -- and where -- a write failed.

    Returns the lowest rank that failed, or None if none did. Collective, so
    it is also the barrier that guarantees every rank has finished writing
    before rank 0 publishes.
    """
    if comm is None:
        return rank if failed else None
    local = np.array([rank if failed else _NO_FAILURE], dtype=np.int32)
    worst = np.zeros(1, dtype=np.int32)
    comm.Allreduce(local, worst, op=MPI.MIN)
    return None if int(worst[0]) == _NO_FAILURE else int(worst[0])


def agree_one_write(opt, state, key):
    """Do the ranks of this cylinder hold one checkpoint, written at once?

    Each rank of a multi-rank cylinder reads its own file, because each owns
    different scenarios -- and the files need not agree, because each rank
    writes at the bottom of its own iteration or pass. A stop lands between
    two of those writes, a rank whose write failed warned and carried on, and
    a rank that had nothing to write wrote nothing.

    Returns ``(verdict, detail)``:

    * ``"none"``    -- no rank has a file.
    * ``"partial"`` -- some ranks have one and some do not; detail is
      ``(how many, out of how many)``.
    * ``"differ"``  -- every rank has one, but ``key`` does not match across
      them, so they are from different writes; detail is the values.
    * ``"agreed"``  -- every rank has one, all from the same write; detail is
      the agreed value.

    ``key`` identifies the write: a field name or a function of the state
    (the write epoch and iteration for a dual cylinder, or
    ``spoke_write_id`` for an xhat spoke).

    Collective. Every rank of the cylinder must call it, including the ranks
    whose ``state`` is None.
    """
    comm = _cylinder_comm(opt)
    if comm is None:
        return ("agreed" if state is not None else "none"), None
    size = comm.Get_size()
    have = comm.allreduce(1 if state is not None else 0, op=MPI.SUM)
    if have < size:
        return _one_write_verdict(have, size, None)
    # Every rank reads these out of the same collective -- an Eobjective
    # reduction for the objective, the shared iteration counter for the
    # generation -- so ranks from one write hold the identical value and an
    # exact comparison is the right one. Gathered rather than reduced so a
    # warning can name what disagrees, which is what a user needs to see.
    values = comm.allgather(key(state) if callable(key) else state.get(key))
    return _one_write_verdict(have, size, values)


def spoke_write_id(objective, progress):
    """An identifier for one write of an xhat spoke's file, the same on every
    rank that made that write.

    The objective alone no longer identifies a write: a spoke also writes when
    its loop cursor moves with the incumbent unchanged, so two files with the
    same objective can be from different moments -- and each rank restores its
    own extension state from its own file. Both inputs are collective (the
    objective an Eobjective reduction, the cursor the same on every rank of
    the loop), so ranks that wrote together compute the same identifier.
    """
    payload = json.dumps(_canonical([objective, progress]), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def XHAT_WRITE_KEY(state):
    """What identifies an xhat spoke's write in its file: the ``write_id``
    recorded at the write, or the objective for a file written before there
    was one."""
    return state.get("write_id", state.get("best_solution_obj_val"))


def _one_write_verdict(have, size, values):
    """``agree_one_write``'s verdict from what it gathered: ``have`` of
    ``size`` ranks hold a file, and ``values`` are their ``key`` values
    (read only when every rank has a file).

    Separate from the collective so the resumed hub, which reads a spoke's
    files for its inner bound (``spoke_inner_bounds_to_restore``), decides by
    the same rule as the spoke's ranks and never credits an incumbent they
    drop.
    """
    if have == 0:
        return "none", None
    if have < size:
        return "partial", (have, size)
    if len(set(values)) != 1:
        return "differ", values
    return "agreed", values[0]


def run_agreed(opt, work, what):
    """Run one local step of checkpoint setup or restore, and agree on it.

    Every step of setting a checkpoint up and of restoring one is inherently
    per rank: it reads a file named after this rank, checks it against the
    scenarios this rank owns, or writes from the node this rank is on. So
    each of them can fail on one rank and succeed on the others -- and the
    step after it is collective. A rank that raised on its own has left the
    others waiting at a collective it will never reach, and neither way that
    ends is the one wanted: ``python -m mpi4py`` aborts the whole job holding
    that one rank's traceback, and a plain ``python`` launch leaves it sitting
    in the collective until its wall-clock limit with nothing in the log.

    So agree first. ``work`` is called on every rank, the ranks exchange
    whether it raised, and either all of them raise or none does. The message
    names how many ranks failed and what each of them said, which is the
    diagnosis a rank-local raise loses -- it aborts the job holding one
    rank's traceback while the collective's own explanation never runs -- and
    the raise on a rank that did fail chains its own exception underneath.

    Anything ``work`` raises is reported, not only ``CheckpointMismatch``: an
    unreadable file, a full disk or a short read strand the other ranks in
    exactly the same way. The type is kept in the message.

    ``what`` completes the sentence "N of M ranks of this cylinder could
    not ...", so it reads as a verb phrase and says what follows from the
    failure ("read their checkpoint, so none of them uses one").

    Returns whatever ``work`` returns. Collective: every rank of the cylinder
    must call it, and none of them may skip it on a condition the other ranks
    do not share.
    """
    try:
        result, failure = work(), None
    except Exception as exc:
        result, failure = None, exc
    comm = _cylinder_comm(opt)
    if comm is None:
        if failure is not None:
            # One rank, so there is nobody to agree with and nothing to add:
            # the caller's own exception, with its own type and traceback.
            raise failure
        return result
    failures = comm.allgather(
        None if failure is None else f"{type(failure).__name__}: {failure}")
    by_message = {}
    for rank, message in enumerate(failures):
        if message is not None:
            by_message.setdefault(message, []).append(rank)
    if not by_message:
        return result
    detail = "; ".join(
        f"{len(ranks)} rank(s), lowest {min(ranks)}: {message}"
        for message, ranks in by_message.items())
    raise CheckpointMismatch(
        f"{sum(len(r) for r in by_message.values())} of {comm.Get_size()} "
        f"ranks of this cylinder could not {what}. {detail}"
    ) from failure

def require_implemented_backend(backend):
    """Refuse a backend that is only designed, or not known at all.

    Shared by the write side (Checkpointer) and the read side (the resume
    branch in PHBase), because a resume need not have a Checkpointer
    attached.
    """
    if backend not in (DILL_RELOAD_BACKEND, PYFORY_REBUILD_BACKEND,
                       PYFORY_RELOAD_BACKEND):
        raise RuntimeError(
            f"--checkpoint-backend '{backend}' is not implemented. "
            f"Supported backends are '{DILL_RELOAD_BACKEND}', "
            f"'{PYFORY_REBUILD_BACKEND}', and '{PYFORY_RELOAD_BACKEND}'."
        )

def agree_spoke_restore(opt, state):
    """Agree across an xhat spoke's ranks on the parts of a restore that are
    the cylinder's rather than a rank's.

    Two things in the file describe the cylinder: the loop cursor and the
    inner bound. The cursor is the expensive one to get wrong. The
    xhatshuffle loop is collective -- every rank picks the same scenario and
    ``_try_one`` broadcasts its nonants from the rank that owns it -- so
    ranks resuming from different cursors pick different scenarios, the
    broadcast has a different root on each rank, and the objective the hub is
    handed blends several scenarios' solutions instead of reporting any one
    of them. It arrives as an ordinary feasible inner bound: no error, no
    warning, exit 0.

    The cached solution values stay rank-local, because each rank owns
    different scenarios and there is nothing to broadcast. What has to be
    agreed about them is that they all came from the same pass.

    Returns ``(state, warning)``: rank 0's cursor and bound written into this
    rank's own state, or ``(None, message)`` where the files do not describe
    one incumbent.

    Collective; see :func:`agree_one_write`.
    """
    verdict, detail = agree_one_write(opt, state, XHAT_WRITE_KEY)
    if verdict == "none":
        return None, None
    if verdict == "partial":
        have, size = detail
        return None, (
            f"only {have} of {size} ranks of this spoke have a "
            f"checkpointed incumbent, so none of them restores one: an "
            f"incumbent assembled from some ranks and not others is not a "
            f"solution this study ever found")
    if verdict == "differ":
        return None, (
            f"the ranks of this spoke checkpointed different incumbents "
            f"({len(set(detail))} different writes across {len(detail)} "
            f"ranks), so none of them restores one: the files were written "
            f"at different passes, and half of one xhat beside half of "
            f"another is not a solution this study ever found")
    comm = _cylinder_comm(opt)
    if comm is not None:
        # Rank 0's, on every rank. Which rank is arbitrary -- what matters is
        # that they stop differing -- so it is the one every other agreement
        # here already uses.
        loop_state, best_inner_bound = comm.bcast(
            (state.get("loop_state"), state.get("best_inner_bound")), root=0)
        state["loop_state"] = loop_state
        state["best_inner_bound"] = best_inner_bound
    return state, None


def agree_dual_spoke_restore(opt, state):
    """The same agreement for a dual cylinder's W, keyed on its write run.

    Nothing here is broadcast: W is per scenario and so per rank. What must
    hold is that every rank restores the W of the *same* iteration of the
    same process invocation. Ranks that restore W from different writes hand
    ``Compute_Xbar`` an allreduce over values belonging to different
    points of the run --
    and under ``--ph-primal-hub`` that blended W is what the hub's W is
    built from. No error and no warning, like the xhat case.

    Returns ``(state, warning)``. Collective; see :func:`agree_one_write`.
    """
    verdict, detail = agree_one_write(
        opt, state, lambda saved: (saved.get("write_epoch"),
                                   saved["generation"]))
    if verdict == "none":
        return None, None
    if verdict == "partial":
        have, size = detail
        return None, (
            f"only {have} of {size} ranks of this cylinder have checkpointed "
            f"dual weights, so none of them restores any: a W taken from "
            f"some ranks and not others is not the W this study had")
    if verdict == "differ":
        if any(not epoch for epoch, _ in detail):
            return None, (
                "some ranks of this cylinder have checkpointed dual weights "
                "without a write identity, so none of them restores any: "
                "their files cannot be proven to belong to one write")
        return None, (
            f"the ranks of this cylinder checkpointed dual weights from "
            f"different iterations or runs ({detail}), so none of them "
            f"restores any: their W cannot be combined safely")
    if not state.get("write_epoch"):
        return None, (
            "this cylinder's checkpointed dual weights have no write "
            "identity, so none of them restores any: their files cannot "
            "be proven to belong to one write")
    return state, None


def require_dill(backend):
    """Backward-compatible name for the backend dependency check."""
    if backend == DILL_RELOAD_BACKEND:
        require_backend_dependency(backend)


def require_extension_rebuild_contract(opt):
    """Refuse extensions that cannot recreate their changes on fresh models."""
    from mpisppy.extensions.extension import Extension

    for name, extension in _extension_objects(opt):
        if not getattr(extension, "checkpoint_rebuild_safe", False):
            raise RuntimeError(
                f"Extension '{name}' has not declared support for the "
                "pyfory-rebuild backend. Add a checkpoint model-state "
                "rebuild contract or use dill-reload."
            )
        cls = type(extension)
        if (cls.checkpoint_model_state is Extension.checkpoint_model_state
                or cls.restore_checkpoint_model_state
                is Extension.restore_checkpoint_model_state):
            raise RuntimeError(
                f"Extension '{name}' declares pyfory-rebuild support but "
                "does not implement both checkpoint model-state hooks."
            )




def probe_directory_is_writable(opt, ckpt_dir):
    """Create the checkpoint directory and prove this rank can write in it.

    Called once at setup. Discovering only at write time that the path is
    unwritable would mean the run never checkpoints. Every rank probes,
    because on a cluster the checkpoint directory can be unwritable from some
    nodes and not others, and that is exactly the failure worth catching
    before a multi-hour run rather than at its first write.

    Per rank, and so a step the caller has to agree on: see
    :func:`run_agreed`.

    The probe file is named for the *global* rank, not the cylinder rank.
    Several cylinders write into one directory and each of them numbers its
    own ranks from zero, so a cylinder-rank name has one copy per cylinder:
    whichever probes second removes the file the first is still using, and
    that one fails with a FileNotFoundError naming a directory that is
    perfectly writable. Global ranks are unique across the job, which is the
    scope this needs.
    """
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        probe = os.path.join(
            ckpt_dir,
            f".mpisppy_write_probe_"
            f"{int(getattr(opt, 'global_rank', opt.cylinder_rank)):04d}")
        with open(probe, "w"):
            pass
        os.remove(probe)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot write to the checkpoint directory '{ckpt_dir}' from "
            f"rank {opt.cylinder_rank} ({type(exc).__name__}: {exc})."
        ) from exc


def probe_model_is_dillable(opt):
    """Serialize every local scenario to memory to prove checkpointing works.

    Called once at setup. A run that only discovers at its first checkpoint --
    possibly many hours in -- that its models cannot be dilled would lose
    exactly the state checkpointing exists to preserve, so this trades the
    serializations up front for a failure that arrives immediately and says
    what to do about it. The probe runs at iteration 0. On a fresh run that is
    when the models are at their smallest, with no accumulated
    prox-approximation cuts; on a resume it runs after the checkpointed models
    have been spliced in, so it dills whatever they had accumulated by the
    stop and costs correspondingly more.

    Every scenario, not just the first: what makes a model undillable is
    usually something its ``scenario_creator`` closed over, and a creator that
    closes over something unserializable for *one* scenario -- a solver
    handle, a file object, a rule that reads a scenario-specific object -- is
    exactly the case a one-scenario probe waves through. The run would then
    fail at every write, survive each failure by design, and finish having
    published nothing at all, which is the outcome the setup-time refusal
    exists to rule out.

    Collective, for the same reason the write is: the ranks own different
    scenarios, so an undillable model is usually rank-local. A rank that
    raised on its own would leave the others to go on and hang at the first
    write barrier, turning a clear setup refusal into a job that stalls with
    no message. Every rank therefore learns that some rank failed and raises,
    naming the one that has the real diagnosis.
    """
    timed = bool(opt.options.get("display_timing", False))
    probe_started = time.perf_counter() if timed else None
    serialize_s = 0.0
    serialize_cpu_s = 0.0
    failure = None
    for sname, s in opt.local_scenarios.items():
        solver_plugin = getattr(s, "_solver_plugin", None)
        if solver_plugin is not None:
            del s._solver_plugin
        try:
            started = time.perf_counter() if timed else None
            cpu_started = time.thread_time() if timed else None
            dill.dumps(s)
            if timed:
                serialize_s += time.perf_counter() - started
                serialize_cpu_s += time.thread_time() - cpu_started
        except Exception as exc:
            failure = RuntimeError(
                "Checkpointing is enabled, but no checkpoint could ever be "
                "written.\n\n"
                + pickle_bundle.describe_dill_failure(
                    s, exc, what=f"scenario '{sname}'")
            )
            failure.__cause__ = exc
            break
        finally:
            if solver_plugin is not None:
                s._solver_plugin = solver_plugin

    comm = _cylinder_comm(opt)
    agreement_started = time.perf_counter() if timed else None
    failing_rank = _first_failing_rank(comm, int(opt.cylinder_rank),
                                       failure is not None)
    if timed:
        print(
            f"checkpoint_timing event=probe backend={DILL_RELOAD_BACKEND} "
            f"rank={int(opt.cylinder_rank)} "
            f"scenarios={len(opt.local_scenarios)} "
            f"serialize_s={serialize_s:.6f} "
            f"serialize_cpu_s={serialize_cpu_s:.6f} "
            f"agreement_s={time.perf_counter() - agreement_started:.6f} "
            f"rank_elapsed_s={time.perf_counter() - probe_started:.6f}",
            flush=True)
    if failing_rank is None:
        return
    if failure is not None:
        raise failure
    raise RuntimeError(
        f"Checkpointing is enabled, but no checkpoint could ever be written: "
        f"rank {failing_rank} has a scenario model that cannot be "
        f"serialized. See that rank's message for which one and why."
    )


def _clone_rebuild_probe_model(scenario):
    """Clone a configured model for the rebuild probe without losing Sets.

    Pyomo cannot deepcopy a Set initialized from a dict_keys view: clone()
    logs an error and silently drops that Set's initializer. Supply a
    snapshot of the view through the deepcopy memo for this probe clone only.
    The live scenario and the checkpoint format remain unchanged.
    """
    from pyomo.environ import Set

    memo = {}
    for component in scenario.component_objects(Set, active=None,
                                                descend_into=True):
        initializer = getattr(getattr(component, "_init_values", None),
                              "_init", None)
        values = getattr(initializer, "val", None)
        if isinstance(values, KeysView):
            memo[id(values)] = tuple(values)
    return scenario.clone(memo=memo)


def probe_model_checkpointable(opt, backend=DILL_RELOAD_BACKEND):
    """Probe each local model codec before the run starts writing.

    For pyfory-rebuild, the clone check proves that the core snapshot can be
    decoded and overlaid on an equivalent Pyomo model. It does not prove that
    future extension behavior is captured; rebuild hooks are checked for
    encodable state here and their component layout is checked on resume.
    """
    require_backend_dependency(backend)
    if backend == DILL_RELOAD_BACKEND:
        # Keep dill's serialize-only probe exactly as it has always worked.
        return probe_model_is_dillable(opt)
    if backend == PYFORY_RELOAD_BACKEND:
        return _probe_pyfory_reload_models(opt)
    if backend != PYFORY_REBUILD_BACKEND:
        raise ValueError(f"Unknown checkpoint backend '{backend}'.")

    require_extension_rebuild_contract(opt)
    extension_state = gather_extension_model_state(opt)
    timed = bool(opt.options.get("display_timing", False))
    use_nl_fingerprint = bool(opt.options.get(
        "checkpoint_nl_fingerprint", False))
    nl_timing = _FileWriteTiming() if timed else None
    probe_started = time.perf_counter() if timed else None
    encode_s = decode_s = encode_cpu_s = decode_cpu_s = 0.0
    snapshot_s = snapshot_cpu_s = 0.0
    fingerprint_s = fingerprint_cpu_s = 0.0
    clone_s = clone_cpu_s = 0.0
    overlay_s = overlay_cpu_s = 0.0
    failure = None
    for extension_name, model_state in extension_state.items():
        try:
            hook_payload = {extension_name: model_state}
            started = time.perf_counter() if timed else None
            cpu_started = time.thread_time() if timed else None
            hook_bytes = _new_fory().serialize(hook_payload)
            if timed:
                encode_s += time.perf_counter() - started
                encode_cpu_s += time.thread_time() - cpu_started
            started = time.perf_counter() if timed else None
            cpu_started = time.thread_time() if timed else None
            hook_restored = _new_fory().deserialize(hook_bytes)
            if timed:
                decode_s += time.perf_counter() - started
                decode_cpu_s += time.thread_time() - cpu_started
            if not _checkpoint_state_equal(hook_restored, hook_payload):
                raise RuntimeError(
                    f"pyfory changed model state during a fresh-reader "
                    "round trip")
        except Exception as exc:
            failure = RuntimeError(
                "Checkpointing is enabled, but pyfory-rebuild cannot encode "
                f"model state from extension '{extension_name}': "
                f"{type(exc).__name__}: {exc}")
            failure.__cause__ = exc
            break

    for sname, scenario in opt.local_scenarios.items():
        if failure is not None:
            break
        try:
            started = time.perf_counter() if timed else None
            cpu_started = time.thread_time() if timed else None
            try:
                state = _snapshot_rebuild_model(scenario)
            finally:
                if timed:
                    snapshot_s += time.perf_counter() - started
                    snapshot_cpu_s += time.thread_time() - cpu_started
            started = time.perf_counter() if timed else None
            cpu_started = time.thread_time() if timed else None
            payload = _new_fory().serialize(state)
            if timed:
                encode_s += time.perf_counter() - started
                encode_cpu_s += time.thread_time() - cpu_started
            started = time.perf_counter() if timed else None
            cpu_started = time.thread_time() if timed else None
            restored = _new_fory().deserialize(payload)
            if timed:
                decode_s += time.perf_counter() - started
                decode_cpu_s += time.thread_time() - cpu_started
            if not _checkpoint_state_equal(restored, state):
                raise RuntimeError(
                    "pyfory changed the plain scenario state during a "
                    "fresh-reader round trip")

            # Pyomo's clone is the closest safe startup check for the overlay:
            # extension hooks may add structure, so the clone begins with the
            # same already-configured structure as the live scenario.
            started = time.perf_counter() if timed else None
            cpu_started = time.thread_time() if timed else None
            try:
                clone = _clone_rebuild_probe_model(scenario)
            finally:
                if timed:
                    clone_s += time.perf_counter() - started
                    clone_cpu_s += time.thread_time() - cpu_started
            started = time.perf_counter() if timed else None
            cpu_started = time.thread_time() if timed else None
            try:
                _restore_rebuild_model_values(clone, restored)
            finally:
                if timed:
                    overlay_s += time.perf_counter() - started
                    overlay_cpu_s += time.thread_time() - cpu_started
            started = time.perf_counter() if timed else None
            cpu_started = time.thread_time() if timed else None
            try:
                clone_state = _snapshot_rebuild_model(clone)
            finally:
                if timed:
                    snapshot_s += time.perf_counter() - started
                    snapshot_cpu_s += time.thread_time() - cpu_started
            if not _checkpoint_state_equal(clone_state, restored):
                raise RuntimeError(
                    "the decoded snapshot could not be overlaid on a clone "
                    "of the configured scenario")
            if use_nl_fingerprint:
                original_fingerprint = _rebuild_model_nl_fingerprint(
                    scenario, timing=nl_timing)
                clone_fingerprint = _rebuild_model_nl_fingerprint(
                    clone, timing=nl_timing)
            else:
                started = time.perf_counter() if timed else None
                cpu_started = time.thread_time() if timed else None
                try:
                    original_fingerprint = _rebuild_model_fingerprint(
                        scenario)
                finally:
                    if timed:
                        fingerprint_s += time.perf_counter() - started
                        fingerprint_cpu_s += (
                            time.thread_time() - cpu_started)
                started = time.perf_counter() if timed else None
                cpu_started = time.thread_time() if timed else None
                try:
                    clone_fingerprint = _rebuild_model_fingerprint(clone)
                finally:
                    if timed:
                        fingerprint_s += time.perf_counter() - started
                        fingerprint_cpu_s += (
                            time.thread_time() - cpu_started)
            if clone_fingerprint != original_fingerprint:
                what = "NL" if use_nl_fingerprint else "pyomo model"
                raise RuntimeError(
                    f"the decoded snapshot changed the {what} fingerprint "
                    "when overlaid on a clone")
        except Exception as exc:
            failure_reason = ("write its NL fingerprint for" if
                              use_nl_fingerprint else "encode")
            failure = RuntimeError(
                "Checkpointing is enabled, but pyfory-rebuild cannot "
                f"{failure_reason} "
                f"scenario '{sname}': {type(exc).__name__}: {exc}"
            )
            failure.__cause__ = exc
            break

    comm = _cylinder_comm(opt)
    agreement_started = time.perf_counter() if timed else None
    failing_rank = _first_failing_rank(comm, int(opt.cylinder_rank),
                                       failure is not None)
    if timed:
        print(
            f"checkpoint_timing event=probe backend={backend} "
            f"rank={int(opt.cylinder_rank)} "
            f"scenarios={len(opt.local_scenarios)} "
            f"serialize_s={encode_s:.6f} "
            f"serialize_cpu_s={encode_cpu_s:.6f} "
            f"deserialize_s={decode_s:.6f} "
            f"deserialize_cpu_s={decode_cpu_s:.6f} "
            f"snapshot_s={snapshot_s:.6f} "
            f"snapshot_cpu_s={snapshot_cpu_s:.6f} "
            f"graph_fingerprint_s={fingerprint_s:.6f} "
            f"graph_fingerprint_cpu_s={fingerprint_cpu_s:.6f} "
            f"nl_write_s={nl_timing.nl_write_s if timed else 0.0:.6f} "
            f"nl_write_cpu_s="
            f"{nl_timing.nl_write_cpu_s if timed else 0.0:.6f} "
            f"nl_digest_s={nl_timing.nl_digest_s if timed else 0.0:.6f} "
            f"nl_digest_cpu_s="
            f"{nl_timing.nl_digest_cpu_s if timed else 0.0:.6f} "
            f"clone_s={clone_s:.6f} clone_cpu_s={clone_cpu_s:.6f} "
            f"overlay_s={overlay_s:.6f} overlay_cpu_s={overlay_cpu_s:.6f} "
            f"agreement_s={time.perf_counter() - agreement_started:.6f} "
            f"rank_elapsed_s={time.perf_counter() - probe_started:.6f}",
            flush=True)
    if failing_rank is None:
        return
    if failure is not None:
        raise failure
    raise RuntimeError(
        "Checkpointing is enabled, but pyfory-rebuild cannot encode a "
        f"scenario on rank {failing_rank}. See that rank's message."
    )


def _validate_reloaded_model(model, expected, sname):
    """Check the graph relationships required before splicing a model."""
    if not hasattr(model, "component_data_objects") or \
            getattr(model, "name", None) != expected.name:
        raise CheckpointMismatch(
            f"The reloaded model for scenario '{sname}' has the wrong type "
            "or name.")
    data = getattr(model, "_mpisppy_data", None)
    fresh_data = getattr(expected, "_mpisppy_data", None)
    nodes = getattr(model, "_mpisppy_node_list", None)
    fresh_nodes = getattr(expected, "_mpisppy_node_list", None)
    if data is None or fresh_data is None or nodes is None or \
            fresh_nodes is None or len(nodes) != len(fresh_nodes):
        raise CheckpointMismatch(
            f"The reloaded model for scenario '{sname}' has no valid "
            "nonanticipative nodes.")
    if tuple(data.nonant_indices) != tuple(fresh_data.nonant_indices):
        raise CheckpointMismatch(
            f"The reloaded model for scenario '{sname}' has different "
            "nonanticipative indices.")
    for node, fresh_node in zip(nodes, fresh_nodes):
        if node.name != fresh_node.name or len(node.nonant_vardata_list) != \
                len(fresh_node.nonant_vardata_list):
            raise CheckpointMismatch(
                f"The reloaded model for scenario '{sname}' has different "
                "nonanticipative node structure.")
        for var in node.nonant_vardata_list:
            if var.model() is not model or model.find_component(var.name) is not var:
                raise CheckpointMismatch(
                    f"The reloaded model for scenario '{sname}' has a "
                    "nonanticipative variable with broken ownership.")
    for var in data.nonant_indices.values():
        if var.model() is not model:
            raise CheckpointMismatch(
                f"The reloaded model for scenario '{sname}' has a "
                "nonanticipative index pointing outside the model.")
    managers = getattr(data, "xsqvar_prox_approx", {})
    if managers is False or managers is None:
        managers = {}
    if not isinstance(managers, dict):
        raise CheckpointMismatch(
            f"The reloaded model for scenario '{sname}' has invalid "
            "proximal approximation state.")
    for key, manager in managers.items():
        if manager.xvar.model() is not model or \
                manager.W is not model._mpisppy_model.W[key] or \
                manager.rho is not model._mpisppy_model.rho[key] or \
                manager.cuts is not model._mpisppy_model.xsqvar_cuts:
            raise CheckpointMismatch(
                f"The reloaded model for scenario '{sname}' has broken "
                "proximal approximation references.")


def _probe_pyfory_reload_models(opt):
    """Round-trip each local scenario with a reader built from its file."""
    from mpisppy.utils import pyfory_model_codec

    timed = bool(opt.options.get("display_timing", False))
    probe_started = time.perf_counter() if timed else None
    encode_s = decode_s = encode_cpu_s = decode_cpu_s = 0.0
    failure = None
    for sname, scenario in opt.local_scenarios.items():
        solver_plugin = getattr(scenario, "_solver_plugin", None)
        if solver_plugin is not None:
            del scenario._solver_plugin
        try:
            stream = io.BytesIO()
            started = time.perf_counter() if timed else None
            cpu_started = time.thread_time() if timed else None
            pyfory_model_codec.dump(scenario, stream, scenario_name=sname)
            if timed:
                encode_s += time.perf_counter() - started
                encode_cpu_s += time.thread_time() - cpu_started
            started = time.perf_counter() if timed else None
            cpu_started = time.thread_time() if timed else None
            restored = pyfory_model_codec.load(io.BytesIO(stream.getvalue()),
                                               scenario_name=sname)
            _validate_reloaded_model(restored, scenario, sname)
            if timed:
                decode_s += time.perf_counter() - started
                decode_cpu_s += time.thread_time() - cpu_started
        except Exception as exc:
            failure = RuntimeError(
                "Checkpointing is enabled, but pyfory-reload cannot round "
                f"trip scenario '{sname}': {type(exc).__name__}: {exc}")
            failure.__cause__ = exc
            break
        finally:
            if solver_plugin is not None:
                scenario._solver_plugin = solver_plugin

    comm = _cylinder_comm(opt)
    agreement_started = time.perf_counter() if timed else None
    failing_rank = _first_failing_rank(comm, int(opt.cylinder_rank),
                                       failure is not None)
    if timed:
        print(
            f"checkpoint_timing event=probe backend={PYFORY_RELOAD_BACKEND} "
            f"rank={int(opt.cylinder_rank)} "
            f"scenarios={len(opt.local_scenarios)} "
            f"serialize_s={encode_s:.6f} "
            f"serialize_cpu_s={encode_cpu_s:.6f} "
            f"deserialize_s={decode_s:.6f} "
            f"deserialize_cpu_s={decode_cpu_s:.6f} "
            f"agreement_s={time.perf_counter() - agreement_started:.6f} "
            f"rank_elapsed_s={time.perf_counter() - probe_started:.6f}",
            flush=True)
    if failure is not None:
        raise failure
    if failing_rank is not None:
        raise RuntimeError(
            "Checkpointing is enabled, but pyfory-reload cannot round trip "
            f"a scenario on rank {failing_rank}. See that rank's message.")


def geometry(opt):
    """The rank layout a resume must reproduce (see design section 5.7)."""
    return {
        "n_proc": int(opt.n_proc),
        "rank": int(opt.cylinder_rank),
        "scenario_names": sorted(opt.local_scenarios.keys()),
    }


def initially_fixed_nonant_names(opt):
    """Per-scenario names of the nonants already fixed when the run started.

    This is the baseline ``_can_update_best_bound`` compares against, and it is
    the one piece of opt-object state that is keyed by variable *identity* --
    a ``ComponentSet`` of vardata belonging to models that a resume replaces.
    Recording it by name is what lets the resume rebuild it correctly; see
    design section 9, item 11, for what goes wrong otherwise.

    The names are keyed by scenario because Pyomo component names are not
    scenario-qualified -- every scenario's ``x[3]`` is named ``x[3]`` -- so a
    flat name list would smear a nonant fixed in one scenario onto all of
    them, and a resumed run would admit best-bound updates the uninterrupted
    run refuses.
    """
    baseline = getattr(opt, "_initial_fixed_varibles", None)
    if baseline is None:
        return {}
    return {
        sname: sorted(v.name
                      for v in s._mpisppy_data.nonant_indices.values()
                      if v in baseline)
        for sname, s in opt.local_scenarios.items()
    }


###############################################################################
# Extension and converger object state (design section 5.5, section 9 item 3).
#
# The dilled models bring back everything that lives *on a model*. What they
# cannot bring back is state an extension keeps on itself, and several
# extensions keep exactly the state that decides what they do next: the
# previous xbar a rho updater compares against, how many iterations a fixer has
# watched a variable hold still, which nonants a slammer has already pinned. An
# extension that starts fresh on a resumed run takes a different action at the
# next iteration than the uninterrupted run would have, so the runs diverge --
# quietly, since nothing is missing and nothing raises.
#
# The contract is two methods on ``Extension`` and ``Converger``, which the
# base classes leave raising: a checkpointed run refuses at startup any class
# that does not override both (require_state_contract). This module
# aggregates them, keyed by class name, because names are what survives
# a resume: object identity does not, and attach order is not stable enough to
# index by. Name keying is also what lets a resume with a *different* extension
# set do something sensible -- entries with no matching extension are reported
# rather than silently dropped.
###############################################################################


def _extension_objects(opt):
    """Yield ``(class name, extension)`` for every attached extension.

    ``MultiExtension`` is a container, not an extension with state of its own,
    so it is flattened away -- and flattened recursively, since nothing stops
    one from holding another. An empty one yields nothing: it is still a
    container, not an extension that failed to implement the hooks. Two extensions of the same class would collide
    on the name key, but ``MultiExtension.extdict`` is itself keyed by class
    name, so they cannot both be attached in the first place.
    """
    ext = getattr(opt, "extobject", None)
    if ext is None:
        return
    stack = [ext]
    while stack:
        obj = stack.pop()
        extdict = getattr(obj, "extdict", None)
        if extdict is not None:
            stack.extend(extdict.values())
        else:
            yield type(obj).__name__, obj


def gather_extension_state(opt):
    """Collect the extension and converger state to write into a checkpoint.

    Extensions that have no state say so by returning None and are left out
    entirely, so the common case adds nothing to the file.

    An extension whose state cannot be pickled will fail the write, loudly and
    at every checkpoint point. That is deliberate: dropping just that
    extension's state would produce checkpoints that look complete and resume
    into a run that silently diverges, which is the failure this whole
    contract exists to prevent.
    """
    extensions = {}
    for name, ext in _extension_objects(opt):
        state = ext.checkpoint_state()
        if state is not None:
            extensions[name] = state

    converger = None
    convobject = getattr(opt, "convobject", None)
    if convobject is not None:
        state = convobject.checkpoint_state()
        if state is not None:
            converger = {"class": type(convobject).__name__, "state": state}

    if not extensions and converger is None:
        return None
    return {"extensions": extensions, "converger": converger}


def _overrides_state_hooks(cls, base):
    """Whether ``cls`` overrides each of the two state hooks of ``base``.

    Compared on the class, not on an instance: a bound method is a new object
    on every attribute access, so it is never the base's function.
    """
    return (cls.checkpoint_state is not base.checkpoint_state,
            cls.restore_state is not base.restore_state)


def require_state_contract(opt):
    """Refuse a checkpointed run whose extensions or converger have not said
    what happens to their state across a resume.

    The base ``checkpoint_state`` and ``restore_state`` raise, so an extension
    that overrides neither would end the run at its first checkpoint write --
    hours in, on a long run -- and one that overrides only
    ``checkpoint_state`` would end it at the resume, which is worse. Both are
    known from the classes alone, before anything is solved, so they are
    refused here instead. A class with no state says so by overriding both to
    return None and do nothing.

    Reads only the classes attached, which every rank of the cylinder was
    given identically, so every rank refuses or none does.
    """
    from mpisppy.convergers.converger import Converger
    from mpisppy.extensions.extension import Extension
    classes = [(name, type(ext), Extension)
               for name, ext in _extension_objects(opt)]
    converger_class = getattr(opt, "ph_converger", None)
    if converger_class is not None:
        classes.append((converger_class.__name__, converger_class, Converger))

    neither, half = [], []
    for name, cls, base in classes:
        saves, restores = _overrides_state_hooks(cls, base)
        if not saves and not restores:
            neither.append(name)
        elif saves != restores:
            half.append(f"{name} (implements "
                        f"{'checkpoint_state' if saves else 'restore_state'} "
                        f"only)")
    if not neither and not half:
        return
    problems = []
    if neither:
        problems.append(
            f"these do not say what happens to their state across a resume: "
            f"{', '.join(sorted(neither))}. One that decides what to do next "
            f"from what it did earlier -- a history, a counter, a record of "
            f"what it already changed -- would not retrace an uninterrupted "
            f"run")
    if half:
        problems.append(
            f"these implement one of the two state hooks without the other: "
            f"{', '.join(sorted(half))}")
    raise RuntimeError(
        "Checkpointing (--checkpoint-dir or --resume-from) needs every "
        "attached extension and the converger to implement both "
        "checkpoint_state and restore_state; " + "; and ".join(problems) +
        ". Implement both on each class named; one with no state of its own "
        "returns None from checkpoint_state and does nothing in "
        "restore_state. Or run without checkpointing.")


def restore_extension_state(opt, state):
    """Hand each extension its own state back. Returns a list of warnings.

    Warnings rather than errors: a resume with a different extension set is
    something a user may legitimately do (the checkpoint's *hub iterate* is
    still valid), so it should say clearly what it could not restore instead
    of refusing the whole checkpoint. The caller prints them.

    Every attached extension has implemented both hooks by the time this
    runs -- require_state_contract refused the run at startup otherwise -- so
    what is left to report is a checkpoint entry with no extension to hand it
    to, and a converger that differs from the checkpoint's.
    """
    warnings = []
    if not state:
        return warnings

    attached = dict(_extension_objects(opt))
    for name, ext_state in state.get("extensions", {}).items():
        ext = attached.get(name)
        if ext is None:
            warnings.append(
                f"the checkpoint holds state for the extension '{name}', "
                f"which is not attached to this run; it was dropped. If this "
                f"run was meant to continue the earlier one, attach it.")
            continue
        # An extension may return a sentence about what it could not put
        # back exactly; it goes out with the rest of the resume's warnings.
        message = ext.restore_state(ext_state)
        if message:
            warnings.append(f"{name}: {message}")

    saved = state.get("converger")
    convobject = getattr(opt, "convobject", None)
    if saved is not None:
        if convobject is None:
            warnings.append(
                f"the checkpoint holds state for the converger "
                f"'{saved['class']}', but this run has no converger.")
        elif type(convobject).__name__ != saved["class"]:
            warnings.append(
                f"the checkpoint's converger was '{saved['class']}' but this "
                f"run uses '{type(convobject).__name__}'; the checkpointed "
                f"converger state was dropped and this converger starts "
                f"fresh.")
        else:
            convobject.restore_state(saved["state"])
    return warnings


def write_checkpoint(opt, ckpt_dir, generation, backend=DILL_RELOAD_BACKEND):
    """Write and atomically publish one checkpoint generation.

    Collective over the cylinder. Every rank writes its own rank-tagged files
    into a shared staging directory that rank 0 prepared; the ranks then agree
    on whether all of those writes succeeded, and only if they did does rank 0
    rename the staging directory into place and rewrite the manifest (itself
    temp-then-rename) to point at it. That manifest flip is the single commit
    point, so a kill before it leaves the previous checkpoint intact and a kill
    after it leaves the new one. The prior generation is deleted once the
    manifest names its replacement. Each rename step is followed by an fsync of
    the directory that recorded it, so the commit point holds across a power
    loss and not just a kill.

    Raising here is what makes the write all-or-nothing: if *any* rank failed,
    every rank raises, no manifest is written, and the caller
    (``Checkpointer.maybe_checkpoint``) warns and carries on with the previous
    generation still published and still resumable.
    """
    timed = bool(opt.options.get("display_timing", False))
    rank_started = time.perf_counter() if timed else None
    run_agreed(
        opt, lambda: require_checkpoint_structural_cfg(opt.options),
        "describe their checkpoint's structural model configuration, so none "
        "of them writes one")

    # The dependency check depends only on the selected codec, so every rank
    # reaches the same verdict before any rank-local filesystem work.
    require_backend_dependency(backend)

    phase = ({"validation_s": time.perf_counter() - rank_started}
             if timed else None)
    model_io = _FileWriteTiming() if timed else None
    leaf_io = _FileWriteTiming() if timed else None
    nl_hash_io = _FileWriteTiming() if timed else None
    if model_io is not None:
        model_io.nl_hash_file_io = nl_hash_io
    comm = _cylinder_comm(opt)
    rank = int(opt.cylinder_rank)
    is_publisher = rank == 0
    write_id = None
    if _is_pyfory_backend(backend):
        proposed_id = uuid.uuid4().hex if is_publisher else None
        write_id = proposed_id if comm is None else comm.bcast(
            proposed_id, root=0)
        if not isinstance(write_id, str) or not write_id:
            raise RuntimeError("Could not agree on the pyfory write identity.")
    hub_dir = os.path.join(ckpt_dir, HUB_SUBDIR)
    final_dir = os.path.join(hub_dir, _generation_dirname(generation))
    staging_dir = f"{final_dir}.tmp"

    # Rank 0's directory preparation is guarded like every other rank-local
    # step, and for the same reason: if it raised straight out of here, rank 0
    # would never reach the barrier below and every other rank would wait at
    # it for the rest of the job.
    failure = None
    started = time.perf_counter() if timed else None
    if is_publisher:
        try:
            # A failed publish can leave a published-but-uncommitted
            # generation behind. Reclaim it before staging another: on a full
            # disk the orphan is what makes the retry fail, so the sweep after
            # a successful write would never be reached.
            _sweep_uncommitted_generations(ckpt_dir, hub_dir)
            if os.path.isdir(staging_dir):
                shutil.rmtree(staging_dir)
            os.makedirs(staging_dir, exist_ok=True)
        except Exception as exc:
            failure = exc
    if timed:
        phase["prepare_s"] = time.perf_counter() - started
    # No rank may write into the staging directory until rank 0 has cleared
    # and recreated it, or its files are deleted out from under it.
    started = time.perf_counter() if timed else None
    _barrier(comm)
    if timed:
        phase["prepare_wait_s"] = time.perf_counter() - started

    local_started = time.perf_counter() if timed else None
    try:
        if failure is not None:
            raise failure
        # Every rank makes the directory anyway: it costs nothing when it is
        # already there, and on a network filesystem the barrier does not
        # guarantee rank 0's mkdir is visible here yet.
        os.makedirs(staging_dir, exist_ok=True)
        # Redundant if the setup-time check passed -- local_scenarios does not
        # change during a run -- but it is per-rank data, so it belongs inside
        # the guarded region rather than ahead of the barrier.
        check_filename_collisions(opt.local_scenarios)
        started = time.perf_counter() if timed else None
        nl_hash_files = {
            sname: _nl_fingerprint_filename(rank, sname)
            for sname in opt.local_scenarios
        } if opt.options.get("checkpoint_nl_fingerprint", False) and \
            backend == PYFORY_REBUILD_BACKEND else {}
        model_files = _write_models(opt, staging_dir, rank, backend,
                                    write_id=write_id, timing=model_io)
        if timed:
            phase["model_s"] = time.perf_counter() - started
        started = time.perf_counter() if timed else None
        leaf = {
            "format_version": FORMAT_VERSION,
            "backend": backend,
            "generation": int(generation),
            "geometry": geometry(opt),
            "structural_fingerprint": structural_fingerprint(opt.options),
            "model_files": model_files,
            "initially_fixed_nonants": initially_fixed_nonant_names(opt),
            "extension_state": gather_extension_state(opt),
            "trivial_bound": _as_float_or_none(
                getattr(opt, "trivial_bound", None)),
            "best_bound_obj_val": _as_float_or_none(
                getattr(opt, "best_bound_obj_val", None)),
            "best_solution_obj_val": _as_float_or_none(
                getattr(opt, "best_solution_obj_val", None)),
            "best_outer_bound": _hub_best_outer_bound(opt),
        }
        if _is_pyfory_backend(backend):
            leaf["write_id"] = write_id
        if backend == PYFORY_REBUILD_BACKEND:
            leaf["model_fingerprint_method"] = (
                NL_FINGERPRINT_METHOD if opt.options.get(
                    "checkpoint_nl_fingerprint", False)
                else GRAPH_FINGERPRINT_METHOD)
            if leaf["model_fingerprint_method"] == NL_FINGERPRINT_METHOD:
                leaf["nl_fingerprint_files"] = nl_hash_files
            leaf["extension_model_state"] = gather_extension_model_state(opt)
        _atomic_write_bytes(
            os.path.join(staging_dir, _leaf_filename(rank)),
            lambda f: pickle.dump(leaf, f),
            timing=leaf_io,
        )
        if timed:
            phase["leaf_s"] = time.perf_counter() - started
        started = time.perf_counter() if timed else None
        _fsync_dir(staging_dir)
        if timed:
            phase["stage_dir_sync_s"] = time.perf_counter() - started
    except Exception as exc:
        failure = exc
    if timed:
        phase["local_s"] = time.perf_counter() - local_started

    # Collective, and therefore also the barrier that says every rank has
    # finished writing. Nothing below may run before it.
    started = time.perf_counter() if timed else None
    failing_rank = _first_failing_rank(comm, rank, failure is not None)
    if timed:
        phase["write_wait_s"] = time.perf_counter() - started
    if failing_rank is not None:
        if is_publisher:
            # Leave no half-written generation behind; the previous checkpoint
            # (if any) stays published, since the manifest was never touched.
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise _write_failure(opt, ckpt_dir, failure, failing_rank, rank)

    # Publishing is rank 0's alone and comes after the last collective, so a
    # failure in it cannot desynchronize anyone: rank 0 raises and warns, the
    # other ranks return, the manifest still names the previous generation and
    # the next checkpoint point retries if the run gets that far.
    if is_publisher:
        try:
            started = time.perf_counter() if timed else None
            _publish_generation(opt, ckpt_dir, hub_dir, final_dir,
                                staging_dir, generation, backend,
                                write_id=write_id, timing=phase)
            if timed:
                phase["publish_s"] = time.perf_counter() - started
        except Exception:
            # The manifest still names the previous generation (or nothing),
            # so everything else is garbage; give the disk back now rather
            # than at the next checkpoint point.
            _sweep_uncommitted_generations(ckpt_dir, hub_dir)
            raise

    if timed:
        # One parseable line per rank. Non-publishers report zero for the
        # publication fields; their elapsed time ends at the write agreement.
        print(
            f"checkpoint_timing event=write backend={backend} "
            f"generation={int(generation)} "
            f"rank={rank} scenarios={len(opt.local_scenarios)} "
            f"model_bytes={model_io.bytes_written} "
            f"model_s={phase['model_s']:.6f} "
            f"snapshot_s={model_io.snapshot_s:.6f} "
            f"snapshot_cpu_s={model_io.snapshot_cpu_s:.6f} "
            f"graph_fingerprint_s={model_io.graph_fingerprint_s:.6f} "
            f"graph_fingerprint_cpu_s="
            f"{model_io.graph_fingerprint_cpu_s:.6f} "
            f"nl_write_s={model_io.nl_write_s:.6f} "
            f"nl_write_cpu_s={model_io.nl_write_cpu_s:.6f} "
            f"nl_digest_s={model_io.nl_digest_s:.6f} "
            f"nl_digest_cpu_s={model_io.nl_digest_cpu_s:.6f} "
            f"model_dump_s={model_io.callback_s:.6f} "
            f"model_dump_cpu_s={model_io.callback_cpu_s:.6f} "
            f"model_flush_s={model_io.flush_s:.6f} "
            f"model_fsync_s={model_io.fsync_s:.6f} "
            f"model_replace_s={model_io.replace_s:.6f} "
            f"nl_hash_bytes={nl_hash_io.bytes_written} "
            f"nl_hash_file_s={nl_hash_io.callback_s:.6f} "
            f"nl_hash_file_fsync_s={nl_hash_io.fsync_s:.6f} "
            f"nl_hash_file_replace_s={nl_hash_io.replace_s:.6f} "
            f"leaf_bytes={leaf_io.bytes_written} "
            f"leaf_s={phase['leaf_s']:.6f} "
            f"leaf_pickle_s={leaf_io.callback_s:.6f} "
            f"leaf_pickle_cpu_s={leaf_io.callback_cpu_s:.6f} "
            f"leaf_flush_s={leaf_io.flush_s:.6f} "
            f"leaf_fsync_s={leaf_io.fsync_s:.6f} "
            f"leaf_replace_s={leaf_io.replace_s:.6f} "
            f"stage_dir_sync_s={phase['stage_dir_sync_s']:.6f} "
            f"local_s={phase['local_s']:.6f} "
            f"validation_s={phase['validation_s']:.6f} "
            f"prepare_s={phase['prepare_s']:.6f} "
            f"prepare_wait_s={phase['prepare_wait_s']:.6f} "
            f"write_wait_s={phase['write_wait_s']:.6f} "
            f"publish_s={phase.get('publish_s', 0.0):.6f} "
            f"publish_swap_s={phase.get('publish_swap_s', 0.0):.6f} "
            f"publish_hub_sync_s={phase.get('publish_hub_sync_s', 0.0):.6f} "
            f"publish_manifest_s={phase.get('publish_manifest_s', 0.0):.6f} "
            f"publish_cleanup_s={phase.get('publish_cleanup_s', 0.0):.6f} "
            f"rank_elapsed_s={time.perf_counter() - rank_started:.6f}",
            flush=True)

    return final_dir


def _write_failure(opt, ckpt_dir, failure, failing_rank, rank):
    """The exception every rank raises when any rank's write failed.

    A rank that succeeded still has to raise -- the generation is
    all-or-nothing -- but it has no exception of its own to describe, so it
    names the rank that does. Otherwise a multi-rank failure would print one
    real diagnosis and n-1 misleading ones.

    The exception carries ``mpisppy_failed_locally`` so the caller can decide
    who reports it. Warnings are normally printed by rank 0 alone, which would
    silence exactly the rank holding the cause.
    """
    if failure is None:
        err = RuntimeError(
            f"Rank {failing_rank} could not write its part of the checkpoint "
            f"in '{ckpt_dir}', so this generation was abandoned on every "
            f"rank. See that rank's message for the cause. Any previously "
            f"published checkpoint is untouched."
        )
        err.mpisppy_failed_locally = False
        return err
    # A bad backend is a programming/configuration error, not a disk problem:
    # it must not be dressed up as a transient write failure.
    if isinstance(failure, ValueError):
        failure.mpisppy_failed_locally = True
        return failure
    first = next(iter(opt.local_scenarios.values()), None)
    detail = (pickle_bundle.describe_dill_failure(first, failure,
                                                  what="scenario model")
              if first is not None
              else f"{type(failure).__name__}: {failure}")
    where = "" if failing_rank == rank else f" (first failure on rank {failing_rank})"
    err = RuntimeError(
        f"Failed to write the checkpoint to '{ckpt_dir}'{where}. Any "
        f"previously published checkpoint is untouched.\n\n" + detail
    )
    err.mpisppy_failed_locally = True
    return err


def _pyfory_version():
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("pyfory")
    except PackageNotFoundError:
        return str(getattr(pyfory, "__version__", "unknown"))


def _publish_generation(opt, ckpt_dir, hub_dir, final_dir, staging_dir,
                        generation, backend, write_id=None, timing=None):
    """Commit the staged generation. Rank 0 only, once every rank has written.

    Publishing order matters. The manifest is the commit point, so the
    generation it currently names must stay on disk and intact until the
    replacement is fully published -- otherwise a kill in between destroys the
    only checkpoint. Stage under a name nothing points at, publish, then sweep.
    Writing the same generation number twice therefore lands in a scratch
    directory first rather than deleting the live one.
    """
    started = time.perf_counter() if timing is not None else None
    scratch_dir = f"{final_dir}.incoming"
    if os.path.isdir(scratch_dir):
        shutil.rmtree(scratch_dir)
    os.replace(staging_dir, scratch_dir)

    # Retire the old generation by *renaming* it rather than deleting it. The
    # window where the manifest names a directory that is momentarily absent
    # shrinks from an rmtree of the whole generation to a single rename, and
    # load_checkpoint knows to look in the retired copy, so an interruption
    # inside that window is still resumable. The sweep below reclaims it.
    retiring_dir = f"{final_dir}.retiring"
    if os.path.isdir(final_dir):
        # Only clear a previous retiring copy when there is a live generation
        # to replace it with. If final_dir is absent we were interrupted
        # between these two renames on an earlier attempt, and the retiring
        # copy is the last good data -- deleting it here would be the retry
        # destroying what it is retrying to protect.
        if os.path.isdir(retiring_dir):
            shutil.rmtree(retiring_dir, ignore_errors=True)
        os.replace(final_dir, retiring_dir)
    os.replace(scratch_dir, final_dir)
    if timing is not None:
        timing["publish_swap_s"] = time.perf_counter() - started
    started = time.perf_counter() if timing is not None else None
    _fsync_dir(hub_dir)
    if timing is not None:
        timing["publish_hub_sync_s"] = time.perf_counter() - started

    manifest = {
        "format_version": FORMAT_VERSION,
        "backend": backend,
        "generation": int(generation),
        "n_proc": int(opt.n_proc),
        "structural_fingerprint": structural_fingerprint(opt.options),
    }
    if _is_pyfory_backend(backend):
        if not write_id:
            raise RuntimeError(f"{backend} publication requires write_id.")
        manifest["write_id"] = write_id
        manifest["codec_version"] = _pyfory_version()
        if backend == PYFORY_RELOAD_BACKEND:
            from mpisppy.utils import pyfory_model_codec
            manifest["reload_schema_version"] = \
                pyfory_model_codec.SCHEMA_VERSION
            manifest["reload_adapter_version"] = \
                pyfory_model_codec.ADAPTER_VERSION
    started = time.perf_counter() if timing is not None else None
    _publish_manifest(ckpt_dir, manifest)
    if timing is not None:
        timing["publish_manifest_s"] = time.perf_counter() - started

    # Sweep everything the manifest does not name, rather than only the
    # generation the previous manifest did. A kill between any two steps above
    # can leave a directory behind, and deleting just the known predecessor
    # would let those accumulate for the life of the run.
    started = time.perf_counter() if timing is not None else None
    _sweep_uncommitted_generations(ckpt_dir, hub_dir)
    if timing is not None:
        timing["publish_cleanup_s"] = time.perf_counter() - started


def _sweep_uncommitted_generations(ckpt_dir, hub_dir):
    """Delete every generation directory the current manifest does not name.

    With no manifest nothing is committed, so every generation directory
    goes. A manifest that cannot be read deletes nothing: it may still name
    the only good checkpoint.
    """
    try:
        manifest = _read_manifest(ckpt_dir, missing_ok=True)
        keep = None if manifest is None else int(manifest["generation"])
    except (OSError, ValueError, KeyError, TypeError):
        return
    keep_copy = None
    if manifest is not None:
        backend = manifest.get("backend")
        write_id = manifest.get("write_id")
        try:
            keep_copy = _generation_copy_for_manifest(
                hub_dir, keep, backend, write_id, rank=0)
        except CheckpointMismatch:
            # Do not delete either candidate if the manifest's committed copy
            # cannot be identified. The next write must fail with the same
            # refusal rather than discard the only possibly good generation.
            raise
    _sweep_stale_generations(hub_dir, keep, keep_copy=keep_copy)


def _sweep_stale_generations(hub_dir, keep, keep_copy=None):
    """Delete every generation directory except the one the manifest names.

    ``keep=None`` means the manifest names none. When the named generation's
    directory is absent, its ``.retiring`` copy is kept: that is the state a
    kill between the two publishing renames leaves, and ``load_checkpoint``
    reads the retired copy.
    """
    keep_names = set()
    if keep is not None:
        keep_name = _generation_dirname(keep)
        if keep_copy is not None:
            keep_names.add(os.path.basename(keep_copy))
        else:
            keep_names.add(keep_name)
            if not os.path.isdir(os.path.join(hub_dir, keep_name)):
                keep_names.add(f"{keep_name}.retiring")
    try:
        entries = os.listdir(hub_dir)
    except OSError:
        return
    for name in entries:
        if name in keep_names or not name.startswith("gen_"):
            continue
        path = os.path.join(hub_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def _write_models(opt, staging_dir, rank, backend, write_id=None,
                  timing=None):
    """Write each local scenario using the selected model-state codec."""
    require_backend_dependency(backend)
    use_nl_fingerprint = bool(
        backend == PYFORY_REBUILD_BACKEND and
        (getattr(opt, "options", None) or {}).get(
            "checkpoint_nl_fingerprint", False))
    nl_hash_file_timing = (getattr(timing, "nl_hash_file_io", None)
                           if timing is not None else None)
    if backend == PYFORY_RELOAD_BACKEND:
        from mpisppy.utils import pyfory_model_codec
    model_files = {}
    for sname, s in opt.local_scenarios.items():
        fname = _model_filename(rank, sname, backend)
        if backend in (DILL_RELOAD_BACKEND, PYFORY_RELOAD_BACKEND):
            # Keep the historic model bytes and plugin removal dance intact.
            solver_plugin = getattr(s, "_solver_plugin", None)
            if solver_plugin is not None:
                del s._solver_plugin
            try:
                if backend == DILL_RELOAD_BACKEND:
                    dump_model = dill.dump
                else:
                    dump_model = lambda model, stream, name=sname: \
                        pyfory_model_codec.dump(
                            model, stream, scenario_name=name,
                            write_id=write_id)
                _atomic_write_bytes(
                    os.path.join(staging_dir, fname),
                    lambda f, model=s: dump_model(model, f),
                    timing=timing,
                )
            finally:
                if solver_plugin is not None:
                    s._solver_plugin = solver_plugin
        else:
            started = time.perf_counter() if timing is not None else None
            cpu_started = time.thread_time() if timing is not None else None
            try:
                snapshot = _snapshot_rebuild_model(s)
            finally:
                if timing is not None:
                    timing.snapshot_s += time.perf_counter() - started
                    timing.snapshot_cpu_s += (
                        time.thread_time() - cpu_started)
            if use_nl_fingerprint:
                fingerprint = _rebuild_model_nl_fingerprint(s, timing=timing)
                hash_fname = _nl_fingerprint_filename(rank, sname)
                _atomic_write_bytes(
                    os.path.join(staging_dir, hash_fname),
                    lambda f, digest=fingerprint: f.write(
                        (digest + "\n").encode("ascii")),
                    timing=nl_hash_file_timing,
                )
            else:
                use_component_cache = bool(
                    (getattr(opt, "options", None) or {}).get(
                        "checkpoint_component_fingerprint_cache", False))
                started = time.perf_counter() if timing is not None else None
                cpu_started = time.thread_time() if timing is not None else None
                try:
                    if use_component_cache:
                        caches = getattr(
                            opt, "_pyfory_rebuild_fingerprint_cache", None)
                        if caches is None:
                            caches = {}
                            opt._pyfory_rebuild_fingerprint_cache = caches
                        dirty_by_scenario = getattr(
                            opt, "_pyfory_rebuild_fingerprint_dirty", None)
                        if dirty_by_scenario is None:
                            dirty_by_scenario = {}
                            opt._pyfory_rebuild_fingerprint_dirty = \
                                dirty_by_scenario
                        fingerprint = _rebuild_model_fingerprint_with_cache(
                            s,
                            caches.setdefault(sname, {}),
                            dirty_by_scenario.setdefault(sname, set()),
                        )
                        snapshot["model_fingerprint_version"] = 2
                    else:
                        fingerprint = _rebuild_model_fingerprint(s)
                finally:
                    if timing is not None:
                        timing.graph_fingerprint_s += (
                            time.perf_counter() - started)
                        timing.graph_fingerprint_cpu_s += (
                            time.thread_time() - cpu_started)
                snapshot["model_fingerprint"] = fingerprint
            _atomic_write_bytes(
                os.path.join(staging_dir, fname),
                lambda f, state=snapshot: _new_fory().dump(state, f),
                timing=timing,
            )
        model_files[sname] = fname
    return model_files


def _as_float_or_none(value):
    return None if value is None else float(value)


def _hub_best_outer_bound(opt):
    """The hub's best outer bound from any source, or None if it has none.

    ``opt.best_bound_obj_val`` holds only the bounds the hub computed itself;
    a bound sent by a spoke, such as a Lagrangian bound, lives on
    ``opt.spcomm.BestOuterBound`` and nowhere else.
    """
    bound = getattr(getattr(opt, "spcomm", None), "BestOuterBound", None)
    if bound is None or not math.isfinite(bound):
        return None
    return float(bound)


def _publish_manifest(ckpt_dir, manifest):
    blob = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(os.path.join(ckpt_dir, MANIFEST_NAME),
                        lambda f: f.write(blob))
    _fsync_dir(ckpt_dir)


def _read_manifest(ckpt_dir, missing_ok=False):
    path = os.path.join(ckpt_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        if missing_ok:
            return None
        raise CheckpointMismatch(
            f"No checkpoint manifest at '{path}'. --resume-from expects a "
            f"directory that a previous run wrote with --checkpoint-dir."
        )
    with open(path) as f:
        return json.load(f)


def _generation_copy_for_manifest(hub_dir, generation, backend, write_id,
                                  rank):
    """Find the final or retiring directory identified by the manifest."""
    if _is_pyfory_backend(backend) and not write_id:
        raise CheckpointMismatch(
            f"The {backend} manifest has no write_id, so its generation "
            "cannot be selected safely."
        )
    dirname = _generation_dirname(int(generation))
    candidates = [os.path.join(hub_dir, dirname),
                  os.path.join(hub_dir, f"{dirname}.retiring")]
    candidates = [path for path in candidates if os.path.isdir(path)]
    matches = []
    for path in candidates:
        leaf_path = os.path.join(path, _leaf_filename(rank))
        try:
            with open(leaf_path, "rb") as stream:
                leaf = pickle.load(stream)
        except (OSError, pickle.PickleError, EOFError, AttributeError,
                ValueError, TypeError):
            continue
        if not isinstance(leaf, dict):
            continue
        if leaf.get("backend") != backend:
            continue
        if _is_pyfory_backend(backend) and \
                leaf.get("write_id") != write_id:
            continue
        matches.append(path)

    if backend == DILL_RELOAD_BACKEND and len(matches) > 1:
        # Two dill copies have no write identity. Preserve their historical
        # final-first behavior; the same-backend ambiguity predates pyfory.
        final = os.path.join(hub_dir, dirname)
        return final if final in matches else matches[0]
    if len(matches) != 1:
        reason = "no matching copy" if not matches else "multiple matching copies"
        raise CheckpointMismatch(
            f"The checkpoint manifest's generation {generation} has "
            f"{reason} for backend '{backend}' and write identity "
            f"'{write_id}'. Both generation directories have been preserved."
        )
    return matches[0]


#: The hub leaf keys a resume reads by name. The load refuses a leaf missing
#: any of them, so the refusal is agreed across the hub's ranks.
LEAF_KEYS_READ_ON_RESUME = (
    "format_version", "backend", "generation", "geometry",
    "structural_fingerprint", "model_files", "initially_fixed_nonants",
    "trivial_bound", "best_bound_obj_val", "best_outer_bound",
    "best_solution_obj_val", "extension_state",
)


def load_checkpoint(opt, ckpt_dir):
    """Load this rank's checkpoint, refusing a mismatch with a clear error.

    Returns ``(leaf_state, {scenario_name: reloaded_model})``. The caller is
    responsible for splicing the models into the run (see
    ``PHBase._resume_from_checkpoint``).
    """
    require_checkpoint_structural_cfg(opt.options)
    manifest = _read_manifest(ckpt_dir)

    if manifest.get("format_version") != FORMAT_VERSION:
        raise CheckpointMismatch(
            f"Checkpoint in '{ckpt_dir}' has format version "
            f"{manifest.get('format_version')}, but this mpi-sppy writes and "
            f"reads version {FORMAT_VERSION}. Checkpoints are not portable "
            f"across format versions."
        )

    backend = manifest.get("backend")
    require_backend_dependency(backend)
    if _is_pyfory_backend(backend):
        saved_version = manifest.get("codec_version")
        if not isinstance(saved_version, str) or not saved_version:
            raise CheckpointMismatch(
                f"The {backend} manifest has no codec_version metadata."
            )
        current_version = _pyfory_version()
        if saved_version != current_version:
            raise CheckpointMismatch(
                f"This checkpoint was written with pyfory {saved_version}, "
                f"but this run has pyfory {current_version}. "
                f"{backend} requires an exact codec version match."
            )
        if backend == PYFORY_RELOAD_BACKEND:
            from mpisppy.utils import pyfory_model_codec
            if manifest.get("reload_schema_version") != \
                    pyfory_model_codec.SCHEMA_VERSION or \
                    manifest.get("reload_adapter_version") != \
                    pyfory_model_codec.ADAPTER_VERSION:
                raise CheckpointMismatch(
                    "The pyfory-reload manifest has an unsupported model "
                    "schema or adapter version.")

    expected_fp = structural_fingerprint(opt.options)
    if manifest.get("structural_fingerprint") != expected_fp:
        raise CheckpointMismatch(
            f"The checkpoint in '{ckpt_dir}' was written by a run whose "
            f"configuration differs from this one in a way that could make "
            f"the checkpoint describe a different problem. Everything is "
            f"compared except a short list of entries a resume may change "
            f"freely -- the iteration and time limits, display and output "
            f"options, checkpoint plumbing, and solver selection. Anything "
            f"else, including options your model's own inparser_adder "
            f"registers, must match the run that wrote the checkpoint."
        )

    if int(manifest.get("n_proc", -1)) != int(opt.n_proc):
        raise CheckpointMismatch(
            f"The checkpoint in '{ckpt_dir}' was written on "
            f"{manifest.get('n_proc')} rank(s) but this run has "
            f"{opt.n_proc}. Resuming across a different rank count is not "
            f"supported; rerun with the original rank count."
        )

    generation = manifest["generation"]
    rank = int(opt.cylinder_rank)
    hub_dir = os.path.join(ckpt_dir, HUB_SUBDIR)
    gen_dir = _generation_copy_for_manifest(
        hub_dir, generation, backend, manifest.get("write_id"), rank)

    leaf_path = os.path.join(gen_dir, _leaf_filename(rank))
    if not os.path.exists(leaf_path):
        raise CheckpointMismatch(
            f"The checkpoint in '{ckpt_dir}' has no state for rank {rank} "
            f"(expected '{leaf_path}')."
        )
    with open(leaf_path, "rb") as f:
        leaf = pickle.load(f)

    # Checked here, inside the agreement the load runs in, because Iter0
    # reads the bounds outside one, before the hub's next collective: a leaf
    # without one would raise KeyError on that rank alone and leave the
    # others waiting in it.
    if not isinstance(leaf, dict):
        raise CheckpointMismatch(
            f"The checkpoint state '{leaf_path}' is not a dictionary, so it "
            "was not written by this mpi-sppy or has been damaged.")
    missing = [key for key in LEAF_KEYS_READ_ON_RESUME if key not in leaf]
    if _is_pyfory_backend(backend):
        missing.extend(key for key in ("write_id",)
                       if key not in leaf)
    if backend == PYFORY_REBUILD_BACKEND and \
            "extension_model_state" not in leaf:
        missing.append("extension_model_state")
    if missing:
        raise CheckpointMismatch(
            f"The checkpoint state '{leaf_path}' has no "
            f"{', '.join(missing)}, so it was not written by this mpi-sppy "
            f"or has been damaged."
        )

    expected_leaf = {
        "format_version": manifest["format_version"],
        "backend": backend,
        "generation": generation,
        "structural_fingerprint": manifest["structural_fingerprint"],
    }
    mismatched = [key for key, value in expected_leaf.items()
                  if leaf[key] != value]
    if _is_pyfory_backend(backend) and \
            leaf.get("write_id") != manifest.get("write_id"):
        mismatched.append("write_id")
    if mismatched:
        raise CheckpointMismatch(
            f"The checkpoint state '{leaf_path}' does not match its manifest "
            f"for {', '.join(mismatched)}, so it is stale or has been "
            "damaged.")

    expected_geometry = geometry(opt)
    if leaf["geometry"] != expected_geometry:
        raise CheckpointMismatch(
            f"Rank {rank}'s checkpoint geometry does not match this run. "
            "Resuming requires an identical scenario-to-rank distribution."
        )

    expected_model_files = {
        sname: _model_filename(rank, sname, backend)
        for sname in opt.local_scenarios
    }
    if leaf["model_files"] != expected_model_files:
        raise CheckpointMismatch(
            f"The checkpoint state '{leaf_path}' does not have exactly the "
            "model files this rank's scenarios require, so it is stale or "
            "has been damaged.")

    fingerprint_method = GRAPH_FINGERPRINT_METHOD
    if backend == PYFORY_REBUILD_BACKEND:
        fingerprint_method = leaf.get(
            "model_fingerprint_method", GRAPH_FINGERPRINT_METHOD)
        if fingerprint_method not in (GRAPH_FINGERPRINT_METHOD,
                                      NL_FINGERPRINT_METHOD):
            raise CheckpointMismatch(
                f"The checkpoint state '{leaf_path}' has unsupported model "
                f"fingerprint method {fingerprint_method!r}.")
        if fingerprint_method == NL_FINGERPRINT_METHOD:
            expected_nl_files = {
                sname: _nl_fingerprint_filename(rank, sname)
                for sname in opt.local_scenarios
            }
            if leaf.get("nl_fingerprint_files") != expected_nl_files:
                raise CheckpointMismatch(
                    f"The checkpoint state '{leaf_path}' does not have "
                    "exactly the NL fingerprint files this rank's scenarios "
                    "require, so it is stale or has been damaged.")
            nl_fingerprints = {}
            for sname, fname in expected_nl_files.items():
                path = os.path.join(gen_dir, fname)
                try:
                    with open(path, "rb") as f:
                        contents = f.read()
                except OSError as exc:
                    raise CheckpointMismatch(
                        f"The checkpoint in '{ckpt_dir}' is missing or "
                        f"cannot read NL fingerprint file '{fname}' for "
                        f"scenario '{sname}': {exc}") from exc
                if not re.fullmatch(rb"[0-9a-f]{64}\n", contents):
                    raise CheckpointMismatch(
                        f"The NL fingerprint file '{fname}' for scenario "
                        f"'{sname}' must contain exactly one lowercase "
                        "64-character SHA-256 digest followed by a newline.")
                nl_fingerprints[sname] = contents[:-1].decode("ascii")
            leaf["nl_fingerprints"] = nl_fingerprints

    models = {}
    if backend == PYFORY_RELOAD_BACKEND:
        from mpisppy.utils import pyfory_model_codec
    for sname, fname in leaf["model_files"].items():
        if not os.path.isfile(os.path.join(gen_dir, fname)):
            raise CheckpointMismatch(
                f"The checkpoint in '{ckpt_dir}' is missing the model file "
                f"'{fname}' for scenario '{sname}'.")
    for sname, fname in leaf["model_files"].items():
        path = os.path.join(gen_dir, fname)
        with open(path, "rb") as f:
            if backend == DILL_RELOAD_BACKEND:
                models[sname] = dill.load(f)
            elif backend == PYFORY_RELOAD_BACKEND:
                models[sname] = pyfory_model_codec.load(
                    f, scenario_name=sname, write_id=manifest["write_id"])
                _validate_reloaded_model(
                    models[sname], opt.local_scenarios[sname], sname)
            else:
                models[sname] = _new_fory().deserialize(f.read())
        if backend == PYFORY_REBUILD_BACKEND:
            state = models[sname]
            if not isinstance(state, dict) or \
                    state.get("state_version") != PYFORY_STATE_VERSION:
                raise CheckpointMismatch(
                    f"The pyfory model file '{fname}' for scenario "
                    f"'{sname}' has an unsupported state format."
                )

    return leaf, models


###############################################################################
# The spoke side: an xhat spoke's best incumbent.
#
# The hub checkpoint carries the PH iterate; the best *solution* the run has
# found does not live there. It lives on the xhat spoke, in
# ``s._mpisppy_data.best_solution_cache``, and ``InnerBoundSpoke.finalize()``
# loads it back at the end of the run -- so without the file written here, a
# resumed run restores its iterate perfectly and still reports whatever it
# happens to find after the restart, having thrown away the answer it had.
#
# Two things make this unlike the hub's checkpoint:
#
# * **It is by name.** The cache is a ComponentMap keyed by variable
#   *objects*. A resumed spoke builds fresh models, so those keys address
#   nothing; names survive the rebuild. Same discipline as
#   ``initially_fixed_nonant_names``.
# * **It holds no models.** A spoke's models are rebuilt by the
#   ``scenario_creator`` at startup and accumulate nothing worth keeping, so
#   only values are stored. That is why this can be written on every
#   improvement while the hub write is paced by ``--checkpoint-every-iterations``.
#
# It is also deliberately not aligned with the hub's generations: one file per
# spoke per rank, overwritten in place whenever the incumbent improves. See
# section 9 item 6 of doc/designs/checkpointing_design.md for why no
# hub-to-spoke coordination is wanted here.
###############################################################################


#: Passed as ``extension_state`` to mean "gather it from the extensions now",
#: which is what every write but one wants. The one is the write straight
#: after a restore, which has to carry what it read: the spoke hands the read
#: state to its extensions only at the end of its prep, so gathering then
#: would write fresh state over the restored one. None cannot mean "gather",
#: because None is also what a file with no extension state holds.
GATHER_EXTENSION_STATE = object()


def _values_by_name(s, cache):
    """This scenario's cached solution keyed by variable name, built once
    per incumbent.

    A spoke writes on nearly every evaluation (its loop cursor moves), while
    the incumbent changes rarely, and building the dict is almost all of a
    write's cost: Pyomo builds each ``var.name`` on demand, about 2 us per
    variable, so a 160,000-variable UC instance paid about 350 ms per write
    to rebuild a dict that had not changed. ``_cache_best_solution`` replaces
    the cache object on every improvement and nothing edits one in place, so
    the object's identity says whether the dict is still current.
    """
    memo = getattr(s._mpisppy_data, "_checkpoint_values_by_name", None)
    if memo is None or memo[0] is not cache:
        memo = (cache, {var.name: value for var, value in cache.items()})
        s._mpisppy_data._checkpoint_values_by_name = memo
    return memo[1]


def spoke_incumbent_state(opt, cylinder, ordinal, best_inner_bound=None,
                          loop_state=None, class_count=None,
                          extension_state=GATHER_EXTENSION_STATE,
                          progress=None):
    """The dict written by ``write_spoke_incumbent``, or None if there is
    nothing to write yet. ``_cache_best_solution`` caches every scenario at
    once, so one scenario without a cache means none has one."""
    require_checkpoint_structural_cfg(opt.options)
    solutions = {}
    for sname, s in opt.local_scenarios.items():
        cache = s._mpisppy_data.best_solution_cache
        if cache is None:
            return None
        # The per-scenario inner bound rides along because send_best_xhat
        # packs it next to the values; a resumed spoke that published its
        # restored incumbent without it would send whatever the fresh models
        # happen to hold, which is None.
        # The objective of the cached solution, snapshotted with it in
        # _cache_best_solution -- not the live inner_bound, which the next
        # solve overwrites while the values beside it stay put.
        objective = _as_float_or_none(
            getattr(s._mpisppy_data, "best_solution_inner_bound", None))
        if objective is None or not math.isfinite(objective):
            # Refuse the whole file rather than write a solution with no
            # objective. send_best_xhat packs this number into a float64
            # buffer, where None is stored as NaN without anything raising,
            # and FWPH reads that slot as the recourse cost of a QP column.
            # A solver is entitled to accept a solution and report no bound
            # for it (ipopt reports an infinite one), so this is reachable
            # without anything being wrong; what must not happen is passing
            # the gap on silently.
            raise ValueError(
                f"scenario '{sname}' has a cached best solution but no "
                f"finite objective recorded with it ({objective}), so this "
                f"incumbent cannot be checkpointed: the objective travels "
                f"with the values and is read downstream as a real number")
        solutions[sname] = {
            "inner_bound": objective,
            "values": _values_by_name(s, cache),
        }
    return {
        "format_version": FORMAT_VERSION,
        "kind": "spoke-incumbent",
        "cylinder": str(cylinder),
        "ordinal": int(ordinal),
        # How many cylinders of this class the writing wheel carried. The
        # ordinal only means the same thing while that is unchanged: drop one
        # of two same-class spokes and the survivor's ordinal becomes the
        # removed one's. Recorded so the resume can say so when it adopts an
        # incumbent that may have belonged to a different cylinder.
        "class_count": None if class_count is None else int(class_count),
        "rank": int(opt.cylinder_rank),
        "geometry": geometry(opt),
        "structural_fingerprint": structural_fingerprint(opt.options),
        # Two numbers rather than one because they are kept in two places:
        # the spoke's own best_inner_bound gates what it sends to the hub,
        # while the opt object's best_solution_obj_val gates what the
        # solution cache accepts. Restoring one and inferring the other would
        # bake in an equality nothing enforces.
        "best_inner_bound": _as_float_or_none(best_inner_bound),
        "best_solution_obj_val": _as_float_or_none(
            getattr(opt, "best_solution_obj_val", None)),
        # Identifies this write across the spoke's ranks; see spoke_write_id.
        "write_id": spoke_write_id(
            _as_float_or_none(getattr(opt, "best_solution_obj_val", None)),
            progress),
        "solutions": solutions,
        # Where the spoke's own loop had got to. Only xhatshuffle has such a
        # place; every other xhatter re-evaluates from scratch when new
        # nonants arrive, so it says None and this stays None.
        "loop_state": loop_state,
        # Same contract as the hub's, for extensions attached to the spoke's
        # Xhat_Eval rather than to the PH hub.
        "extension_state": (gather_extension_state(opt)
                            if extension_state is GATHER_EXTENSION_STATE
                            else extension_state),
    }


def write_spoke_incumbent(opt, ckpt_dir, cylinder, ordinal,
                          best_inner_bound=None, loop_state=None,
                          class_count=None,
                          extension_state=GATHER_EXTENSION_STATE,
                          progress=None):
    """Write this spoke's best incumbent, latest-wins. Returns the path, or
    None when there is no incumbent to write.

    The write is the same temp-then-rename used for the hub's files, so a
    kill mid-write leaves the previous incumbent intact rather than a
    truncated file. There are no generations to publish and nothing else to
    be consistent with, so the rename *is* the commit point -- no manifest is
    involved.
    """
    state = spoke_incumbent_state(opt, cylinder, ordinal,
                                  best_inner_bound=best_inner_bound,
                                  loop_state=loop_state,
                                  class_count=class_count,
                                  extension_state=extension_state,
                                  progress=progress)
    if state is None:
        return None
    spokes_dir = os.path.join(ckpt_dir, SPOKES_SUBDIR)
    os.makedirs(spokes_dir, exist_ok=True)
    path = os.path.join(
        spokes_dir,
        _spoke_filename(cylinder, ordinal, opt.cylinder_rank),
    )
    _atomic_write_bytes(path, lambda f: pickle.dump(state, f))
    _fsync_dir(spokes_dir)
    return path


def load_spoke_incumbent(opt, ckpt_dir, cylinder, ordinal):
    """Read this spoke's incumbent file, or None if it is not there.

    A missing file is not an error: the run being resumed may have stopped
    before this spoke found anything, and a checkpoint directory is allowed
    to carry no incumbent at all. A file that *is* there but does not match
    this run is an error, for the same reason the hub refuses one -- values
    from a different model are wrong answers, not stale ones.
    """
    require_checkpoint_structural_cfg(opt.options)
    path = os.path.join(
        ckpt_dir, SPOKES_SUBDIR,
        _spoke_filename(cylinder, ordinal, opt.cylinder_rank),
    )
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        state = pickle.load(f)

    if state.get("format_version") != FORMAT_VERSION:
        raise CheckpointMismatch(
            f"The incumbent file '{path}' has format version "
            f"{state.get('format_version')}, but this mpi-sppy writes "
            f"version {FORMAT_VERSION}."
        )
    # Checked here, inside the agreement the load runs in, because the
    # agreed restore relies on both: without one, agree_spoke_restore would
    # compare or broadcast None as though the file had recorded it.
    missing = [key for key in ("best_solution_obj_val", "best_inner_bound")
               if key not in state]
    if missing:
        raise CheckpointMismatch(
            f"The incumbent file '{path}' has no {' or '.join(missing)}, so "
            f"it was not written by this mpi-sppy or has been damaged."
        )
    if state.get("structural_fingerprint") != structural_fingerprint(opt.options):
        raise CheckpointMismatch(
            f"The incumbent file '{path}' was written by a run configured "
            f"differently from this one, so its variable values do not "
            f"describe this model."
        )
    if state.get("kind") != "spoke-incumbent":
        raise CheckpointMismatch(
            f"The incumbent file '{path}' holds {state.get('kind')!r}, not "
            "a spoke incumbent.")
    if (state.get("cylinder"), state.get("ordinal"), state.get("rank")) != (
            str(cylinder), int(ordinal), int(opt.cylinder_rank)):
        raise CheckpointMismatch(
            f"The incumbent file '{path}' does not belong to this spoke.")
    if state.get("geometry") != geometry(opt):
        raise CheckpointMismatch(
            f"The incumbent file '{path}' has checkpoint geometry that does "
            "not match this spoke's rank layout."
        )
    return state


def spoke_inner_bounds_to_restore(communicators, ckpt_dir):
    """``[(strata rank, inner bound)]`` for each spoke in this run that has
    an incumbent file in ``ckpt_dir`` -- the bounds the spokes are about to
    restore and publish.

    Read by the resumed hub, which first hears from its spokes only after
    its first resumed solve. It reads the spokes' own files, not a number of
    its own, because only a bound a spoke holds a solution for may be
    credited to it: the credited spoke writes the solution at the end of the
    run, and the hub's last word can be newer than the spoke's file (a spoke
    publishes an improvement before it writes it, and a failed write is not
    retried).

    A spoke with several ranks keeps its incumbent only if every rank's file
    is there and all are from one write (``agree_spoke_restore``), and then
    takes rank 0's inner bound on every rank, so the same rule decides here.
    How many ranks wrote is read from rank 0's file. A file that cannot be
    read is skipped rather than refused here; the spoke that owns it raises
    on it.
    """
    names = [d["spcomm_class"].__name__ for d in communicators]
    found = []
    for strata_rank, name in enumerate(names):
        if strata_rank == 0:
            continue
        ordinal = names[:strata_rank].count(name)

        def read(rank):
            path = os.path.join(ckpt_dir, SPOKES_SUBDIR,
                                _spoke_filename(name, ordinal, rank))
            if not os.path.exists(path):
                return None
            with open(path, "rb") as f:
                return pickle.load(f)

        # Anything that goes wrong with one spoke's files skips that spoke.
        # Raising here would stop hub rank 0 alone, and the other hub ranks
        # would wait for it forever.
        try:
            first = read(0)
            if first is None:
                continue
            states = [first] + [read(rank) for rank in
                                range(1, int(first["geometry"]["n_proc"]))]
            states = [s if s is not None
                      and s.get("format_version") == FORMAT_VERSION
                      and s.get("kind") == "spoke-incumbent" else None
                      for s in states]
            present = [s for s in states if s is not None]
            verdict, _ = _one_write_verdict(
                len(present), len(states),
                [XHAT_WRITE_KEY(s) for s in present])
            if verdict != "agreed":
                continue
            bound = float(first["best_inner_bound"])
        except Exception:
            continue
        if math.isfinite(bound):
            found.append((strata_rank, bound))
    return found


def restore_spoke_incumbent(opt, state):
    """Rebuild ``best_solution_cache`` on this spoke's models from a loaded
    state, by variable name. Returns the incumbent objective value.

    Every variable in the file must still exist on the model: a name that no
    longer resolves means the file describes a different model, and a
    partially restored incumbent is a solution that was never feasible for
    anything. The structural fingerprint should have caught that already, so
    reaching the error here means a model changed without its configuration
    changing.
    """
    import pyomo.environ as pyo

    for sname, s in opt.local_scenarios.items():
        entry = state["solutions"][sname]
        by_name = entry["values"]
        model_vars = {
            var.name: var for var in s.component_data_objects(pyo.Var)
        }
        missing = set(model_vars) - set(by_name)
        unknown = set(by_name) - set(model_vars)
        if missing or unknown:
            raise CheckpointMismatch(
                f"The checkpointed incumbent for scenario '{sname}' is not "
                f"a complete model snapshot: it is missing {len(missing)} "
                f"model variable(s) and names {len(unknown)} unknown one(s) "
                f"(e.g. missing {sorted(missing)[:3]}, unknown "
                f"{sorted(unknown)[:3]}), so it cannot be restored."
            )
        cache = pyo.ComponentMap()
        for name, value in by_name.items():
            cache[model_vars[name]] = value
        if entry["inner_bound"] is None or \
                not math.isfinite(entry["inner_bound"]):
            # Written before the write path refused this, or hand-edited.
            # Adopting it would put NaN or inf into the BEST_XHAT buffer, so
            # the incumbent is unusable however it got here.
            raise CheckpointMismatch(
                f"the checkpointed incumbent for scenario '{sname}' has no "
                f"finite objective recorded with it "
                f"({entry['inner_bound']}), so it cannot be restored: the "
                f"objective is published beside the values and is read "
                f"downstream as a real number."
            )
        s._mpisppy_data.best_solution_cache = cache
        # Both, and to the same number. The first is the one that goes back
        # out: send_best_xhat publishes the objective snapshotted with the
        # values. The second leaves this scenario reading as it did when the
        # incumbent was found, which is what it means before this spoke has
        # solved anything of its own.
        s._mpisppy_data.best_solution_inner_bound = entry["inner_bound"]
        s._mpisppy_data.inner_bound = entry["inner_bound"]

    opt.best_solution_obj_val = state["best_solution_obj_val"]
    return state["best_solution_obj_val"]


# ---------------------------------------------------------------------------
# The dual cylinders' own PH state.
#
# A cylinder that runs PH to produce duals -- relaxed_ph, ph_dual -- keeps the
# state that matters on its own models, and none of it is in the hub's
# checkpoint: the hub dills its scenarios, not theirs. So a resumed wheel that
# restored the hub perfectly still handed it a cylinder starting from W = 0,
# which under --ph-primal-hub is where the hub's own W comes from.
#
# What is written is W and the nonanticipative values, per scenario, and not
# the models: W is what the next iteration's Update_W adds to, and the values
# are what its Compute_Xbar averages. Everything else the cylinder needs it
# rebuilds every run -- rho from the rho setter, xbar from the values, the
# prox terms from PH_Prep -- so carrying it would be carrying a copy of a
# derivation. That keeps this file small enough to write at every iteration,
# which is what a cylinder nobody synchronizes with needs.
# ---------------------------------------------------------------------------

def dual_spoke_state(opt, cylinder, ordinal, generation, write_epoch,
                     class_count=None, wbar=None):
    """The dict written by ``write_dual_spoke_state``.

    W is keyed by ``(ndn, i)`` and the values by variable name -- the two
    keyings the rest of this module uses, and neither of them an identity.
    ``wbar`` is E[W] per node at the time of the write (``Wbar_by_node``),
    which the restore checks the restored weights reproduce.
    """
    if not isinstance(write_epoch, str) or not write_epoch:
        raise ValueError("A dual-spoke write requires a nonempty write identity")
    require_checkpoint_structural_cfg(opt.options)
    duals = {}
    for sname, s in opt.local_scenarios.items():
        model = s._mpisppy_model
        nonants = s._mpisppy_data.nonant_indices
        duals[sname] = {
            "W": {ndn_i: _as_float_or_none(model.W[ndn_i]._value)
                  for ndn_i in nonants},
            "values": {var.name: var._value for var in nonants.values()},
        }
    return {
        "format_version": FORMAT_VERSION,
        "kind": "dual-spoke-ph-state",
        "cylinder": str(cylinder),
        "ordinal": int(ordinal),
        # As for the incumbent file: the ordinal means the same thing only
        # while the wheel carries the same number of cylinders of this
        # class. Recorded so a resume can say when it does not.
        "class_count": None if class_count is None else int(class_count),
        "rank": int(opt.cylinder_rank),
        # This cylinder's own iteration count, which is not the hub's: the
        # wheel does not march the cylinders in step and this is not an
        # attempt to. It is paired with write_epoch to ensure all ranks
        # restore weights from the same write run and iteration.
        "generation": int(generation),
        "write_epoch": write_epoch,
        "geometry": geometry(opt),
        "structural_fingerprint": structural_fingerprint(opt.options),
        "Wbar": None if wbar is None else {
            ndn: [float(v) for v in values] for ndn, values in wbar.items()},
        "duals": duals,
    }


def write_dual_spoke_state(opt, ckpt_dir, cylinder, ordinal, generation,
                           write_epoch, class_count=None, wbar=None):
    """Write this cylinder's PH state, latest-wins. Returns the path.

    Temp-then-rename like every other file here, so a kill mid-write leaves
    the previous iteration's state intact rather than a truncated file.
    """
    state = dual_spoke_state(opt, cylinder, ordinal, generation, write_epoch,
                             class_count=class_count, wbar=wbar)
    spokes_dir = os.path.join(ckpt_dir, SPOKES_SUBDIR)
    os.makedirs(spokes_dir, exist_ok=True)
    path = os.path.join(
        spokes_dir,
        _spoke_filename(cylinder, ordinal, opt.cylinder_rank),
    )
    _atomic_write_bytes(path, lambda f: pickle.dump(state, f))
    _fsync_dir(spokes_dir)
    return path


def load_dual_spoke_state(opt, ckpt_dir, cylinder, ordinal):
    """Read this cylinder's PH state, or None if it is not there.

    Missing is normal -- the run being resumed may not have reached this
    cylinder's first completed iteration -- and present but wrong is an
    error, as everywhere else here.
    """
    require_checkpoint_structural_cfg(opt.options)
    path = os.path.join(
        ckpt_dir, SPOKES_SUBDIR,
        _spoke_filename(cylinder, ordinal, opt.cylinder_rank),
    )
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        state = pickle.load(f)

    if state.get("format_version") != FORMAT_VERSION:
        raise CheckpointMismatch(
            f"The cylinder state file '{path}' has format version "
            f"{state.get('format_version')}, but this mpi-sppy writes "
            f"version {FORMAT_VERSION}."
        )
    if state.get("kind") != "dual-spoke-ph-state":
        # Spoke files are named for their cylinder class, so this can only
        # happen if a class changed what it writes between the two runs.
        raise CheckpointMismatch(
            f"'{path}' holds {state.get('kind')!r}, not this cylinder's PH "
            f"state."
        )
    if state.get("structural_fingerprint") != structural_fingerprint(opt.options):
        raise CheckpointMismatch(
            f"The cylinder state file '{path}' was written by a run "
            f"configured differently from this one, so its dual weights do "
            f"not belong to this model."
        )
    # Checked here, at the load, so that a file without one is refused by
    # name rather than as a bare KeyError from the restore that reads it.
    missing = [key for key in ("generation", "Wbar") if key not in state]
    if missing:
        raise CheckpointMismatch(
            f"The cylinder state file '{path}' has no "
            f"{' or '.join(missing)}, so it was not written by this mpi-sppy "
            f"or has been damaged."
        )
    if "write_epoch" in state and (not isinstance(state["write_epoch"], str)
                                    or not state["write_epoch"]):
        raise CheckpointMismatch(
            f"The cylinder state file '{path}' has an invalid write identity."
        )
    if (state.get("cylinder"), state.get("ordinal"), state.get("rank")) != (
            str(cylinder), int(ordinal), int(opt.cylinder_rank)):
        raise CheckpointMismatch(
            f"The cylinder state file '{path}' does not belong to this "
            "cylinder.")
    if state.get("geometry") != geometry(opt):
        raise CheckpointMismatch(
            f"The cylinder state file '{path}' has checkpoint geometry that "
            "does not match this cylinder's rank layout."
        )
    return state


def _is_finite_number(value):
    """True for an int or float (numpy's included) that is not NaN or inf."""
    return (isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(value))


def restore_dual_spoke_state(opt, state):
    """Put W and the nonanticipative values back on this cylinder's models.

    A W entry or a nonant value that no longer resolves is an error rather
    than a skipped key: a cylinder carrying half the study's duals and half of
    zero is not a continuation of anything, and it would show up only as a
    bound that quietly stopped improving. The nonant values matter as much as
    W: the next xbar is computed from them, and a missing one would be left at
    the throwaway Iter0 solve's value.

    So is a weight or a value that is not a finite number. A None weight
    raises in the Wbar_by_node that follows, on this rank alone and outside
    any agreement, while the other ranks wait in its allreduce. A string
    either raises there too or is read there as a number and fails later, in
    Update_W. A NaN nonant passes the E[W] check, which looks only at W, and
    then becomes the next xbar.

    Every scenario is checked before any model is changed, so a refusal leaves
    the models as Iter0 left them rather than partly restored.
    """
    for sname, s in opt.local_scenarios.items():
        entry = state["duals"][sname]
        nonants = s._mpisppy_data.nonant_indices
        missing_w = [ndn_i for ndn_i in nonants if ndn_i not in entry["W"]]
        missing_x = [var.name for var in nonants.values()
                     if var.name not in entry["values"]]
        if missing_w or missing_x:
            raise CheckpointMismatch(
                f"The checkpointed dual state for scenario '{sname}' is "
                f"missing {len(missing_w)} dual weight(s) and "
                f"{len(missing_x)} nonant value(s) this model has (e.g. "
                f"{(missing_w + missing_x)[:3]}), so it cannot be restored."
            )
        bad = [(ndn_i, entry["W"][ndn_i]) for ndn_i in nonants
               if not _is_finite_number(entry["W"][ndn_i])]
        bad += [(var.name, entry["values"][var.name])
                for var in nonants.values()
                if not _is_finite_number(entry["values"][var.name])]
        if bad:
            raise CheckpointMismatch(
                f"The checkpointed dual state for scenario '{sname}' has "
                f"{len(bad)} dual weight(s) or nonant value(s) that are not "
                f"finite numbers (e.g. {bad[:3]}), so it cannot be restored."
            )
    for sname, s in opt.local_scenarios.items():
        entry = state["duals"][sname]
        model = s._mpisppy_model
        saved_w, by_name = entry["W"], entry["values"]
        for ndn_i, var in s._mpisppy_data.nonant_indices.items():
            model.W[ndn_i]._value = saved_w[ndn_i]
            var._value = by_name[var.name]


#: How far the restored E[W] may stray from the E[W] recorded with the file,
#: as a fraction of the size of the weights being summed (E[|W|]). The two are
#: sums of the same numbers with the same probabilities, so in practice they
#: agree exactly; the tolerance is only there so that a difference in the
#: order of accumulation is never a refusal. A file that was edited, or
#: probabilities that changed, are off by a visible fraction of the weights.
RESTORED_WBAR_RTOL = 1e-6


def require_restored_duals_match_their_file(opt, cylinder, generation,
                                            recorded, bars, sizes):
    """Refuse restored dual weights that do not reproduce the file's E[W].

    The rest of this cylinder's restore checks that the file describes this
    model: its format, its fingerprint, this rank's scenario names, that
    every nonant has a weight, and that every rank restored the same
    iteration. None of that looks at the numbers, and the numbers become
    another cylinder's Lagrangian bound, which the hub keeps as best-so-far.
    This recomputes E[W] per node from the restored models and compares it
    with the E[W] the writing run computed from the same weights. That
    catches an edited file and changed scenario probabilities. It compares a
    sum per variable, so W exchanged between two scenarios of equal
    probability passes.

    Not a check that E[W] is zero. PH keeps it at zero only when rho is the
    same in every scenario (see ``phbase.Wbar_by_node``); with a
    scenario-dependent rho the weights an uninterrupted run publishes do not
    sum to zero either, and a resume has no business refusing what the run it
    continues was doing. What a resume must not do is publish weights other
    than the ones it was checkpointed with.

    ``bars`` and ``sizes`` are ``phbase.Wbar_by_node`` and
    ``phbase.W_magnitude_by_node`` of the restored models. They are passed in
    because each sum spans the scenarios of a tree node and so the ranks of
    the cylinder: the caller computes them on every rank first. That makes
    them the same on every rank, but ``recorded`` is not -- each rank read it
    from its own file -- so a damaged file can make this refuse on that rank
    alone. Per rank, then, and a step the caller has to agree on: see
    :func:`run_agreed`.
    """
    if recorded is None or set(recorded) != set(bars):
        raise CheckpointMismatch(
            f"The dual weights file {cylinder} wrote at its iteration "
            f"{generation} does not record E[W] for this model's tree nodes "
            f"({sorted(bars)}), so its weights cannot be checked. Remove the "
            f"file to start this cylinder from W = 0.")

    worst = None
    for ndn, Wbars in bars.items():
        for i, Wbar in enumerate(Wbars):
            want = float(recorded[ndn][i])
            size = float(sizes[ndn][i])
            # A NaN anywhere here makes the excess NaN, and every comparison
            # with NaN is false, so it would pass the check below unseen.
            if not all(math.isfinite(x) for x in (float(Wbar), want, size)):
                raise CheckpointMismatch(
                    f"The dual weights {cylinder} wrote at its iteration "
                    f"{generation} are not finite at node {ndn}, index {i} "
                    f"(restored E[W] {float(Wbar)}, recorded {want}, size "
                    f"{size}), so they cannot be checked. Remove the file to "
                    f"start this cylinder from W = 0.")
            tolerance = opt.E1_tolerance + RESTORED_WBAR_RTOL * size
            excess = abs(float(Wbar) - want) / tolerance
            if worst is None or excess > worst[0]:
                worst = (excess, float(Wbar), want, size, tolerance, ndn, i)

    if worst is not None and worst[0] > 1.0:
        _, Wbar, want, size, tolerance, ndn, i = worst
        raise CheckpointMismatch(
            f"The dual weights restored for {cylinder} from the checkpoint "
            f"it wrote at its iteration {generation} do not reproduce the "
            f"E[W] that file recorded: at '{_nonant_name(opt, ndn, i)}' the "
            f"probability-weighted sum of the restored weights is "
            f"{Wbar:.6e}, and the file recorded {want:.6e}, against a "
            f"tolerance of {tolerance:.3e} -- E1_tolerance "
            f"({opt.E1_tolerance:.1e}) plus {RESTORED_WBAR_RTOL:.0e} of the "
            f"size of the weights summed there ({size:.6e}). The file has "
            f"been changed, or the scenario probabilities have. A Lagrangian "
            f"bound computed from them would not be the study's, and the hub "
            f"keeps the best bound it is ever told, so this refuses rather "
            f"than resumes. Remove the file to start this cylinder from "
            f"W = 0."
        )


def _nonant_name(opt, ndn, i):
    """The name of nonant ``i`` of node ``ndn``, for an error message."""
    for s in opt.local_scenarios.values():
        for node in s._mpisppy_node_list:
            if node.name == ndn and i < len(node.nonant_vardata_list):
                return node.nonant_vardata_list[i].name
    return f"{ndn}[{i}]"
