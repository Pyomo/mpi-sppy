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

The ``dill-reload`` backend dills each mid-run scenario model, which brings back
the dual weights, rho, nonant values and fixedness, the recourse values that
serve as a MIP warm start, and the proximal-approximation cuts, all mutually
consistent. Everything that does *not* live on a scenario model -- the global
iteration counter, bounds, and the initially-fixed-nonant baseline -- is written
alongside as a small pickle of plain data.
"""

import json
import os
import pickle
import re
import shutil
import hashlib

from pyomo.common.dependencies import attempt_import

import mpisppy.utils.pickle_bundle as pickle_bundle

dill, dill_available = attempt_import("dill")

# Bump when the on-disk layout changes in a way older readers cannot handle.
FORMAT_VERSION = 1

DILL_RELOAD_BACKEND = "dill-reload"
LEAF_BACKEND = "leaf"

MANIFEST_NAME = "manifest.json"
HUB_SUBDIR = "hub"

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
    # How long to run. Resuming with a different budget is the point.
    "max_iterations", "time_limit", "intra_hub_conv_thresh", "rel_gap",
    "abs_gap", "max_stalled_iters",
    # Checkpoint plumbing itself.
    "checkpoint_dir", "checkpoint_at_termination", "checkpoint_backend",
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
    "iter0_solver_options", "iterk_solver_options", "presolve",
})


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


def sanitize_for_filename(name):
    """Make a scenario name safe to embed in a file name.

    Never use ``sputils.extract_num`` here: it scrapes trailing digits, which
    are not unique for ADMM's wrapped scenario names.
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))


def _generation_dirname(generation):
    return f"gen_{generation:04d}"


def _leaf_filename(rank):
    return f"hub_rank_{rank:04d}.pkl"


def _model_filename(rank, sname):
    return f"hub_rank_{rank:04d}_scen_{sanitize_for_filename(sname)}.dill"


def _atomic_write_bytes(path, write_callback):
    """Write via a temp file in the same directory, then rename into place."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        write_callback(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def require_dill(backend):
    if backend == DILL_RELOAD_BACKEND and not dill_available:
        raise RuntimeError(
            "The '{}' checkpoint backend requires dill, which is not "
            "installed. Install the optional dependencies with "
            "'pip install mpi-sppy[extras]' (or 'pip install dill'), or "
            "choose a different --checkpoint-backend.".format(backend)
        )




def probe_model_is_dillable(opt):
    """Serialize one local scenario to memory to prove checkpointing will work.

    Called once at setup. A run that only discovers at its terminal checkpoint
    -- possibly many hours in -- that its models cannot be dilled would lose
    exactly the state checkpointing exists to preserve, so this trades one
    model serialization up front for a failure that arrives immediately and
    says what to do about it. The probe runs at iteration 0, when the model is
    at its smallest (no accumulated prox-approximation cuts).
    """
    if not opt.local_scenarios:
        return
    sname, s = next(iter(opt.local_scenarios.items()))
    solver_plugin = getattr(s, "_solver_plugin", None)
    if solver_plugin is not None:
        del s._solver_plugin
    try:
        dill.dumps(s)
    except Exception as exc:
        raise RuntimeError(
            "Checkpointing is enabled, but no checkpoint could ever be "
            "written.\n\n"
            + pickle_bundle.describe_dill_failure(
                s, exc, what=f"scenario '{sname}'")
        ) from exc
    finally:
        if solver_plugin is not None:
            s._solver_plugin = solver_plugin


# The per-nonant Params that the pre-solve half of a PH iteration advances.
# Caching these at enditer is O(nonants) -- cheap next to a model dill -- and
# is enough to rewind an interrupted iteration to its last completed one.
COHERENT_ITERATE_PARAMS = ("W", "xbars", "xsqbars", "z")


def capture_iterate_params(opt):
    """Snapshot the iterate Params that Compute_Xbar/Update_W/Update_z advance.

    Returns a plain nested dict of floats -- no Pyomo objects -- so holding one
    across an iteration costs nothing worth measuring.
    """
    cache = {}
    for sname, s in opt.local_scenarios.items():
        per_scenario = {}
        for pname in COHERENT_ITERATE_PARAMS:
            param = getattr(s._mpisppy_model, pname, None)
            if param is None:
                continue
            per_scenario[pname] = {
                ndn_i: param[ndn_i]._value
                for ndn_i in s._mpisppy_data.nonant_indices
            }
        cache[sname] = per_scenario
    return cache


def restore_iterate_params(opt, cache):
    """Write a snapshot from capture_iterate_params back onto the models."""
    for sname, per_scenario in cache.items():
        s = opt.local_scenarios.get(sname)
        if s is None:
            continue
        for pname, values in per_scenario.items():
            param = getattr(s._mpisppy_model, pname, None)
            if param is None:
                continue
            for ndn_i, value in values.items():
                param[ndn_i]._value = value


def geometry(opt):
    """The rank layout a resume must reproduce (see design section 5.7)."""
    return {
        "n_proc": int(opt.n_proc),
        "rank": int(opt.cylinder_rank),
        "scenario_names": sorted(opt.local_scenarios.keys()),
    }


def initially_fixed_nonant_names(opt):
    """Names of the nonants that were already fixed when the run first started.

    This is the baseline ``_can_update_best_bound`` compares against, and it is
    the one piece of opt-object state that is keyed by variable *identity* --
    a ``ComponentSet`` of vardata belonging to models that a resume replaces.
    Recording it by name is what lets the resume rebuild it correctly; see
    design section 9, item 11, for what goes wrong otherwise.
    """
    baseline = getattr(opt, "_initial_fixed_varibles", None)
    if baseline is None:
        return []
    return sorted(v.name for v in baseline)


def write_checkpoint(opt, ckpt_dir, generation, backend=DILL_RELOAD_BACKEND):
    """Write and atomically publish one checkpoint generation.

    The rank writes its own files into a temporary generation directory, which
    is renamed into place; the manifest is then rewritten (itself
    temp-then-rename) to point at the new generation. That manifest flip is the
    single commit point, so a kill before it leaves the previous checkpoint
    intact and a kill after it leaves the new one. The prior generation is
    deleted once the manifest names its replacement.
    """
    require_dill(backend)

    rank = int(opt.cylinder_rank)
    hub_dir = os.path.join(ckpt_dir, HUB_SUBDIR)
    final_dir = os.path.join(hub_dir, _generation_dirname(generation))
    staging_dir = f"{final_dir}.tmp"

    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    try:
        model_files = _write_models(opt, staging_dir, rank, backend)
    except Exception as exc:
        # Leave no half-written generation behind; the previous checkpoint (if
        # any) stays published, since the manifest was never touched.
        shutil.rmtree(staging_dir, ignore_errors=True)
        if isinstance(exc, ValueError):
            raise
        first = next(iter(opt.local_scenarios.values()), None)
        detail = (pickle_bundle.describe_dill_failure(first, exc,
                                                      what="scenario model")
                  if first is not None
                  else f"{type(exc).__name__}: {exc}")
        raise RuntimeError(
            f"Failed to write the checkpoint to '{ckpt_dir}'. Any previously "
            f"published checkpoint is untouched.\n\n" + detail
        ) from exc

    leaf = {
        "format_version": FORMAT_VERSION,
        "backend": backend,
        "generation": int(generation),
        "geometry": geometry(opt),
        "structural_fingerprint": structural_fingerprint(opt.options),
        "model_files": model_files,
        "initially_fixed_nonants": initially_fixed_nonant_names(opt),
        "trivial_bound": _as_float_or_none(getattr(opt, "trivial_bound", None)),
        "best_bound_obj_val": _as_float_or_none(
            getattr(opt, "best_bound_obj_val", None)),
        "best_solution_obj_val": _as_float_or_none(
            getattr(opt, "best_solution_obj_val", None)),
    }
    _atomic_write_bytes(
        os.path.join(staging_dir, _leaf_filename(rank)),
        lambda f: pickle.dump(leaf, f),
    )

    if os.path.isdir(final_dir):
        shutil.rmtree(final_dir)
    os.replace(staging_dir, final_dir)

    previous = _read_manifest(ckpt_dir, missing_ok=True)
    _publish_manifest(ckpt_dir, {
        "format_version": FORMAT_VERSION,
        "backend": backend,
        "generation": int(generation),
        "n_proc": int(opt.n_proc),
        "structural_fingerprint": structural_fingerprint(opt.options),
    })

    # Exactly one committed generation is kept; retaining more is not
    # supported. Two exist only transiently, between the generation rename
    # above and this delete.
    if previous is not None and previous.get("generation") != int(generation):
        stale = os.path.join(hub_dir, _generation_dirname(previous["generation"]))
        if os.path.isdir(stale):
            shutil.rmtree(stale, ignore_errors=True)

    return final_dir


def _write_models(opt, staging_dir, rank, backend):
    """Dill each local scenario model into the staging directory."""
    if backend != DILL_RELOAD_BACKEND:
        raise ValueError(
            f"Unknown checkpoint backend '{backend}'. The only implemented "
            f"backend is '{DILL_RELOAD_BACKEND}'."
        )
    model_files = {}
    for sname, s in opt.local_scenarios.items():
        fname = _model_filename(rank, sname)
        # The solver plugin is a live C handle plus a license session; it
        # cannot be serialized and is rebuilt by _create_solvers on resume.
        solver_plugin = getattr(s, "_solver_plugin", None)
        if solver_plugin is not None:
            del s._solver_plugin
        try:
            _atomic_write_bytes(
                os.path.join(staging_dir, fname),
                lambda f, model=s: dill.dump(model, f),
            )
        finally:
            if solver_plugin is not None:
                s._solver_plugin = solver_plugin
        model_files[sname] = fname
    return model_files


def _as_float_or_none(value):
    return None if value is None else float(value)


def _publish_manifest(ckpt_dir, manifest):
    path = os.path.join(ckpt_dir, MANIFEST_NAME)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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


def load_checkpoint(opt, ckpt_dir):
    """Load this rank's checkpoint, refusing a mismatch with a clear error.

    Returns ``(leaf_state, {scenario_name: reloaded_model})``. The caller is
    responsible for splicing the models into the run (see
    ``PHBase._resume_from_checkpoint``).
    """
    manifest = _read_manifest(ckpt_dir)

    if manifest.get("format_version") != FORMAT_VERSION:
        raise CheckpointMismatch(
            f"Checkpoint in '{ckpt_dir}' has format version "
            f"{manifest.get('format_version')}, but this mpi-sppy writes and "
            f"reads version {FORMAT_VERSION}. Checkpoints are not portable "
            f"across format versions."
        )

    backend = manifest.get("backend")
    require_dill(backend)

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
    gen_dir = os.path.join(ckpt_dir, HUB_SUBDIR, _generation_dirname(generation))
    rank = int(opt.cylinder_rank)

    leaf_path = os.path.join(gen_dir, _leaf_filename(rank))
    if not os.path.exists(leaf_path):
        raise CheckpointMismatch(
            f"The checkpoint in '{ckpt_dir}' has no state for rank {rank} "
            f"(expected '{leaf_path}')."
        )
    with open(leaf_path, "rb") as f:
        leaf = pickle.load(f)

    have = sorted(opt.local_scenarios.keys())
    want = leaf["geometry"]["scenario_names"]
    if have != want:
        raise CheckpointMismatch(
            f"Rank {rank} now owns scenarios {have}, but the checkpoint in "
            f"'{ckpt_dir}' was written with {want} on that rank. Resuming "
            f"requires an identical scenario-to-rank distribution."
        )

    models = {}
    for sname, fname in leaf["model_files"].items():
        path = os.path.join(gen_dir, fname)
        if not os.path.exists(path):
            raise CheckpointMismatch(
                f"The checkpoint in '{ckpt_dir}' is missing the model file "
                f"'{fname}' for scenario '{sname}'."
            )
        with open(path, "rb") as f:
            models[sname] = dill.load(f)

    return leaf, models
