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
import math
import os
import pickle
import re
import shutil
import hashlib

import numpy as np
from pyomo.common.dependencies import attempt_import

import mpisppy.MPI as MPI
import mpisppy.utils.pickle_bundle as pickle_bundle

dill, dill_available = attempt_import("dill")

# Bump when the on-disk layout changes in a way older readers cannot handle.
#: Bumped when the on-disk shape changes. Version 2 replaced the spoke
#: incumbent file's per-scenario ``inner_bound`` -- read live at write time,
#: so it could describe a later solve than the values beside it -- with the
#: objective cached when the incumbent itself was.
FORMAT_VERSION = 2

DILL_RELOAD_BACKEND = "dill-reload"
LEAF_BACKEND = "leaf"

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
    # and --stop-at-iteration-number sizes the study it belongs to.
    "max_iterations", "stop_at_iteration_number", "time_limit",
    "intra_hub_conv_thresh", "rel_gap", "abs_gap", "max_stalled_iters",
    # Checkpoint plumbing itself. How often a checkpoint is written is a
    # cadence knob like the iteration limit: it changes what a stop costs, not
    # what problem the checkpoint describes.
    "checkpoint_dir", "checkpoint_backend", "checkpoint_every_iterations",
    "checkpoint_before_seconds", "resume_from",
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
    "slammax", "slammin",
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


def _model_filename(rank, sname):
    return f"hub_rank_{rank:04d}_scen_{sanitize_for_filename(sname)}.dill"


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


def _atomic_write_bytes(path, write_callback):
    """Write via a temp file in the same directory, then rename into place."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        write_callback(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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

    ``key`` names the field that identifies the write: the objective of the
    cached solution for an xhat spoke, the iteration for a dual cylinder.

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
    values = comm.allgather(state.get(key))
    return _one_write_verdict(have, size, values)


#: The field that identifies an xhat spoke's write: the objective of the
#: cached solution.
XHAT_WRITE_KEY = "best_solution_obj_val"


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
    if backend != DILL_RELOAD_BACKEND:
        raise RuntimeError(
            f"--checkpoint-backend '{backend}' is not implemented. "
            f"The only supported backend is '{DILL_RELOAD_BACKEND}'."
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
            f"(objectives {detail}), so none of them restores one: the "
            f"files were written at different passes, and half of one xhat "
            f"beside half of another is not a solution this study ever found")
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
    """The same agreement for a dual cylinder's W, keyed on the iteration.

    Nothing here is broadcast: W is per scenario and so per rank. What must
    hold is that every rank restores the W of the *same* iteration. Ranks
    that restore W from different iterations hand ``Compute_Xbar`` an
    allreduce over values belonging to two different points of the run --
    and under ``--ph-primal-hub`` that blended W is what the hub's W is
    built from. No error and no warning, like the xhat case.

    Returns ``(state, warning)``. Collective; see :func:`agree_one_write`.
    """
    verdict, detail = agree_one_write(opt, state, "generation")
    if verdict == "none":
        return None, None
    if verdict == "partial":
        have, size = detail
        return None, (
            f"only {have} of {size} ranks of this cylinder have checkpointed "
            f"dual weights, so none of them restores any: a W taken from "
            f"some ranks and not others is not the W this study had")
    if verdict == "differ":
        return None, (
            f"the ranks of this cylinder checkpointed dual weights from "
            f"different iterations ({detail}), so none of them restores any: "
            f"a W half from one iteration and half from another is not the W "
            f"this study had")
    return state, None


def require_dill(backend):
    if backend == DILL_RELOAD_BACKEND and not dill_available:
        raise RuntimeError(
            "The '{}' checkpoint backend requires dill, which is not "
            "installed. Install the optional dependencies with "
            "'pip install mpi-sppy[extras]' (or 'pip install dill'), or "
            "choose a different --checkpoint-backend.".format(backend)
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
    failure = None
    for sname, s in opt.local_scenarios.items():
        solver_plugin = getattr(s, "_solver_plugin", None)
        if solver_plugin is not None:
            del s._solver_plugin
        try:
            dill.dumps(s)
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
    failing_rank = _first_failing_rank(comm, int(opt.cylinder_rank),
                                       failure is not None)
    if failing_rank is None:
        return
    if failure is not None:
        raise failure
    raise RuntimeError(
        f"Checkpointing is enabled, but no checkpoint could ever be written: "
        f"rank {failing_rank} has a scenario model that cannot be "
        f"serialized. See that rank's message for which one and why."
    )


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
# The contract is two no-op methods on ``Extension`` and ``Converger``. This
# module aggregates them, keyed by class name, because names are what survives
# a resume: object identity does not, and attach order is not stable enough to
# index by. Name keying is also what lets a resume with a *different* extension
# set do something sensible -- entries with no matching extension are reported
# rather than silently dropped.
###############################################################################


def _extension_objects(opt):
    """Yield ``(class name, extension)`` for every attached extension.

    ``MultiExtension`` is a container, not an extension with state of its own,
    so it is flattened away -- and flattened recursively, since nothing stops
    one from holding another. Two extensions of the same class would collide
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
        if extdict:
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


def extensions_without_a_state_contract(opt):
    """Yield the name of every attached extension that answers neither question.

    An extension says what happens to its state across a resume in one of two
    ways: it implements ``checkpoint_state``, or it declares
    ``checkpoint_stateless``. Answering neither is not the same as having no
    state -- it is nobody having decided -- and the two used to look identical
    from here, because the base-class hook returns None and this function only
    ever reported checkpoint entries with no extension. The reverse direction,
    an extension with no entry, was silent, which is how a shipped rho updater
    with exactly the state this phase carries for its neighbours went unnoticed.

    ``checkpoint_state`` is matched with inheritance, since a subclass of an
    extension that implements it inherits a real implementation.
    ``checkpoint_stateless`` is matched without, so a subclass that adds state
    to a stateless parent is named instead of covered by it.
    """
    from mpisppy.extensions.extension import Extension
    for name, ext in _extension_objects(opt):
        cls = type(ext)
        if cls.checkpoint_state is not Extension.checkpoint_state:
            continue
        if cls.__dict__.get("checkpoint_stateless", False):
            continue
        yield name


def restore_extension_state(opt, state):
    """Hand each extension its own state back. Returns a list of warnings.

    Warnings rather than errors: a resume with a different extension set is
    something a user may legitimately do (the checkpoint's *hub iterate* is
    still valid), so it should say clearly what it could not restore instead
    of refusing the whole checkpoint. The caller prints them.

    Reported in both directions. A checkpoint entry with no extension means
    state that could not be handed to anybody. An extension that has neither
    implemented ``checkpoint_state`` nor declared ``checkpoint_stateless``
    means state nobody has decided about -- which is not the same as an
    extension added since the checkpoint, and used to be indistinguishable
    from one.
    """
    warnings = []
    missing = sorted(extensions_without_a_state_contract(opt))
    if missing:
        warnings.append(
            f"these attached extensions carry no state across the resume: "
            f"{', '.join(missing)}. Each keeps whatever it had at the start of "
            f"a fresh run, so one that decides what to do next from what it "
            f"did earlier -- a history, a counter, a record of what it already "
            f"changed -- will not retrace an uninterrupted run. Implement "
            f"checkpoint_state and restore_state on it, or set "
            f"checkpoint_stateless = True to say it has nothing to carry.")
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


def converger_state_is_carried(opt, state):
    """Whether the run's converger (if any) had its state restored.

    The resume warns when a converger starts fresh, because one that
    accumulates history can then terminate the run at a different iteration
    than an uninterrupted run would. That warning is right for a converger
    that does not implement the contract and wrong for one that does, so the
    resume asks here rather than warning unconditionally.
    """
    convobject = getattr(opt, "convobject", None)
    if convobject is None:
        return True
    if type(convobject).__dict__.get("checkpoint_stateless", False):
        return True
    saved = (state or {}).get("converger")
    return (saved is not None
            and saved["class"] == type(convobject).__name__)


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
    # Checked before anything rank-local: it depends only on the backend name
    # and whether dill is importable, so every rank reaches the same verdict
    # and there is nothing to agree on.
    require_dill(backend)

    comm = _cylinder_comm(opt)
    rank = int(opt.cylinder_rank)
    is_publisher = rank == 0
    hub_dir = os.path.join(ckpt_dir, HUB_SUBDIR)
    final_dir = os.path.join(hub_dir, _generation_dirname(generation))
    staging_dir = f"{final_dir}.tmp"

    # Rank 0's directory preparation is guarded like every other rank-local
    # step, and for the same reason: if it raised straight out of here, rank 0
    # would never reach the barrier below and every other rank would wait at
    # it for the rest of the job.
    failure = None
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
    # No rank may write into the staging directory until rank 0 has cleared
    # and recreated it, or its files are deleted out from under it.
    _barrier(comm)

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
        model_files = _write_models(opt, staging_dir, rank, backend)
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
            # How long a PH iteration of this run takes, so a resume can seed
            # --checkpoint-before-seconds with a measurement instead of with
            # its own iteration 0, which on a resume reloads models rather
            # than solving them. This is the iteration *before* the one being
            # written -- the current one is not over until after this write --
            # which is the recent iteration the trigger wants either way.
            "last_iteration_seconds": _as_float_or_none(
                getattr(opt, "_last_iteration_seconds", None)),
            "best_outer_bound": _hub_best_outer_bound(opt),
        }
        _atomic_write_bytes(
            os.path.join(staging_dir, _leaf_filename(rank)),
            lambda f: pickle.dump(leaf, f),
        )
        _fsync_dir(staging_dir)
    except Exception as exc:
        failure = exc

    # Collective, and therefore also the barrier that says every rank has
    # finished writing. Nothing below may run before it.
    failing_rank = _first_failing_rank(comm, rank, failure is not None)
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
            _publish_generation(opt, ckpt_dir, hub_dir, final_dir,
                                staging_dir, generation, backend)
        except Exception:
            # The manifest still names the previous generation (or nothing),
            # so everything else is garbage; give the disk back now rather
            # than at the next checkpoint point.
            _sweep_uncommitted_generations(ckpt_dir, hub_dir)
            raise

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


def _publish_generation(opt, ckpt_dir, hub_dir, final_dir, staging_dir,
                        generation, backend):
    """Commit the staged generation. Rank 0 only, once every rank has written.

    Publishing order matters. The manifest is the commit point, so the
    generation it currently names must stay on disk and intact until the
    replacement is fully published -- otherwise a kill in between destroys the
    only checkpoint. Stage under a name nothing points at, publish, then sweep.
    Writing the same generation number twice therefore lands in a scratch
    directory first rather than deleting the live one.
    """
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
    _fsync_dir(hub_dir)

    _publish_manifest(ckpt_dir, {
        "format_version": FORMAT_VERSION,
        "backend": backend,
        "generation": int(generation),
        "n_proc": int(opt.n_proc),
        "structural_fingerprint": structural_fingerprint(opt.options),
    })

    # Sweep everything the manifest does not name, rather than only the
    # generation the previous manifest did. A kill between any two steps above
    # can leave a directory behind, and deleting just the known predecessor
    # would let those accumulate for the life of the run.
    _sweep_stale_generations(hub_dir, keep=int(generation))


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
    _sweep_stale_generations(hub_dir, keep)


def _sweep_stale_generations(hub_dir, keep):
    """Delete every generation directory except the one the manifest names.

    ``keep=None`` means the manifest names none. When the named generation's
    directory is absent, its ``.retiring`` copy is kept: that is the state a
    kill between the two publishing renames leaves, and ``load_checkpoint``
    reads the retired copy.
    """
    keep_names = set()
    if keep is not None:
        keep_name = _generation_dirname(keep)
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


#: The hub leaf keys a resume reads by name. The load refuses a leaf missing
#: any of them, so the refusal is agreed across the hub's ranks.
LEAF_KEYS_READ_ON_RESUME = (
    "generation", "initially_fixed_nonants", "trivial_bound",
    "best_bound_obj_val", "best_outer_bound", "best_solution_obj_val",
)


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
    if not os.path.isdir(gen_dir) and os.path.isdir(f"{gen_dir}.retiring"):
        # A write of this same generation was interrupted between retiring the
        # old copy and moving the new one into place. The retired copy is the
        # generation the manifest names, intact.
        gen_dir = f"{gen_dir}.retiring"
    rank = int(opt.cylinder_rank)

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
    missing = [key for key in LEAF_KEYS_READ_ON_RESUME if key not in leaf]
    if missing:
        raise CheckpointMismatch(
            f"The checkpoint state '{leaf_path}' has no "
            f"{', '.join(missing)}, so it was not written by this mpi-sppy "
            f"or has been damaged."
        )

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


def spoke_incumbent_state(opt, cylinder, ordinal, best_inner_bound=None,
                          loop_state=None, class_count=None,
                          extension_state=GATHER_EXTENSION_STATE):
    """The dict written by ``write_spoke_incumbent``, or None if there is
    nothing to write yet. ``_cache_best_solution`` caches every scenario at
    once, so one scenario without a cache means none has one."""
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
            "values": {var.name: value for var, value in cache.items()},
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
                          extension_state=GATHER_EXTENSION_STATE):
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
                                  extension_state=extension_state)
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
    have = sorted(opt.local_scenarios.keys())
    want = state["geometry"]["scenario_names"]
    if have != want:
        raise CheckpointMismatch(
            f"Rank {opt.cylinder_rank} of this spoke owns scenarios {have}, "
            f"but '{path}' was written with {want} on that rank."
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
                [s[XHAT_WRITE_KEY] for s in present])
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
        cache = pyo.ComponentMap()
        found = 0
        for var in s.component_data_objects(pyo.Var):
            if var.name not in by_name:
                continue
            cache[var] = by_name[var.name]
            found += 1
        if found != len(by_name):
            missing = set(by_name) - {
                var.name for var in s.component_data_objects(pyo.Var)
            }
            raise CheckpointMismatch(
                f"The checkpointed incumbent for scenario '{sname}' names "
                f"{len(missing)} variable(s) this model does not have "
                f"(e.g. {sorted(missing)[:3]}), so it cannot be restored."
            )
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

def dual_spoke_state(opt, cylinder, ordinal, generation, class_count=None,
                     wbar=None):
    """The dict written by ``write_dual_spoke_state``.

    W is keyed by ``(ndn, i)`` and the values by variable name -- the two
    keyings the rest of this module uses, and neither of them an identity.
    ``wbar`` is E[W] per node at the time of the write (``Wbar_by_node``),
    which the restore checks the restored weights reproduce.
    """
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
        # attempt to. It is recorded so a log can say how far the cylinder
        # had got, and it is not compared against anything on the way back in.
        "generation": int(generation),
        "geometry": geometry(opt),
        "structural_fingerprint": structural_fingerprint(opt.options),
        "Wbar": None if wbar is None else {
            ndn: [float(v) for v in values] for ndn, values in wbar.items()},
        "duals": duals,
    }


def write_dual_spoke_state(opt, ckpt_dir, cylinder, ordinal, generation,
                           class_count=None, wbar=None):
    """Write this cylinder's PH state, latest-wins. Returns the path.

    Temp-then-rename like every other file here, so a kill mid-write leaves
    the previous iteration's state intact rather than a truncated file.
    """
    state = dual_spoke_state(opt, cylinder, ordinal, generation,
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
    have = sorted(opt.local_scenarios.keys())
    want = state["geometry"]["scenario_names"]
    if have != want:
        raise CheckpointMismatch(
            f"Rank {opt.cylinder_rank} of this cylinder owns scenarios "
            f"{have}, but '{path}' was written with {want} on that rank."
        )
    return state


def restore_dual_spoke_state(opt, state):
    """Put W and the nonanticipative values back on this cylinder's models.

    A W entry that no longer resolves is an error rather than a skipped key:
    a cylinder carrying half the study's duals and half of zero is not a
    continuation of anything, and it would show up only as a bound that
    quietly stopped improving.
    """
    for sname, s in opt.local_scenarios.items():
        entry = state["duals"][sname]
        model = s._mpisppy_model
        nonants = s._mpisppy_data.nonant_indices
        saved_w = entry["W"]
        missing = [ndn_i for ndn_i in nonants if ndn_i not in saved_w]
        if missing:
            raise CheckpointMismatch(
                f"The checkpointed dual weights for scenario '{sname}' are "
                f"missing {len(missing)} nonanticipative variable(s) this "
                f"model has (e.g. {missing[:3]}), so they cannot be restored."
            )
        by_name = entry["values"]
        for ndn_i, var in nonants.items():
            model.W[ndn_i]._value = saved_w[ndn_i]
            if var.name in by_name:
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
