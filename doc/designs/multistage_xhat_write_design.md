# Design: Writing a multi-stage xhat (nonant tree) to a file

**Status:** Draft — up for discussion. Nothing implemented yet.
**Author:** dlw (captured with Claude Code assistance)
**Last updated:** 2026-06-14
**Branch:** `feasible-xhat-multistage-file`

## Where this fits

The larger goal is two-fold: (1) a **multi-stage** `feasible_xhat_creator`
contract, and (2) the ability to **read an xhat from a file** for any
number of stages. The runtime fixing path
(`SPOpt._fix_nonants` / `Xhat_Eval.fix_nonants_upto_stage`) already
consumes a full `{node_name: array}` cache across every tree node, and
`--xhat-from-file` already exists for the two-stage case
(`cylinders/xhatbase.py::_try_file_xhat`, reading via
`ciutils.read_xhat`). So the missing pieces are narrower than the docs
imply.

This note covers **only the write side**, as a first, self-contained,
green-on-its-own PR. We establish *and produce* the canonical on-disk
format that the later reader will consume. Reading, the `ciutils`
generalization, the multi-stage `feasible_xhat_creator` contract, and
any new CLI surface are explicitly **out of scope here** (see below).

Rationale for writing first: the reader must consume a concrete format,
and the most useful producer of that format — a real multi-stage
cylinders run's incumbent — does not exist today. Settling the format
by *writing* it first lets the reader phase be pure consumption.

## What mpi-sppy writes today

| Writer | Format | Names | Scope |
|---|---|---|---|
| `ciutils.write_xhat` / `writetxt_xhat` | `.npy` / `.txt`, bare ROOT vector | positional | 2-stage only; MMW round-trips it |
| `sputils.first_stage_nonant_writer` | CSV `var,value` | by name | ROOT only; cylinders `--solution-base-name.csv` |
| `sputils.first_stage_nonant_npy_serializer` | `.npy`, ROOT vector | positional | ROOT only; `--solution-base-name.npy` |
| `sputils.ef_nonants_csv` | CSV `Node, EF_VarName, Value` | by name (**EF-qualified**) | **whole tree**; EF `--solution-base-name.csv` (`generic/ef.py`) |
| `sputils.scenario_tree_solution_writer` | directory of per-scenario CSVs `var,value` | by name | whole tree, but **all** Vars+Expressions; `--solution-base-name_soldir` |

Two observations drive the design:

- **`ef_nonants_csv` is already a single-file, by-name, multi-stage
  nonant dump** — the only one in the tree. Its rows are deduped to one
  per `(node, var)` (it walks `ef.ref_vars`). It is the natural
  canonical format ("support what we already write"). Its only defects
  for round-trip use are EF-qualified variable names and no comment
  convention.
- **A cylinders run has no whole-tree, single-rank nonant cache.**
  `Hub.send_nonants` packs only each rank's *local* scenarios into a
  per-rank buffer; cross-cylinder exchange is rank-to-rank.
  `write_tree_solution` sidesteps assembly by having every rank write
  its own scenarios into a shared directory. A *single file* cannot be
  sharded that way, and one representative scenario only knows the nodes
  on its own root-to-leaf path. So a single-file multi-stage writer for
  cylinders needs the nonant tree **assembled on one rank**.

## Decisions taken (the two we discussed)

**(a) Reuse, via a shared serializer core.** Factor one serializer that
turns `{node -> [(local_name, value)]}` into the canonical CSV, and make
`ef_nonants_csv` a thin adapter onto it (rather than grow a second,
drifting CSV writer). This unifies a latent inconsistency: today EF
`--solution-base-name.csv` emits a whole-tree file while *cylinders*
`--solution-base-name.csv` emits ROOT-only.

  *Accepted cost:* `ef_nonants_csv`'s on-disk bytes change — EF-qualified
  names (`Scenario1.DevotedAcres[CORN]`) become **node-local**
  (`DevotedAcres[CORN]`), plus a `#`-comment header. One format-pinning
  test (`test_sputils.py::test_ef_nonants_csv_creates_file`) is updated;
  `generic/ef.py`'s EF `.csv` output changes shape. For nonants
  node-local names are strictly better (the scenario prefix is noise on
  shared vars and blocks reading the file back into a single scenario).

**(b) A new method, not a mode on `write_tree_solution`.** A single-file
nonant writer differs on three axes at once — parallelism
(rank-0-after-gather vs every-rank-writes), content (nonants vs all
Vars+Expressions), and shape (one file vs a directory). It gets its own
method, `write_tree_nonants`, sitting between `write_first_stage_solution`
(ROOT, one file, rank 0) and `write_tree_solution` (all vars, directory,
sharded). `write_tree_solution` is untouched, so the heavily-used
`_soldir` path carries no regression risk.

## Canonical format

A single CSV. Comment lines start with `#`; data lines are
`node_name, variable_name, value` with **node-local** variable names.
Readers strip surrounding whitespace from each field and skip blank /
`#` lines.

```
# mpi-sppy xhat: nonant tree, node-local variable names
# node_name, variable_name, value
ROOT, DevotedAcres[CORN], 80.0
ROOT, DevotedAcres[SUGAR_BEETS], 250.0
ROOT, DevotedAcres[WHEAT], 170.0
ROOT_0, ProductionAdjust[CORN], 0.0
ROOT_1, ProductionAdjust[CORN], 48.0
```

- **By name, not positional.** Robust to variable-ordering changes
  between the writing run and any future reading run, human-readable,
  hand-editable. Matches `first_stage_nonant_writer` and the wxbar CSV
  reader. (The positional `.npy` ROOT vector stays for the existing
  2-stage MMW path; it is not touched here.)
- **Single file** for the whole tree; the `node_name` column
  distinguishes nodes. Answers the "file vs directory" question for the
  multi-stage case.
- **Node-local names** so the same file reads back into any single
  scenario model regardless of EF-vs-cylinders origin.
- **Nodes written are the non-leaf nodes** in each scenario's
  `_mpisppy_node_list` — exactly the set `_fix_nonants` requires. Leaf
  recourse is scenario-specific and not part of an xhat.

## Architecture

Three small pieces in `mpisppy/utils/sputils.py`, plus method hooks.

### 1. Shared serializer (format, no extraction)

```python
def write_nonant_tree_csv(file_name, node_to_rows):
    # node_to_rows: {node_name: [(local_var_name, value), ...]}
    # writes the canonical CSV (header + node-local rows), deterministic order
```

`ef_nonants_csv` is reimplemented to build `node_to_rows` from
`ef.ref_vars` (stripping the EF block prefix to node-local) and call
this. A single `_node_local_name(name)` helper centralizes the
prefix-strip rule already used by `first_stage_nonant_writer`'s bundling
branch.

### 2. Whole-tree assembly for cylinders (the genuinely new code)

```python
def gather_nonant_tree(opt):
    # On each rank: local {node -> [(local_name, value)]} from opt.local_scenarios,
    #   one entry per node (first local scenario through the node wins; NAC
    #   guarantees the rest agree).
    # opt.mpicomm.gather(local, root=0); on rank 0 merge (first wins) and return;
    #   non-root ranks return None.
```

~25 lines around `mpicomm.gather`. In the no-MPI mock (`mpisppy/MPI.py`)
the gather is trivially the single local dict. The merge takes
first-rank-wins, but guards it: if two ranks report different values for
the same node beyond a small tolerance, **raise**. A genuine
disagreement means the incumbent is not nonanticipative, so the file
would be ill-defined (which rank's value would we write?) — failing
loudly beats silently writing a wrong xhat. Integer/binary nonants must
match exactly; the tolerance only absorbs floating-point residue.

### 3. The new method

`SPBase.write_tree_nonants(file_name)` (mirrors
`write_first_stage_solution`): ensure `tree_solution_available`
(`load_best_solution()` if needed), `gather_nonant_tree(self)`, and on
rank 0 `os.makedirs` the parent dir if any and
`write_nonant_tree_csv`. `WheelSpinner.write_tree_nonants` delegates to
`self.spcomm.opt.write_tree_nonants`, mirroring the existing two
delegators. EF parity comes free: `ef_nonants_csv` already produces the
same format via the shared serializer.

## Explicitly out of scope (deferred to the reader phase)

- **The reader** (`{node: array}` from the CSV) and any change to
  `--xhat-from-file` / `xhatbase._try_file_xhat`.
- **`ciutils.write_xhat`/`read_xhat` generalization.** Deliberately not
  touched here: `write_xhat` receives a bare `{node: array}` cache with
  **no variable names**, so it cannot emit the by-name CSV without a
  model. Whether the multi-stage `ciutils` path becomes model-aware
  (by-name) or stays model-free (a positional `.npz`/`node,index,value`)
  is a reader-phase decision about round-trip semantics, made where the
  consumers (MMW, zhat4xhat) and their model access are in view. The
  by-name CSV format chosen here is the input to that decision, not a
  prejudgment of it.
- The multi-stage `feasible_xhat_creator` contract and docs.

This boundary keeps the writing PR small and avoids baking half of a
reader decision into the writer.

## Backward compatibility

- `ef_nonants_csv` output changes (names node-local; `#` header). Update
  its one format test; note the EF `--solution-base-name.csv` change in
  the PR description and user docs.
- `write_tree_solution`, `first_stage_nonant_writer`, the `.npy` ROOT
  serializers, and the `ciutils` 2-stage path are **unchanged**.

## PR scope checklist

1. `write_nonant_tree_csv` serializer + `_node_local_name` helper.
2. `ef_nonants_csv` refactored onto the serializer (node-local names).
3. `gather_nonant_tree(opt)` assembler.
4. `SPBase.write_tree_nonants` + `WheelSpinner.write_tree_nonants`.
5. `cfg.write_xhat_file_args()` (`--write-xhat-file`), wired into the
   generic parsing and into both `generic/decomp.py::_write_solutions`
   (cylinders) and `generic/ef.py` (EF).
6. Tests (below), wired into `run_coverage.bash` **and**
   `test_pr_and_main.yml` in the same commit.
7. Doc: short note in `doc/src/` on the format + `--write-xhat-file`;
   mention the EF `.csv` format change.

## Tests

- **Serial / EF:** `ef_nonants_csv` (now node-local) writes the expected
  rows for a small multi-stage example (e.g. `hydro`); reload the values
  into a fresh scenario by name and assert they match. Update
  `test_ef_nonants_csv_creates_file`.
- **Cylinders gather (needs MPI):** a 2-rank multi-stage run; assert
  rank 0's `write_tree_nonants` file contains every node exactly once and
  values match the incumbent. Add to the mpiexec test list.
- **No-MPI path:** `gather_nonant_tree` returns the single local dict
  under the `MPI.py` mock.
- Coverage-harness wiring as in (5) above.

## Resolved

- **Node ordering.** Sort by the node-name *path tuple*: split the name
  on `_` and read trailing segments as ints (`ROOT`, `ROOT_0`,
  `ROOT_0_1`, ...). Stage is implied by the name (depth = segment count),
  so no `node.stage` lookup is needed; integer-izing the segments makes
  `ROOT_10` sort after `ROOT_2`.
- **Shared-node disagreement is an error, not a warning** (see the
  assembler section). The tolerance only forgives floating-point residue;
  integer/binary nonants must match exactly.

## How the file gets produced: a dedicated `--write-xhat-file` flag

**Decided:** a dedicated `--write-xhat-file <path>` flag, *not* a
piggyback on `--solution-base-name` and not method-only. Rationale: it is
turnkey, behaves identically for EF and cylinders (both route through the
shared serializer, so the file is byte-identical regardless of how the
xhat was produced), and it stays clear of the EF-vs-cylinders
`--solution-base-name.csv` asymmetry. `write_tree_nonants` remains a
public method too, for driver scripts and `custom_writer` hooks.

Wiring:

- A new `cfg.write_xhat_file_args()` registering `--write-xhat-file`
  (`domain=str`, default `None` = off), added to the generic parsing
  alongside the other solution-output options.
- Cylinders: when set, `generic/decomp.py::_write_solutions` calls
  `wheel.write_tree_nonants(cfg.write_xhat_file)`.
- EF: when set, `generic/ef.py` writes the same format to the same path
  via the shared serializer (the node-local `ef_nonants_csv` adapter).

This is write-only: the flag produces a file. The matching
`--xhat-from-file` *read* of a multi-stage file is the next phase.
