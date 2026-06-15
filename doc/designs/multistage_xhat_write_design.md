# Design: Reading and writing a multi-stage xhat (nonant tree) file

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

This note covers **both write and read** of the canonical multi-stage
xhat file, as one self-contained PR. (It began as write-only; the read
side was folded in so the file format is exercised end-to-end by a real
write→read→fix round trip, rather than shipping a producer with no
consumer.) The `ciutils` generalization and the multi-stage
`feasible_xhat_creator` contract remain out of scope (see below).

The read side reuses the existing `--xhat-from-file` surface
(`xhatbase._try_file_xhat`), which previously accepted only a two-stage
`.npy` ROOT vector; it now also accepts the canonical `.csv` for any
number of stages, matched to the model by variable name.

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
- **A cylinders run already gathers nonant values to rank 0 — but
  scenario-keyed, not node-keyed.** `SPBase.gather_var_values_to_rank0`
  walks every local scenario's node list, builds
  `{(scenario_name, var_name): value}` (bundle prefix stripped, zero-prob
  aware), `mpicomm.gather`s to rank 0 and merges — so the whole tree's
  nonant values *do* land on one rank today (used by
  `report_var_values_at_rank0`). What it does **not** do is key by
  *node*: it carries per-scenario redundancy for shared nodes and drops
  the node association. `WheelSpinner.local_nonant_cache` is the
  complementary piece — node-keyed and deduped, but *local* (per-rank,
  values only). So the writer needs **no novel gather**: it needs a
  node-keyed rank-0 assembly reusing these two existing patterns.
  (`write_tree_solution` itself still sidesteps assembly by having every
  rank write its own scenarios into a directory — which is why a *single*
  file needs the node-keyed rank-0 form.)

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
  scenario model regardless of EF-vs-cylinders origin. Localizing is
  prefix-aware (`sputils._node_local_nonant_name`): a leading bundle
  segment, or a `<scenario_name>.` prefix that some spokes introduce
  (the multistage xhatshuffle stage-2 EF rebinds nonants into a
  scenario-named sub-block), is stripped on both write and read so the
  names stay scenario-independent and round-trip. (aircond surfaced
  this; without it the file carried per-node scenario prefixes.)
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

### 2. Whole-tree assembly for cylinders (reuse, not new code)

This is **not** a novel gather. `SPBase.gather_var_values_to_rank0`
already does the intra-cylinder `mpicomm.gather` of nonant values to
rank 0, and `WheelSpinner.local_nonant_cache` already builds the
node-keyed, deduped *local* dict. The assembler is the node-keyed analog
of the former, reusing the latter's per-node dedup:

```python
def gather_nonant_tree_to_rank0(opt):
    # local: {node_name: [(local_var_name, value), ...]}, one entry per node
    #   (first local scenario through the node wins -- mirrors
    #   local_nonant_cache's dedup; adds the var name + bundle-prefix strip
    #   from gather_var_values_to_rank0).
    # opt.mpicomm.gather(local, root=0); merge on rank 0 (first wins);
    #   non-root ranks return None.
```

Whether this lands as a new sibling or as a `by_node=True` option on
`gather_var_values_to_rank0` is an implementation call; either way it
reuses the tested gather/strip logic rather than re-deriving it. In the
no-MPI mock (`mpisppy/MPI.py`) the gather is trivially the single local
dict. The merge takes first-rank-wins but guards it: if two ranks report
different values for the same node beyond a small tolerance, **raise** —
a genuine disagreement means the incumbent is not nonanticipative, so the
file would be ill-defined (which rank's value would we write?) and
failing loudly beats silently writing a wrong xhat. Integer/binary
nonants must match exactly; the tolerance only absorbs floating-point
residue. Zero-prob nonants follow `gather_var_values_to_rank0`'s handling
(request their values explicitly when writing an xhat).

### 3. The new method

`SPBase.write_tree_nonants(file_name)` (mirrors
`write_first_stage_solution`): ensure `tree_solution_available`
(`load_best_solution()` if needed), `gather_nonant_tree(self)`, and on
rank 0 `os.makedirs` the parent dir if any and
`write_nonant_tree_csv`. `WheelSpinner.write_tree_nonants` delegates to
`self.spcomm.opt.write_tree_nonants`, mirroring the existing two
delegators. EF parity comes free: `ef_nonants_csv` already produces the
same format via the shared serializer.

### 4. The reader (`--xhat-from-file` for any number of stages)

`sputils.read_nonant_tree_csv(file_name, node_varname_order)` is the
inverse of the serializer: it parses the CSV to `{node: {name: value}}`
and returns `{node: np.ndarray}` ordered to a caller-supplied per-node
variable order. Because the file is keyed by node-local **name**, the
consumer supplies the order it wants — typically each
`node.nonant_vardata_list`'s node-local names — so the array lands in the
cache's positional order without depending on file row order. Parsing
splits each data line on the **first** comma (node) and the **last**
comma (value) so multi-index Var names (`x[a,b]`) survive.

`xhatbase._try_file_xhat` dispatches on extension: `.csv` →
`read_nonant_tree_csv` against the spoke's own local scenarios (any
number of stages; a file may carry more nodes than a rank needs, and the
extras are ignored); anything else → the existing two-stage `.npy`
`ciutils.read_xhat` path (a multi-stage `.npy` still hard-fails, now
pointing at the `.csv` format). The assembled cache flows into the
unchanged `self.opt.evaluate(...)` / `update_if_improving(...)` path.

## Explicitly out of scope

- **`ciutils.write_xhat`/`read_xhat` generalization.** Deliberately left
  alone: `write_xhat` receives a bare `{node: array}` cache with **no
  variable names**, and `read_xhat` is model-free (`np.load`), used by
  MMW/zhat4xhat with no reference model. The by-name CSV needs a model to
  resolve order, so it lives in `sputils` (writer has the model; reader
  takes the order map) and `--xhat-from-file` dispatches to it. `ciutils`
  stays the model-free two-stage `.npy` path, untouched.
- The multi-stage `feasible_xhat_creator` contract and docs.

## Backward compatibility

- `ef_nonants_csv` output changes (names node-local; `#` header). Update
  its one format test; note the EF `--solution-base-name.csv` change in
  the PR description and user docs.
- `write_tree_solution`, `first_stage_nonant_writer`, the `.npy` ROOT
  serializers, and the `ciutils` 2-stage path are **unchanged**.

## PR scope checklist

1. `write_nonant_tree_csv` serializer + `_node_local_name` helper.
2. `ef_nonants_csv` refactored onto the serializer (node-local names).
3. `gather_nonant_tree_to_rank0(opt)` — node-keyed analog of the existing
   `gather_var_values_to_rank0`, reusing `local_nonant_cache`'s per-node
   dedup (not a new gather mechanism).
4. `SPBase.write_tree_nonants` + `WheelSpinner.write_tree_nonants`.
5. `cfg.write_xhat_file_args()` (`--write-xhat-file`), wired into the
   generic parsing and into both `generic/decomp.py::_write_solutions`
   (cylinders) and `generic/ef.py` (EF).
6. **Reader:** `sputils.read_nonant_tree_csv`; `.csv` dispatch in
   `xhatbase._try_file_xhat` (+ `_read_xhat_csv`); `xhat_from_file`
   description updated for the multi-stage `.csv`.
7. Tests (below); all added to existing harness-wired files
   (`test_sputils.py`, `test_xhat_from_file.py`, `test_ef_ph.py`), so no
   new coverage-harness wiring is needed.
8. Doc: short note in `doc/src/` on the format + `--write-xhat-file` +
   the multi-stage `--xhat-from-file`; mention the EF `.csv` format
   change.

## Tests

- **Serializer / EF round-trip (no solver):** `write_nonant_tree_csv`
  format + node order; `ef_nonants_csv` (now node-local) round-trips
  position-for-position against `nonant_cache_from_ef`. Updated
  `test_ef_nonants_csv_creates_file` and the `test_ef_ph` hydro EF check.
- **Reader (no solver):** `read_nonant_tree_csv` round-trips the writer,
  follows the requested order, ignores unrequested nodes, survives
  multi-index names, and raises on a missing node/variable.
  `_try_file_xhat` `.csv` path via the existing `test_xhat_from_file.py`
  stubs (happy path, model-order wins, extra nodes ignored, missing var
  raises).
- **Gather (no MPI):** `gather_nonant_tree_to_rank0` returns the
  node-keyed, deduped local dict under the serial path;
  `_assert_nonant_node_agreement` passes within tolerance and raises on
  disagreement.
- **End-to-end MPI round trip (automated):**
  `test_xhat_file_multistage.py` (mpiexec -np 4) runs a multistage
  aircond cylinder system, writes the incumbent tree, and reads it back
  by name. At np 4 the writing cylinder spans two ranks, so it exercises
  the cross-rank gather/merge (a single rank holds only some subtrees).
  Wired into `run_coverage.bash` and `test_pr_and_main.yml`.
- **End-to-end smoke (manual, this PR):** farmer (2-stage) and hydro
  (3-stage) write→read round trips through `generic_cylinders`, EF and
  cylinders.

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
