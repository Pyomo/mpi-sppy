# File-based scenario module: LP files and rho-from-CSV — Design

Status: draft for review. Small, **additive, back-compatible** changes to
`mpisppy/problem_io/mps_module.py` (plus a one-line generalization in
`mpisppy/utils/rho_utils.py`). Motivated by the PyPSA integration
(`doc/designs/pypsa_stochastic_design.md`) but useful for any file-based scenario
workflow. No changes to the driver, the reader, or PH.

## 1. Scope

The file-based scenario path is the module selected by `--mps-files-directory`,
which `generic/parsing.py` (L24–25) maps to `mpisppy.problem_io.mps_module`. Two
additions:

1. **Accept `.lp` files** in addition to `.mps`.
2. **Apply per-nonant rho** from `{s}_rho.csv` via a module `_rho_setter`.

MPS-only usage is unaffected.

## 2. Background — what already works (no change needed)

- **Reader handles LP already.** `mps_reader.read_mps_and_create_pyomo_model`
  (L158–168) calls coin-or `mip.Model().read()`, which auto-detects `.lp` by
  extension. Verified end to end: a `.lp` path round-trips through the *unchanged*
  reader and `find_component` resolves the variables.
- **`_rho_setter` is auto-discovered.** `generic/decomp.py:_get_rho_setter`
  (L87–95) does `module._rho_setter if hasattr(module, '_rho_setter') else None`
  and threads it to hub/spokes (`generic/hub.py`, `generic/spokes.py`). So a
  rho_setter added to `mps_module` is picked up with no driver change.
- **rho_setter contract:** `rho_setter(scenario) -> list[(id(vardata), rho)]`
  (see `rho_utils.rho_list_from_csv`, L26–44).

## 3. Change 1 — accept `.lp` files

`.mps` is currently hardcoded in two places:

- `scenario_creator` (L47–48): `mpsPath = sharedPath + ".mps"`.
- `scenario_names_creator` (L94–112): globs `*.mps` and strips the extension with
  `[:-4]` — **wrong for `.lp`** (3 chars, not 4).

Proposed: a small resolver supporting both extensions (search order = preference),
and `os.path.splitext` for base names.

```python
_SCENARIO_EXTS = (".lp", ".mps")   # both supported; .lp preferred when both exist

def _scenario_model_path(directory, sname):
    for ext in _SCENARIO_EXTS:
        p = os.path.join(directory, sname + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No scenario model file for '{sname}' in {directory} "
        f"(looked for {', '.join(sname + e for e in _SCENARIO_EXTS)})")
```

- `scenario_creator`:
  `model = mps_reader.read_mps_and_create_pyomo_model(
       _scenario_model_path(cfg.mps_files_directory, sname))`.
- `scenario_names_creator`: glob both extensions, derive base names with
  `os.path.splitext`, dedup:

  ```python
  files = []
  for ext in _SCENARIO_EXTS:
      files += glob.glob(os.path.join(mps_files_directory, "*" + ext))
  names = sorted({os.path.splitext(os.path.basename(f))[0] for f in files})
  ```

  then the existing numbering logic runs on `names` (replacing `[:-4]`).

- **Precedence:** when both `{s}.lp` and `{s}.mps` exist, `.lp` wins (search
  order). MPS-only dirs are unaffected (only `.mps` present). A
  `--scenario-file-ext` override could be added later if needed.
- **Option name:** keep `mps_files_directory` (it may now hold `.lp` files) for
  back-compat. (The author's own "note to dlw" at `mps_module.py` L23 already
  anticipated LP.)

## 4. Change 2 — per-nonant rho from `{s}_rho.csv`

Add a module `_rho_setter` so the driver applies per-nonant rho automatically.
To avoid depending on a scenario-name attribute, `scenario_creator` stashes the
rho-file path on the model (it already computes `sharedPath`):

```python
# in scenario_creator, after building `model`:
rho_path = sharedPath + "_rho.csv"
model._mps_rho_csv = rho_path if os.path.exists(rho_path) else None
```

```python
import mpisppy.utils.rho_utils as rho_utils

def _rho_setter(scenario):
    path = getattr(scenario, "_mps_rho_csv", None)
    if not path:
        return []            # no rho file -> fall back to --default-rho
    return rho_utils.rho_list_from_csv(scenario, path)
```

- Returns `[]` when no rho file is present, so existing `--default-rho` behavior
  is preserved.
- **Guard interaction:** because defining `_rho_setter` makes
  `_get_rho_setter` non-None, the "must specify `--default-rho`" check (L90–94)
  is bypassed. Per-nonant rho should therefore **cover every nonant** (PyPSA's
  `_rho.csv` does), and callers should still pass `--default-rho` as a backstop
  for any nonant not listed.

### 4.1 Header reconciliation (`varname` vs `fullname`)

The file-path `_rho.csv` — written by `scenario_lp_mps_files.py` (L84–87), by
PyPSA, and in `_delme_test_write_mp_mps_dir/` — uses header **`varname,rho`**.
But `rho_utils.rho_list_from_csv` (L34/L37) reads a **`fullname`** column.
Reconcile by generalizing the reader to accept either header and to apply the
same `(`→`_`, `)`→`_` normalization that `scenario_creator` uses (L67), so names
match the Pyomo components built from the file:

```python
rhodf = pd.read_csv(filename)
namecol = "varname" if "varname" in rhodf.columns else "fullname"
for _, row in rhodf.iterrows():
    name = str(row[namecol]).replace('(', '_').replace(')', '_')
    vo = s.find_component(name)
    ...
```

(With PyPSA's implicit `x{label}` names there are no parens, so normalization is a
no-op there; it keeps general Pyomo-name cases working.)

## 5. Backward compatibility

- MPS-only directories: `scenario_creator` / `scenario_names_creator` still find
  `.mps`; behavior identical.
- No rho csv: `_rho_setter` returns `[]` → `--default-rho` path unchanged.
- Driver, reader, and PH are untouched.

## 6. Testing (extend `mpisppy/tests/test_mps.py`)

1. Round-trip an `.lp` scenario through `scenario_creator` (build + solve).
2. `scenario_names_creator` over a dir of `.lp` files returns correct base names.
3. A dir with `{s}_rho.csv` → `_rho_setter` returns the expected
   `(id(vardata), rho)` list; PH run uses those rhos.
4. Regression: the existing `.mps`-only test still passes; a dir with no rho csv
   falls back to `--default-rho`.
5. Edge: a dir containing both `{s}.lp` and `{s}.mps` resolves to `.lp`.

## 7. Touch list

- `mpisppy/problem_io/mps_module.py`:
  - add `_SCENARIO_EXTS` + `_scenario_model_path`;
  - `scenario_creator` (L47–49) use the resolver; stash `_mps_rho_csv`;
  - `scenario_names_creator` (L94–112) glob both exts, `os.path.splitext`;
  - add `_rho_setter`; `import mpisppy.utils.rho_utils`.
- `mpisppy/utils/rho_utils.py`: `rho_list_from_csv` (L34–43) accept `varname` and
  normalize parens.
- `mpisppy/tests/test_mps.py`: cases in §6.
- Driver / reader / PH: **no changes**.
