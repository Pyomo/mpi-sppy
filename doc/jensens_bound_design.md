# Jensen's-bound design (two-stage only)

Status: design draft. No code yet. Asking for sign-off before implementation.

Related: upstream issue #594 (Jensen's bound). This design departs from the
issue in several ways, noted inline.

---

## 0. Goals

1. Let any lower-bounder spoke (lagrangian, lagranger, subgradient,
   reduced_costs) optionally compute a Jensen's lower bound from a single
   expected-value (EV) scenario before it starts its normal iterations, and
   send that as its first outer bound.
2. Let any xhat / upper-bounder spoke (xhatshuffle, xhatxbar, xhatlooper,
   xhatspecific) optionally take the EV-problem's first-stage solution as an
   xhat, evaluate it across all scenarios, and send that as its first inner
   bound.
3. Do NOT introduce an "iteration -1" in PH, APH, L-shaped, or any hub. All
   Jensen's work happens inside the spoke that opts in.
4. Two-stage only for this round. Multi-stage is explicitly out of scope.
5. Teach users a clean module-authoring pattern via well-crafted farmer and
   sizes examples.

Non-goals:

- Sharing/caching the EV solve across spokes (each spoke solves its own).
- Any change to hub iteration logic.
- Multi-stage EV-per-node computation.

---

## 1. Jensen's inequality: front-and-center warning

Jensen's bound is valid **only when the recourse value function Q(x, ξ) is
convex in the random parameters ξ.**

Necessary conditions (not sufficient):

1. The second-stage problem is an LP — no integer recourse.
2. Random parameters appear only in objective coefficients and RHS, not in
   constraint-matrix coefficients in ways that break convexity.
3. Two-stage structure.

mpi-sppy checks (1) automatically and refuses to compute the bound when
non-nonant integer/binary Vars exist in the EV model. (2) and (3) are the
user's responsibility.

For `sense=minimize`:
  `E[Q(x, ξ)] >= Q(x, E[ξ])` → EV optimum is a valid **lower bound** on the
  true optimal expected cost → a valid outer bound for a minimization hub.

For `sense=maximize`: inequality flips; EV optimum is a valid **upper bound**
on the true optimum → still a valid outer bound in the max-sense convention.

This warning must appear at the top of `doc/src/jensens.rst` (new) and at the
top of any docstring for `expected_value_creator` examples we ship.

---

## 2. User-facing module contract

A scenario module that wants to participate must define:

```python
def expected_value_creator(scenario_name, **kwargs):
    """Return a Pyomo model with the same shape as scenario_creator(...),
    but built from expectation-valued random data.

    Two-stage: _mpisppy_probability must be 1.0 (a single deterministic
    scenario, not a member of a probabilistic ensemble).
    """
```

Discovered via the same `hasattr(module, ...)` pattern already used for
`scenario_denouement` (`generic_cylinders.py:45,88`, `amalgamator.py:173`).

Name rationale: `expected_value_creator`, **not** `jensens_creator` as the
upstream issue suggested. The function computes the EV scenario; Jensen's is
how the resulting bound is *interpreted*. The name should describe what is
built, not what it will be used for.

### 2.1 Best-practice pattern: underscore helpers

Users will copy our examples. We want the right pattern in front of them.

```python
def _scenario_data(scenario_name, **kwargs):
    """Pure-Python random data as a plain dict. No Pyomo."""
    ...

def _build_model(scenario_name, data, *, probability, **kwargs):
    """Build Pyomo model from the data dict. Shared build path."""
    ...

def scenario_creator(scenario_name, **kwargs):
    data = _scenario_data(scenario_name, **kwargs)
    prob = 1.0 / kwargs["num_scens"] if kwargs.get("num_scens") else "uniform"
    return _build_model(scenario_name, data, probability=prob, **kwargs)

def expected_value_creator(scenario_name, **kwargs):
    # NOTE: could be multi-threaded for large num_scens.
    snames = scenario_names_creator(kwargs["num_scens"])
    datas  = [_scenario_data(s, **kwargs) for s in snames]
    avg    = _average_scenario_data(datas)
    return _build_model(scenario_name, avg, probability=1.0, **kwargs)
```

Why this matters:

- One source of truth for "what does the Pyomo model look like." If
  `scenario_creator` and `expected_value_creator` each build the model
  inline, they drift.
- Separating data from model makes averaging trivial: average the dict, not
  the Pyomo components.
- Two-stage: every rank independently calls `expected_value_creator` and
  gets an identical model. No collective communication needed.

---

## 3. CLI / Config options

In `mpisppy/utils/config.py`, one `add_to_config` call appended to each
existing spoke-args function. All bool, default False.

| Existing function (line)   | New flag                          |
| :------------------------- | :-------------------------------- |
| `lagrangian_args` (676)    | `lagrangian_try_jensens_first`    |
| `lagranger_args`           | `lagranger_try_jensens_first`     |
| `subgradient_args` (517)   | `subgradient_try_jensens_first`   |
| `reduced_costs_args` (687) | `reduced_costs_try_jensens_first` |
| `xhatshuffle_args` (827)   | `xhatshuffle_try_jensens_first`   |
| `xhatxbar_args` (888)      | `xhatxbar_try_jensens_first`      |
| `xhatlooper_args` (815)    | `xhatlooper_try_jensens_first`    |
| `xhatspecific_args` (879)  | `xhatspecific_try_jensens_first`  |

CLI form (Pyomo's ConfigDict convention): `--lagrangian-try-jensens-first`.

### 3.1 Interaction with each spoke's existing iter-0

- `lagrangian`: its normal iter-0 trivial bound (W=0) is already
  wait-and-see and is itself a valid Jensen-like outer bound. The new
  flag's actual value is getting a cheap EV bound out **before** iter-0
  finishes — useful when the problem has many scenarios and iter-0 is
  slow. Document this; do not disable or replace iter-0.
- `subgradient`: iter-0 solves all scenarios. EV bound arrives earlier.
- `reduced_costs`: inherits from lagrangian; same behavior.
- `lagranger`: does its own PH from scratch; EV bound is essentially free
  relative to its iter-0.

---

## 4. Wiring through `cfg_vanilla`

The flag-bearing spoke factories need access to the user module's
`expected_value_creator`. Two choices:

- **(a, preferred)** Add `expected_value_creator=None` kwarg to each
  affected factory. Driver code (`generic_cylinders.py`, `farmer_cylinders.py`,
  etc.) does `getattr(module, "expected_value_creator", None)` once and
  threads it in. Consistent with how `scenario_creator` is already passed.
- (b) Let each factory take a `module` object. Bigger surface change; more
  coupling.

With choice (a), every affected factory gains a block like:

```python
if cfg.get("<name>_try_jensens_first", False):
    if expected_value_creator is None:
        raise RuntimeError(
            "--<name>-try-jensens-first was requested but the scenario "
            "module has no expected_value_creator function."
        )
    spoke_dict["opt_kwargs"]["options"]["jensens"] = {
        "expected_value_creator": expected_value_creator,
        "scenario_creator_kwargs": scenario_creator_kwargs,
    }
```

Scenario-list size is already available on the spoke at runtime via
`self.opt.all_scenario_names` — no need to thread it separately.

---

## 5. Spoke mechanics — a shared mixin

New file `mpisppy/cylinders/_jensens_mixin.py`:

```python
class _JensensMixin:
    def _jensens_enabled(self):
        return "jensens" in self.opt.options

    def _jensens_build_and_check(self):
        j = self.opt.options["jensens"]
        ev_creator = j["expected_value_creator"]
        sname = self.opt.all_scenario_names[0]  # name only; data is averaged
        ev_model = ev_creator(sname, **(j["scenario_creator_kwargs"] or {}))
        _assert_two_stage(ev_model)              # len(_mpisppy_node_list) == 1
        _assert_jensen_integer_safe(ev_model)    # see section 6
        return ev_model

    def _jensens_solve(self, ev_model):
        """Solve ev_model using this spoke's solver. Return (obj, nonants)."""
        ...
```

### 5.1 Lower-bounder integration

In `_LagrangianMixin` (`lagrangian_bounder.py:12-55`) and the three class
`main()` bodies, insert right after `self.lagrangian_prep()` / equivalent
and **before** iter-0:

```python
if self._jensens_enabled():
    ev_model = self._jensens_build_and_check()
    ev_obj, _ = self._jensens_solve(ev_model)
    self.send_bound(ev_obj)     # first outer bound — sent before iter-0
```

Hook points:

- `LagrangianOuterBound.main`  → after line 78 (`self.lagrangian_prep()`),
  before line 85 (iter-0 lagrangian solve).
- `LagrangerOuterBound.main`   → after line 86, before line 91.
- `SubgradientOuterBound.main` → after `PH_Prep` at line 28, before
  `self.opt.Iter0()` at line 29.
- `ReducedCostsSpoke.main`     → inherits from Lagrangian; free.

### 5.2 Xhat integration

Each xhat spoke's `main()` calls `self.xhat_prep()` first
(`xhatbase.py:21`), which sets up `Xhat_Eval`. We insert the EV-heuristic
try **after** `xhat_prep` and **before** the spoke's main scenario-cycling
loop:

```python
def main(self):
    self.xhat_prep()
    if self._jensens_enabled():
        ev_model = self._jensens_build_and_check()
        _, nonant_values = self._jensens_solve(ev_model)
        nonant_cache = _pack_nonant_cache(self.opt, nonant_values)
        Eobj = self.opt.evaluate(nonant_cache)   # xhat_eval.py:257
        self.update_if_improving(Eobj)            # spoke.py:173
    # ... existing main-loop body
```

`self.opt.evaluate(nonant_cache)` already fixes nonants and calls
`solve_loop` across local scenarios. Reuse, don't reinvent.

Hook points:

- `XhatShuffleInnerBound.main`  → after line 68 (`self.xhat_prep()`).
- `XhatXbarInnerBound.main`     → after line 43.
- `XhatLooperInnerBound.main`   → after line 32.
- `XhatSpecificInnerBound.main` → after line 43.

---

## 6. Convexity / integer-safety guard

New helper in `mpisppy/utils/sputils.py`:

```python
def assert_jensen_integer_safe(scenario):
    """Raise if any integer/binary Var is outside the nonant set.

    Necessary (not sufficient) check that recourse is convex in the
    random parameters — Jensen's bound is only valid under that
    assumption. Convexity in the random parameters cannot be checked
    statically; this integer check catches the common failure mode.
    """
    nonant_ids = {id(v) for v in scenario._mpisppy_data.nonant_indices.values()}
    for v in scenario.component_data_objects(pyo.Var, descend_into=True):
        if id(v) in nonant_ids:
            continue
        if v.is_integer() or v.is_binary():
            raise RuntimeError(
                f"Jensen's bound requires convex recourse, but non-nonant "
                f"integer/binary Var found: {v.name}. Disable the "
                f"--*-try-jensens-first flag, or reformulate."
            )
```

Called once, on the EV model (not on scenario models — those are never
built in the Jensen's path).

---

## 7. Farmer rewrite (`examples/farmer/farmer.py`) — the showcase

Current state: `scenario_creator` interleaves seeding, yield generation,
and Pyomo model build across lines 56–211. Split into three named helpers;
keep the public API identical.

```python
def _scenario_data(scenario_name, crops_multiplier=1, num_scens=None,
                   seedoffset=0):
    """Pure-Python random data for one scenario. No Pyomo.
    Deterministic given (scenario_name, seedoffset)."""
    scennum  = sputils.extract_num(scenario_name)
    basenum  = scennum % 3
    groupnum = scennum // 3
    basename = ['BelowAverageScenario', 'AverageScenario',
                'AboveAverageScenario'][basenum]

    farmerstream.seed(scennum + seedoffset)

    base_yield = {
        'BelowAverageScenario': {'WHEAT': 2.0, 'CORN': 2.4, 'SUGAR_BEETS': 16.0},
        'AverageScenario':      {'WHEAT': 2.5, 'CORN': 3.0, 'SUGAR_BEETS': 20.0},
        'AboveAverageScenario': {'WHEAT': 3.0, 'CORN': 3.6, 'SUGAR_BEETS': 24.0},
    }[basename]

    yields = {}
    for i in range(crops_multiplier):
        for crop in ['WHEAT', 'CORN', 'SUGAR_BEETS']:
            jitter = farmerstream.rand() if groupnum != 0 else 0.0
            yields[crop + str(i)] = base_yield[crop] + jitter
    return {"Yield": yields}


def _build_model(scenario_name, data, *, use_integer=False, sense=pyo.minimize,
                 crops_multiplier=1, probability):
    """Build the Pyomo model from a data dict. Shared by scenario_creator
    and expected_value_creator — that sharing is the whole point of the
    split."""
    # ... body is current lines 74-199, reading data["Yield"][crop]
    #     instead of computing the yield inline.
    model._mpisppy_probability = probability
    sputils.attach_root_node(model, model.FirstStageCost,
                             [model.DevotedAcreage])
    return model


def scenario_creator(scenario_name, use_integer=False, sense=pyo.minimize,
                     crops_multiplier=1, num_scens=None, seedoffset=0):
    data = _scenario_data(scenario_name, crops_multiplier, num_scens,
                          seedoffset)
    prob = (1.0 / num_scens) if num_scens is not None else "uniform"
    return _build_model(scenario_name, data,
                        use_integer=use_integer, sense=sense,
                        crops_multiplier=crops_multiplier, probability=prob)


def expected_value_creator(scenario_name, use_integer=False,
                           sense=pyo.minimize, crops_multiplier=1,
                           num_scens=None, seedoffset=0):
    """Deterministic EV scenario for the two-stage scalable farmer.

    Every rank independently constructs the same model. Not parallelized.
    For large num_scens this could be multi-threaded — for example,

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as ex:
            datas = list(ex.map(lambda s: _scenario_data(s, ...), snames))

    but that is left for a later pass.
    """
    if num_scens is None:
        raise ValueError("expected_value_creator requires num_scens")
    snames = scenario_names_creator(num_scens)
    datas  = [_scenario_data(s, crops_multiplier, num_scens, seedoffset)
              for s in snames]
    avg = {"Yield": {k: sum(d["Yield"][k] for d in datas) / len(datas)
                     for k in datas[0]["Yield"]}}
    return _build_model(scenario_name, avg,
                        use_integer=use_integer, sense=sense,
                        crops_multiplier=crops_multiplier, probability=1.0)
```

Notes:

- `num_scens` is required for `expected_value_creator`. The expectation is
  taken over the explicit scenario set the run uses, per user guidance:
  "compute the scenario data for every scenario that would be called and
  then average the scenario data."
- `sample_tree_scen_creator` unchanged — still delegates to
  `scenario_creator`.
- `_build_model` takes `probability` as an explicit keyword so the build
  step never has to guess what kind of scenario it is producing.
- Multi-threading comment is in the docstring, per user request.

---

## 8. Sizes rewrite (`examples/sizes/sizes.py`)

Harder than farmer because data lives in `.dat` files read by
`ref.model.create_instance(fname)`. The same split still works but needs a
`.dat`-to-dict adapter so the two paths share a build step.

```python
def _scenario_data(scenario_name, scenario_count):
    """Return {param_name -> {index -> value}} for one scenario.

    Loaded from the .dat via the existing AbstractModel. First-stage params
    are identical across scenarios and are included too; averaging them is
    a no-op, which is fine.
    """
    fname = os.path.join(os.path.dirname(__file__),
                         f"SIZES{scenario_count}", f"{scenario_name}.dat")
    instance = ref.model.create_instance(fname)
    data = {}
    for p in instance.component_objects(pyo.Param, active=True):
        data[p.name] = {idx: pyo.value(p[idx]) for idx in p}
    data["_NumSizes"] = pyo.value(instance.NumSizes)
    return data


def _build_model(scenario_name, data, probability):
    """Build a ConcreteModel directly from a data dict — no AbstractModel
    roundtrip. Shared by scenario_creator and expected_value_creator."""
    m = pyo.ConcreteModel(scenario_name)
    m.NumSizes = pyo.Param(initialize=data["_NumSizes"],
                           within=pyo.NonNegativeIntegers)
    m.ProductSizes = pyo.Set(initialize=list(range(1, data["_NumSizes"] + 1)))
    for pname in ("DemandsFirstStage", "DemandsSecondStage",
                  "UnitProductionCosts", "StockingCost", "UnitReductionCost"):
        setattr(m, pname, pyo.Param(m.ProductSizes, initialize=data[pname]))
    # ... Vars, Constraints, Expressions, Objective lifted from
    #     ReferenceModel.py but written as a ConcreteModel.
    sputils.attach_root_node(
        m, m.FirstStageCost,
        [m.NumProducedFirstStage, m.NumUnitsCutFirstStage],
    )
    m._mpisppy_probability = probability
    return m


def scenario_creator(scenario_name, scenario_count=None):
    data = _scenario_data(scenario_name, scenario_count)
    return _build_model(scenario_name, data, probability=1.0 / scenario_count)


def expected_value_creator(scenario_name, scenario_count=None):
    snames = scenario_names_creator(scenario_count)
    datas  = [_scenario_data(s, scenario_count) for s in snames]
    avg    = _average_param_dicts(datas)
    return _build_model(scenario_name, avg, probability=1.0)
```

Design tension: this grows a ConcreteModel path alongside the existing
AbstractModel in `ReferenceModel.py`. I lean toward the full
concretization because sizes is explicitly being used as a teaching
example and the data-driven ConcreteModel pattern is exactly what users
should copy.

Lighter alternative: keep `ReferenceModel.py` and `scenario_creator`
untouched, and only add `expected_value_creator` as a minimal adapter
that builds a ConcreteModel alongside. Easier review; worse pedagogy.

**Open question for DLW, section 11 below.**

---

## 9. Testing plan

New test module `mpisppy/tests/test_jensens.py`:

1. Positive (lower bound, farmer, 3 scenarios):
   - Run with `--lagrangian-try-jensens-first`.
   - Assert the first `send_bound` value equals the EV problem's optimum
     (solve it independently in the test).
   - Assert this arrives before the iter-0 lagrangian bound.
2. Positive (upper bound, farmer, 3 scenarios):
   - Run with `--xhatshuffle-try-jensens-first`.
   - Assert the first inner bound equals
     `Xhat_Eval.evaluate(ev_nonant_cache)`.
3. Negative (integer recourse):
   - Build a farmer variant with a non-nonant integer Var.
   - Flip `--lagrangian-try-jensens-first`. Expect `RuntimeError` from
     `assert_jensen_integer_safe`.
4. Negative (missing callable):
   - Module lacks `expected_value_creator`. Flip the flag. Expect the
     `cfg_vanilla`-level error.

---

## 10. Documentation

- New `doc/src/jensens.rst` with the convexity warning at the top
  (section 1 of this doc).
- Link from `doc/src/pickling.rst` follow-up list (already a convention
  on this branch).
- Link from the farmer and sizes example README/docstrings to the new
  RST.

---

## 11. Open questions

1. **Sizes refactor scope**: full ConcreteModel rewrite (preferred,
   §8 main path) or minimal adapter (§8 alternative)?
2. **`cfg_vanilla` factory signature**: OK to add
   `expected_value_creator=None` kwarg to each affected factory and
   update `generic_cylinders.py` / `farmer_cylinders.py` callers to pass
   it via `getattr(module, "expected_value_creator", None)`?
3. **Naming**: `expected_value_creator` — confirmed (not `jensens_creator`
   from issue #594). OK?
4. **Flag naming**: kebab-case on CLI
   (`--lagrangian-try-jensens-first`), snake_case in `Config`
   (`lagrangian_try_jensens_first`). Matches existing conventions. OK?
5. **`num_scens` required for farmer EV**: consistent with user
   guidance, but we could fall back to reading `all_scenario_names` if
   that attribute is somehow visible. Confirm: keep the explicit require?
