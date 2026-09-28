# A freshly declared Pyomo `ConfigValue` cannot be pickled

**Status:** upstream Pyomo issue, not filed. Affects mpi-sppy today.
**Verified against:** Pyomo 6.10.0, dill 0.3.8, Python 3.12.

## Symptom

Any object holding an mpi-sppy `Config` with unresolved defaults fails to
serialize, with an error that names nothing useful:

```
_pickle.PicklingError: args[0] from __newobj__ args has the wrong class
```

Three lines reproduce it, with no dill and no mpi-sppy:

```python
import io, pickle
from pyomo.common.config import ConfigValue
pickle.dump(ConfigValue(default=1, domain=int), io.BytesIO())
```

## Cause

A freshly declared `ConfigValue` is not one — it is an
`UninitializedConfigValue`, with mro
`[UninitializedConfigValue, UninitializedMixin, ConfigValue, ConfigBase, object]`.
The first read of `_data` resolves the default and executes
`self.__class__ = _mro[2]` in `UninitializedMixin`
(`pyomo/common/config.py`), promoting the instance in place.

That collides with pickle protocol 2 and later. `copyreg` captures
`args[0] = obj.__class__` *first*, then `__getstate__` walks the slots — and
`ConfigBase.__slots__` lists `_data` sixth, with its own comment reading
*"__getstate__ relies on this field ordering. Do not change."* So the class
flips mid-reduce and `save_reduce`'s `cls is not obj.__class__` check fires.

The decisive tell: touching `_data` **without** pickling also makes the next
attempt succeed. It is not the failed attempt that repairs anything — it is any
first read.

## Consequences worth knowing

- **It is not a dill quirk.** The standard `pickle` module fails identically.
  Two independent investigations concluded the opposite, in mirror image, by
  serializing the *same object* twice in one script: the first attempt fails
  and the second succeeds, whichever library goes first. **Always use a fresh
  object per attempt when testing this.**
- Each failed attempt resolves exactly one entry, so an object needs roughly
  one attempt per unresolved entry — `Config().popular_args()` succeeds on
  attempt 31, and a real `stoch_distr` scenario model on attempt 112. A retry
  loop therefore appears to "eventually work".
- **Intra-process only.** Every fresh process fails on the first attempt, so
  mpi-sppy's exposure is deterministic; nothing is flaky in production.

## How it reaches mpi-sppy

A Pyomo rule written as a nested function inside a `scenario_creator` that
reads `cfg` directly captures the entire `Config`. Pyomo keeps the rule
function on the component it built, so the `Config` becomes reachable from the
model and the model stops being serializable — breaking
`--pickle-scenarios-dir`, `--pickle-bundles-dir` and checkpointing.

The fix in the model is to read what the rule needs into plain locals before
defining it, so the closure captures values rather than the `Config`. See
issue #828 (documenting the constraint) and #830 (which fixed the in-repo
example that demonstrated it). `utils/pickle_bundle.py::describe_dill_failure`
now names the offending rule and captured variable when this happens.

## If someone fixes it upstream

Candidates: resolve the class promotion in `__getstate__` before the slot walk,
or give `UninitializedMixin` its own `__reduce__`. Re-verify against Pyomo
`main` first — this was measured on 6.10.0.
