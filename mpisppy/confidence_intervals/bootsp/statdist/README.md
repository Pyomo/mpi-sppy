# statdist (trimmed)

This is a trimmed port of the **statdist** statistical-distribution library,
bundled here for the smoothed bootstrap methods in
`mpisppy.confidence_intervals.bootsp`.

## What is here

Only the **univariate** distributions and their support modules:

- `base_distribution.py` — the distribution base classes and helpers
- `distributions.py` — the univariate distribution classes (uniform, normal,
  student-t, Gaussian kernel, epi-spline, empirical, discrete)
- `distribution_factory.py` — name → class registry (`distribution_factory`)
- `splines.py` — epi-spline fitting (builds a small Pyomo model)
- `utilities.py`, `sampler.py` — memoization/context helpers and the sampler

## What was dropped

The multivariate machinery — `copula.py`, `vine.py`, `bicop.py`, and the
multivariate distribution classes in `distributions.py` — is **not** included.
Dropping it also removes the `from scipy.stats import mvn` import (removed in
scipy 1.14) and the optional `gosm` hook, neither of which the smoothed
bootstrap methods use. scipy is imported lazily (via
`pyomo.common.dependencies`) so the empirical bootstrap path stays scipy-free.

The full library, including the multivariate code, lives in the archived
boot-sp repository: https://github.com/boot-sp/boot-sp

## Provenance

statdist was developed under separate funding, always intended to be
open-source, and shares lineage with the GOSM/Prescient scenario-generation
tools.
