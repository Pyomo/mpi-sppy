# Checkpointing: what was tried and rejected

Companion to `../checkpointing_design.md`. That document says what the design
*is*; this one records what it is **not**, and why, so that ideas which look
obviously better in isolation are not re-attempted from scratch.

Each entry below was reached by measurement, and the measurement is given.

## Snapshot at termination, from `post_everything`

**Tried, failed, removed.** This was the original phase 1a design and it
survives in some of this PR's history.

`iterk_loop` computes xbar, updates `W`, runs the `miditer` extension hook, and
*may break* — user converger, convergence threshold, `--time-limit` — and only
then solves. A run ending through one of those breaks leaves the models
describing half an iteration: `W` at iteration *k*, nonants still at *k−1*'s
solve. `--time-limit`, the trigger the planned-stop use case is built on, exits
that way every time.

Measured: resuming such a checkpoint diverged from an uninterrupted run by
**37.8** on farmer, versus 0.0 for a clean iteration-limit exit.

## Rewinding the interrupted iteration

**Tried, failed, removed.** The natural repair for the above: cache the
iterate at `enditer` and roll back the half-finished iteration before writing.

It works for what it covers and cannot be made to cover enough. `miditer` gives
every extension a chance to change model state before the break — the rho
updaters change `rho`, `fixer` and `relaxed_ph_fixer` change nonant fixedness,
`integer_relax_then_enforce` changes domains, `cross_scen_extension` adds cuts.
Any rewind list is a list of the extensions someone has thought about so far.

Measured: with `W`, `xbars`, `xsqbars` and `z` cached, a checkpoint labelled
iteration 4 still carried iteration **5**'s rho. The pathology was not fixed,
only relocated.

**Therefore:** write only where the state is already coherent. `enditer` fires
after the solve, so a checkpoint written there always describes a completed
iteration, whatever extensions are loaded and whatever they touched. The
invariant needs no knowledge of any extension, which is the whole point.

## Checkpointing iteration 0

**Considered, declined.** A run that ends before completing iteration 1 now
publishes nothing, which loses the iteration-0 solve.

The obvious fix — write from `post_iter0_after_sync` — is wrong: `Iter0`
splices the `W`/proximal terms into the objective *after* that hook, so the
checkpoint captures a model whose objective is not the one PH goes on to
iterate, and the resume branch disarms the deferred attach, so a resume from it
has no proximal term at all.

Measured: **330** divergence. Caught by the acceptance matrix within minutes of
the hook being added.

Preserving iteration 0 would need a new core hook at the true end of `Iter0`.
That remains available if the lost work ever matters enough.

## An allowlist of "structural" options

**Tried, inverted.** The resume fingerprint originally named the options that
must match. It silently missed everything a model's own `inparser_adder`
registers, none of which reaches `opt.options`.

Measured: a farmer LP checkpoint resumed with `--farmer-with-integers` exited 0
and reported the LP bound. The user asked for a MIP and quietly got something
else, because dill-reload replaces the models wholesale and the new kwargs are
ignored.

**Therefore:** every cfg entry is compared except a named exempt list. A new
option is checked by default and becomes exempt only by explicit decision —
the safe direction for a mechanism whose job is refusing a resume that would be
silently wrong.

## What the per-iteration write costs

The accepted trade, measured over ten iterations against the same run without
`--checkpoint-dir`:

| instance | no ckpt | with ckpt | overhead |
|---|---|---|---|
| farmer, 3 scenarios (LP) | 0.62 s | 0.88 s | 43% |
| farmer, 50 scenarios (LP) | 1.34 s | 4.85 s | 262% |
| sizes, 3 scenarios (MIP) | 2.28 s | 3.15 s | 38% |
| sizes, 10 scenarios (MIP) | 5.05 s | 7.46 s | 48% |

Roughly 7–25 ms per scenario per iteration: negligible for the large-MIP target
where a solve takes minutes, dominant for many cheap scenarios.

## What this does to the planned phase 1b triggers

Writing every iteration **subsumes** `--checkpoint-every-seconds` and the
anticipatory `--checkpoint-before-seconds`: a checkpoint from the last
completed iteration always exists, so neither has a gap left to fill.

The trigger still worth building was `--checkpoint-every-iterations K`, and its
meaning had **inverted**. It was designed as insurance — write *more* often. It
is now a cost control — write *less* often, to buy back the overhead above on
models with many cheap scenarios. It has since been implemented on exactly
those terms: every K-th completed iteration by absolute number, losing up to
K−1 iterations on an unplanned stop, plus an unconditional write at the last
iteration of an exhausted iteration limit (that iterate is coherent, already in
memory, and resuming with a raised limit is the ordinary way to extend a
study).
