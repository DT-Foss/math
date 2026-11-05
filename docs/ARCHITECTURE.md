# Architecture

Wirkung Math treats **state transitions**, not dense tensors, as the primary execution object.

## Core state model

A value is represented as a base plus structured deltas. A delta may carry multiple simultaneous structure coordinates:

- sparse support `U`;
- low rank `A B^T`;
- local/dirty dependency cone;
- row scaling;
- normalized-state statistics `(output, logZ)`;
- exact fallback capability when the structured representation becomes numerically or computationally unfavorable.

The runtime chooses the cheapest exact representation before materializing anything.

## Current exact rewrite closure

- sparse value update;
- sparse × low-rank update;
- right-linear composition without dense materialization;
- residual addition by factor concatenation;
- factor rank compaction by QR + small-core SVD;
- stable normalized weighted-sum update with cancellation-gated row rebasing;
- exact structured RMSNorm rewrite;
- exact structured LayerNorm rewrite;
- local linear operator composition and dirty-cone propagation.

## Router principle

The router compares *work implied by the representation*, not tensor size alone. For sequence length `n`, edit count `e`, width `d`, rank `r`:

- full: `n^2 d`
- sparse: `n e d`
- sparse × low-rank: `n r (e+d)`

The sparse×low-rank route beats direct sparse when `r < ed/(e+d)`.

## Materialization rule

Dense materialization is an explicit terminal/fallback operation, never an implicit step after every rewrite. `FactorDelta` therefore propagates across linear maps as factors.
