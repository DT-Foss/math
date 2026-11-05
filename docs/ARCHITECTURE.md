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


## Dirty graph compiler

For local nonlinear graphs, the runtime preserves geometry instead of collapsing support into one rectangle:

1. dirty indices are converted to connected components;
2. components are expanded by the graph receptive field;
3. physically overlapping cones are coalesced;
4. remaining cones are partitioned into execution tiles by a calibrated cost model;
5. only those tiles execute;
6. cached global consumers are repaired from the dirty terminal slices.

The tile partition is solved globally with dynamic programming under

`T = launch_cost * tile_count + cell_cost * total_hull_width`.

The E186 calibration gives `launch_cost≈3.4347 ms` and `cell_cost≈0.0204997 ms/cell`, hence `gap*=167.55` for that executor.

This representation lets nonlinear local layers retain sparse geometry even though low-rank structure may be destroyed by the nonlinearity.
