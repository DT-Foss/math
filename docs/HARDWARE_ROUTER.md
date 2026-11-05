# CPU ↔ GPU structure router

The hardware choice is a function of the **required representation**, not just tensor dimensions.

## Local nonlinear residual graph

Measured path: 12 local residual layers, channels=16, patch=8, resident V100/cuDNN full graph versus CPU cached dirty cone.

Small-area fit (256–512):

`T_V100(N) ~= 0.1505 ms + 1.2262e-5 ms * N^2`

with R²≈0.99956. CPU dirty time over the same range is ~1.724 ms.

Predicted crossover: ~358 pixels; direct interpolation of the 320/384 measurements gives ~354.6. Measured bracket:

- 320²: V100 wins.
- 384²: CPU delta wins.
- 512²: CPU delta wins 1.95×.
- 1024²: CPU delta wins 5.51×.
- 1536²: CPU delta wins 13×.

This is a piecewise hardware calibration: do not extrapolate a large-area GPU model through a cuDNN algorithm-regime change.

## Pairwise RBF kernel

At n=8192,d=64:

### Consumer requires full dense K

- e=1: CPU changed-row/column materialization wins 3.75×.
- e=8: V100 full K is already faster.
- e=64: V100 decisively wins.

Interpolated materialization crossover: ~6.7 edits.

### Consumer is a solver

Do **not** materialize K'. Retain the row/column delta as rank <= 2e and feed it directly to Woodbury.

At n=4096 with 8 RHS:

- e=1: CPU structured path wins 16.57×.
- e=4: 11.51×.
- e=16: 4.54×.

The exact same logical data edit therefore routes differently depending on whether the next consumer requires a dense matrix or accepts structured state.

## Core router contract

`route = f(operator, dimensions, delta_structure, required_representation, hardware_calibration)`

A hardware router that ignores `required_representation` will choose the wrong machine for cases such as RBF→Solve.
