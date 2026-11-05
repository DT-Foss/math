# CPU ↔ GPU structure router

The hardware choice is a function of the **required representation**, not just tensor dimensions.

## Local nonlinear residual graph

Measured path: 12 local residual layers, channels=16, patch=8, resident V100/cuDNN full graph versus CPU cached dirty cone.

Small-area fit (256–512):

`T_V100(N) ~= 0.1667 ms + 1.2223e-5 ms * N^2`

with R²≈0.9989 after including the 352² point. CPU dirty time over 256–512 is ~1.737 ms.

Predicted crossover from the local fit is ~358.4 pixels. The final measured bracket is:

- 320²: V100 wins.
- 352²: near parity; V100 is ~1.04× faster.
- 384²: CPU delta wins 1.15×.
- 512²: CPU delta wins 1.95×.
- 1024²: CPU delta wins 5.51×.
- 1536²: CPU delta wins 13×.
- 2048²: CPU delta wins 18.97×.

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


## Output-demand routing

Output demand is a first-class structural coordinate. For a fixed linear map `y=A x`, changed input set `U` and requested output set `K`:

`delta y_K = A[K,U] delta x_U`.

The full output need not exist.

### Sparse spectrum versus V100 full FFT

n=1,048,576, 16 changed input values:

- requested 1 coefficient: CPU 11.40×
- 16: 11.67×
- 256: 5.70×
- 4,096: 3.02×
- 16,384: 1.42×
- 65,536: V100 wins
- 262,144: V100 wins.

Measured crossover bracket: 16,384–65,536 requested frequencies. Current interpolation is ~45,975.

### Cached normalized queries versus V100 requested-query recomputation

n=8192, heads=2, d=64, 16 changed terms:

- 1 query: CPU 1.18×
- 16: 1.93×
- 32: 2.63×
- 64: 1.65×
- 128: 1.03×
- 256 and above: V100 wins.

Measured crossover bracket: 128–256 query rows. Current interpolation is ~137.

## Time-horizon / state-residency routing

Pinned GPU->CPU activation bootstrap reaches ~13.2 GB/s and amortizes an eager CPU activation cache after 6–8 repeated local updates for the measured 1024²–2048² graph.

However, eager full-state migration is not the only CPU regime.

For a strictly local 12-layer radius-1 graph with patch=8, the complete old/new dirty output is reconstructible from an input crop of side

`p + 4R = 56`.

At channels=16 FP32 this is only 200,704 bytes, versus 3.49 GB for the full 2048² 13-state activation cache. A consumer that needs only the dirty final output plus an incrementally repairable summary can therefore choose a **cacheless local** route with no full activation bootstrap.

The lifetime planner must compare at least:

1. V100 full each update;
2. eager pinned full-state CPU handoff + cached CPU delta;
3. cacheless CPU old/new crop reconstruction;
4. demand-paged local state/crop fetch;
5. eventual GPU rebase when accumulated structure becomes too expensive.

## Whole-graph routing

The route is not chosen per operator in isolation:

`route = f(operator, delta_structure, requested_output, required_representation, expected_horizon, state_residency, hardware_calibration, downstream_consumer)`.

`RepresentationPlanGraph` represents these choices as a shortest-path problem over representation+device states. `HorizonPlanner` handles repeated-update economics.
