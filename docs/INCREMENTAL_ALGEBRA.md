# Incremental Algebra: canonical view

Wirkung Math is converging on a compiler for **structured state transitions**.

The recurring pattern is:

```
small description of change
        ↓
operator-specific structure transform
        ↓
new compact delta representation
        ↓
cached downstream state repair
```

A structure is not required to survive an operator unchanged. The compiler may switch representations.

## 1. Additive sufficient-statistic algebra

For any feature map `phi(x)` whose aggregate is

`T = sum_i phi(x_i)`,

a row replacement set `U` updates exactly as

`T' = T - sum_{i in U} phi(x_i) + sum_{i in U} phi(x'_i)`.

This one law includes:

- sums and counts;
- first/second componentwise moments;
- Gram/scatter matrices with `phi(x)=xx^T`;
- cross-covariance / normal equations with `phi(x,y)=xy^T`;
- histograms/count vectors when the feature map is discrete;
- any finite-dimensional additive statistic.

Runtime: `statistics.SufficientStatisticsState`, `statistics.NormalEquationsState`.

## 2. Symmetric low-rank downstream algebra

Row replacement in a Gram matrix gives

`Delta G = X'_U^T X'_U - X_U^T X_U = U C U^T`

with nominal rank at most `2|U|`.

One prepared state

`A^-1 U` and `I + C U^T A^-1 U`

fans out to:

- inverse action;
- one or many linear solves;
- log determinant;
- quadratic forms;
- any consumer expressible through `A'^{-1}b`.

The preparation is shared. Consumers must not rebuild it independently.

Runtime: `symmetric.SymmetricLowRankDelta`, `symmetric.WoodburyCache`.

### Rank reveal before Woodbury

The nominal `2|U|` rank is only an upper bound. QR + a small symmetric eigendecomposition compresses a dependent update before the `r^3` Woodbury core.

Runtime: `SymmetricLowRankDelta.compact`.

## 3. SPD factor-state route

For SPD matrices, storing an explicit inverse is not mandatory.

A row replacement in a Gram matrix can be expressed as rank-1 downdates for old rows followed by rank-1 updates for new rows. The Cholesky factor then provides both solve and logdet without an inverse matrix.

Runtime: `cholesky.CholeskyState`.

The router should benchmark/choose:

- Woodbury on an inverse/action state;
- Cholesky update/downdate on an SPD factor state;
- full refactorization when the update rank is too large.

## 4. Pairwise symmetric operator algebra

If only `e` points of any symmetric pairwise matrix change, only those rows/columns change.

Let `S` select the changed indices and `R = Delta A[U,:]`. Then

`Delta A = S R + R^T S^T - S R_UU S^T`

and therefore

`rank(Delta A) <= 2e`.

This is not RBF-specific. It applies to any symmetric pairwise matrix whose changed rows can be recomputed.

Runtime: `kernel.RowColumnDelta`.

### Implicit selector

The selector `S` is never materialized. Woodbury uses direct indexed columns of `A^-1` plus `A^-1 R^T`.

Runtime: `kernel.RowColumnWoodbury`.

## 5. Sparse-feature × pairwise-kernel cross

For a single changed point `x_u -> x_u + delta`, if only `s` feature coordinates changed,

`||x_u+delta-x_j||^2 = ||x_u-x_j||^2 + 2 delta^T(x_u-x_j) + ||delta||^2`.

With cached old distances this updates the full RBF row in `O(ns)`, not `O(nd)`.

The resulting row/column delta still has rank at most two for one point and can feed the same solver state.

Runtime: `kernel.rbf_single_point_sparse_feature_update`.

## 6. Spectral response algebra

A point edit has a known global Fourier response:

`F(x+Delta x) = F(x) + sum_u Delta x_u F(e_u)`.

The basis column does not need to be stored: its entries form a geometric progression and can be generated on demand.

Runtime: `spectral.SparseDFTState`.

### Shared butterfly paths

For multiple edits, direct response summation costs `O(eN)`. In a stored radix-2 FFT DAG, dirty paths merge. The number of updated stage nodes is bounded by approximately

`sum_s min(N, e*2^s) = O(N(1+log e))`.

This trades checkpoint memory for update compute.

Runtime correctness reference: `spectral.ButterflyFFTState`.

## 7. Dynamic dimension

Replacement is not the only state transition.

- additive statistics support insert/delete directly;
- a pairwise/kernel matrix can append a point/block through a Schur complement;
- solve/logdet can update from the same block state;
- deleting a trailing block can recover the retained inverse without reinversion.

Runtime: `block.BlockAppendCache`.

## 8. Representation switching

The central compiler rule is:

**preserve whatever structure survives; switch representation when another structure becomes cheaper.**

Examples:

- sparse-row + low-rank -> elementwise nonlinearity: rank may disappear, row support survives;
- point edit -> pairwise kernel: point support becomes row/column support + rank <= 2e;
- dirty terminal rows -> global reduction: geometry becomes an additive sufficient statistic;
- Gram delta -> solver/logdet: row support becomes a small symmetric low-rank core;
- sparse time-domain edit -> FFT: sparse support becomes response-basis structure.

Runtime descriptor: `delta_ir.DeltaDescriptor`.

## 9. Numerical lifetime

Persistent incremental state needs a lifetime policy:

- FP64 accumulators for global statistics;
- compensated or hierarchical summation for very long update streams;
- condition monitoring for Woodbury/Schur cores;
- periodic exact rebasing triggered by conditioning/error budget, not by arbitrary update count.

This is a runtime concern, not a reason to discard incremental algebra.
