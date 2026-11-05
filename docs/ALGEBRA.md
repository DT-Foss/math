# Algebra registry

This is an index. Executable truth lives in `src/wirkung_math/` and the test suite.

## A001 — Sparse × low-rank value update

For `ΔV_U = A B^T`:

`ΔN = E[:,U] A B^T = C B^T`, with `C = E[:,U] A`.

Work changes from `O(n e d)` to `O(n e r + n r d)`.

Crossover: `r* = ed/(e+d)`.

Implementation: `lowrank.sparse_lowrank_update`.

## A002 — Factor propagation through a linear map

If `ΔX = C B^T`, then for `Y=XW`:

`ΔY = C (B^T W) = C (W^T B)^T`.

Implementation: `FactorDelta.compose_right`.

## A003 — Normalized weighted-sum state update

With old output `O`, old log partition `L`, edited columns `U`:

`a=exp(S_U-L)`, `b=exp(S'_U-L)`, `q=1-sum(a)+sum(b)`.

`O'=(O-aV_U+bV'_U)/q`, `L'=L+log(q)`.

Rows where `q` is cancellation-prone are exactly rebased by stable row recomputation.

Implementation: `normalized.mass_gated_update`.

## A004 — RMSNorm under a low-rank delta

For `X'=X+AB^T`, the new normalization scale is computed from `XB` and `B^TB`; the output is exactly a row-scaled old normalized base plus a rank-r factor.

Implementation: `norms.rmsnorm_lowrank_update`.

## A005 — LayerNorm under a low-rank delta

Center the basis `B` to `Bc=B-mean(B)`. The centered delta remains `A Bc^T`, and the variance update uses the same rank-r quadratic form as RMSNorm.

Implementation: `norms.layernorm_lowrank_update`.

## A006 — Factor compaction without dense materialization

For `Δ = L R^T`, compute reduced QR factorizations `L=Q_L R_L`, `R=Q_R R_R`, then SVD only the small core `R_L R_R^T = U Σ V^T`.

The recompressed factors are `Q_L U sqrt(Σ)` and `Q_R V sqrt(Σ)`.

This recovers the numerical rank of an accumulated factor ledger without constructing `Δ`.

Implementation: `FactorDelta.compact`.


## A007 — Geometric dirty-component propagation

A sparse support set is decomposed into connected components before receptive-field expansion. Components are kept separate unless their propagated cones overlap or the executor cost model makes filling the gap cheaper than paying another launch.

For adjacent cones separated by gap `g`, the two-tile versus merged decision under the calibrated linear model is:

`merge iff cell_cost * g <= launch_cost`.

Implementation: `dirty.GeometryCostModel`, `compiler.DirtyGraphCompiler`.

## A008 — Optimal dirty-tile partition

For ordered cone components, choose contiguous groups minimizing:

`sum_groups [launch_cost + cell_cost * hull_width(group)]`.

The implementation uses dynamic programming over all possible final groups, avoiding greedy-order artifacts.

Implementation: `dirty.schedule_components_optimal`.

## A009 — Cached global consumer repair

For a cached global sum `S=sum_i x_i` and dirty terminal set `D`:

`S' = S - sum_{i in D} x_i + sum_{i in D} x'_i`.

For a narrow linear consumer `G=sum_i x_i W`:

`G' = G + (sum_{i in D}(x'_i-x_i)) W`.

Only dirty terminal rows are read.

Implementation: `dirty.update_global_sum`, `dirty.update_projected_global_sum`.


## A010 — Additive sufficient-statistic replacement

For `T=sum_i phi(x_i)` and replacement set `U`:

`T' = T - sum_{i in U} phi(x_i) + sum_{i in U} phi(x'_i)`.

This covers sums, component moments, Gram/scatter, cross-covariance and normal equations.

Implementation: `statistics.SufficientStatisticsState`, `statistics.NormalEquationsState`.

## A011 — Symmetric Gram replacement delta

`Delta G = X'_U^T X'_U - X_U^T X_U = U C U^T`

with nominal rank at most `2|U|`. The nominal factor is rank-revealed before downstream use.

Implementation: `symmetric.SymmetricLowRankDelta`.

## A012 — Shared Woodbury fanout

For `A'=A+UCU^T`, cache `A^-1U` and `M=I+C U^T A^-1U` once.

`A'^{-1}B = A^-1B - A^-1U M^-1 C U^T A^-1B`.

The same cache supplies solve, inverse action, logdet and quadratic forms. No `C^-1` is required.

Implementation: `symmetric.WoodburyCache`.

## A013 — Pairwise row/column delta

If only `e` points of a symmetric pairwise matrix change and `R=Delta A[U,:]`:

`Delta A = S R + R^T S^T - S R_UU S^T`.

Therefore `rank(Delta A) <= 2e`.

Implementation: `kernel.RowColumnDelta`.

## A014 — Implicit selector Woodbury

The pairwise selector `S` is not materialized. Indexed inverse columns `A^-1[:,U]` and `A^-1 R^T` generate the Woodbury state directly.

Implementation: `kernel.RowColumnWoodbury`.

## A015 — Sparse-feature RBF row update

For `x_u -> x_u+delta` with `delta` supported on only `s` features:

`d_j'^2 = d_j^2 + 2 delta^T(x_u-x_j) + ||delta||^2`.

With cached old distances, an RBF row costs `O(ns)` rather than `O(nd)`.

Implementation: `kernel.rbf_single_point_sparse_feature_update`.

## A016 — Dynamic Fourier response

A sparse time-domain replacement updates a cached DFT by adding the analytic Fourier response of each point delta. Basis columns are generated on demand.

A checkpointed radix-2 DAG further shares overlapping downstream butterfly paths between multiple edits.

Implementation: `spectral.SparseDFTState`, `spectral.ButterflyFFTState`.

## A017 — Dynamic block Schur update

For `A_new=[[A,B],[B^T,D]]`, define Schur complement `S=D-B^T A^-1 B`. Solve, inverse action and logdet follow from the cached block state without rebuilding the full factorization.

Implementation: `block.BlockAppendCache`.

## A018 — SPD Cholesky update/downdate

A Gram row replacement is a sequence of `-x_old x_old^T` downdates and `+x_new x_new^T` updates. Maintaining Cholesky directly gives solve and logdet without storing an explicit inverse.

Implementation: `cholesky.CholeskyState`.

## A019 — Representation switching

Delta structure is operator-relative. An operator may convert one structure into another rather than preserving or destroying it globally.

Implementation scaffold: `delta_ir.DeltaDescriptor`.


## A020 — Dynamic normalized bilinear row/column update

Replacing `e` rows of Q/K/V changes:
- `e` complete bilinear score rows from Q;
- `e` score columns on every clean row from K;
- `e` value rows from V.

Clean rows use the cached normalized output/log-partition replacement algebra. Only changed query rows and numerically gated fallback rows require full row recomputation.

Implementation: `bilinear.DynamicNormalizedBilinearState`.

## A021 — Associative reduction tree

For any associative binary operator, store a balanced reduction tree. A point replacement changes only one root path, requiring `O(log n)` operator applications; the full aggregate remains at the root.

This covers reductions without an algebraic inverse and noncommutative transfer/matrix-product chains.

Implementation: `reduction.SegmentReductionTree`.

## A022 — Event coalescing

Repeated writes to one row are reduced to the first old value and the final new value before downstream algebra. If the final value returns to the original, the edit vanishes.

Implementation: `coalesce.RowReplacementLog`.

## A023 — General nonsymmetric Woodbury

For `A'=A+UV^T`:

`A'^{-1}=A^{-1}-A^{-1}U(I+V^T A^{-1}U)^{-1}V^T A^{-1}`.

The same small core also supplies the determinant lemma. No symmetry assumption is required.

Implementation: `woodbury.GeneralWoodburyCache`.

## A024 — Selective factor materialization

For `Delta X = AB^T`, a consumer requesting only rows `I` and columns `J` evaluates

`Delta X[I,J] = A[I] B[J]^T`

without constructing the full delta.

Implementation: `FactorDelta.materialize_block`.


## A025 — Two-sided sparse linear response

For `y=A x`, input edit support `U` and requested outputs `K`:

`delta y_K = A[K,U] delta x_U`.

Neither the full input response nor the full operator needs to be materialized if the requested block can be generated/fetched directly.

Implementation: `demand_linear.sparse_requested_linear_delta`.

## A026 — Deferred normalized-query reconciliation

K/V edits are appended without eagerly touching every query row. Each requested row applies only outstanding edit batches to its stored normalized sufficient state. Stale rows may be invalidated when edit-log prefixes are compacted and rebuilt exactly on next access.

Implementation: `lazy_normalized.LazyNormalizedKVState`.

## A027 — Cacheless local reconstruction

For a local graph of total receptive radius `R`, an edit patch side `p` has final dirty side `p+2R`. Exact reconstruction of that final dirty output requires only an input crop side `p+4R`.

This trades two bounded local evaluations (old/new) for eliminating all dense intermediate activation state.

Implementation: `cacheless_local.LocalGraphDependency`.

## A028 — Lazy Gram / scatter representation

For row replacements:

`delta G = X_new^T X_new - X_old^T X_old = U V^T`.

Keep `G = G_base + U_all V_all^T` rather than rewriting the dense matrix. Solve and logdet consume the shared streaming Woodbury core; arbitrary requested blocks use factor submatrices.

Implementation: `lazy_gram.LazyGramState`.

## A029 — Streaming determinant correction

When the Woodbury core grows by a new block

`M_new = [[M,A],[B,D]]`,

with Schur complement `S=D-B M^-1 A`:

`det(M_new)=det(M)det(S)`.

Thus the determinant correction is updated during the same block append that updates `M^-1`; no independent factorization is required.

Implementation: `update_ledger.IncrementalWoodburyLedger.logdet_update`.
