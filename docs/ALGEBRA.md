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
