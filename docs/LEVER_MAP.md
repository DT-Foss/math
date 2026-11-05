# Lever map

This is the active map of structure-preserving compute replacement.

## Implemented

### Representation transformations
- sparse × low-rank
- lazy low-rank through linear maps
- selective block materialization
- sparse-row structure surviving elementwise nonlinear barriers
- changed points -> row/column + rank<=2e pairwise delta
- dirty geometry -> sufficient-statistic global repair
- sparse signal -> Fourier response / dirty butterfly DAG
- row replacement -> Gram low-rank core
- Gram low-rank core -> shared solve/logdet/inverse action

### Persistent state forms
- normalized output + log partition
- additive FP64 sufficient statistics
- Gram / normal equations
- factor ledgers with rank compaction
- inverse-action Woodbury cache
- Cholesky factor
- Schur block state
- dirty component geometry
- segment reduction tree
- coalesced replacement log

### Scheduling
- full / sparse / sparse×low-rank router
- dirty component dynamic-programming tile partition
- thread envelope should be conditioned on dirty geometry
- rank reveal before Woodbury
- output materialization only at actual consumer

## High-value next levers

### 1. Rank-reveal every downstream matrix delta
Nominal rank bounds such as `2e` are not execution ranks. Compress before any `r^3` small-core work.

### 2. One delta preparation, many consumers
Treat prepared delta state as a node in the execution graph. Gram repair, solve, logdet, quadratic forms and multiple RHS must share it.

### 3. Query-selective materialization
If only rows/columns/outputs are consumed, evaluate only the requested block from factors instead of materializing the state.

### 4. Event coalescing before algebra
Repeated writes to one row collapse to first-old / last-new. Cancellation can eliminate the edit entirely before any kernel executes.

### 5. Associative-reduction trees
For reductions lacking an inverse, a segment tree replaces O(n) recomputation with O(log n) updates. Matrix-product trees extend this to transfer operators and recurrence chains.

### 6. Dynamic spectral DAG
Direct sparse DFT update is O(eN). A checkpointed FFT DAG shares merged dirty paths, with updated-node count O(N(1+log e)). The reference exists; the performance path needs a compiled microkernel/checkpoint policy.

### 7. Pairwise-kernel feature sparsity
Point support and feature support should be multiplied, not treated separately: sparse point edits whose feature delta has support s permit O(ens) kernel-row updates instead of O(end).

### 8. Dynamic shape
Insert/delete should route through additive statistics and block Schur state rather than forcing rebuilds.

### 9. Eigenspace / PCA state
A low-rank Gram/covariance delta should feed a rank-k eigensystem update instead of a new O(d^3) decomposition. This is the major unimplemented statistics consumer.

### 10. Numerical lifetime compiler
Use FP64/compensated summaries, condition estimates, and routed rebase points. Rebase is maintenance, not failure.

### 11. Pairwise-matrix applications beyond RBF
The row/column rank<=2e law applies to arbitrary symmetric pairwise matrices, graph kernels, distances after symmetric transform, and Laplacian-like states. Downstream solve/logdet/quadratic forms reuse the same machinery.

### 12. Dynamic normalized bilinear row/column cross
A small Q/K/V row replacement creates full dirty query rows plus sparse changed key/value columns. Clean rows update from normalized sufficient statistics; only changed query rows recompute.


## Newly closed gaps

### 13. True 2-D dirty geometry
The strongest local/V100 measurements are two-dimensional. `DirtyGraphCompiler2D` now represents dirty components as rectangles, expands receptive fields in 2-D, and minimizes

`sum_tiles [launch_cost + cell_cost * area(hull(tile_group))]`.

For <=12 components it searches the exact set-partition optimum; larger sets use a deterministic positive-saving agglomerative fallback.

### 14. Streaming solver delta ledger
Repeated low-rank matrix updates are concatenated against one base solver rather than building an ever-deeper chain of nested Woodbury wrappers. `LowRankUpdateLedger` can rank-reveal the accumulated update by QR + small SVD and regenerate one compact solver cache.

### 15. Consumer-aware hardware routing
The CPU/GPU decision includes `required_representation`.

The measured RBF example proves why:
- full dense K at n=8192 crosses back to V100 after only a few edits;
- RBF -> solve can keep rank<=2e and still beats V100 full kernel+solve by 16.57× / 11.51× / 4.54× for e=1/4/16 at n=4096.

### 16. Remath semantic front-end
Remath's recognized mathematical meaning now feeds Wirkung through `remath_bridge.py`. Formula families such as GEMM, Softmax, RMSNorm, Attention and elementwise activation are mapped to the corresponding delta rewrite family before the cost/hardware router runs.

The intended compiler path is now:

`program -> Remath execution/formula graph -> Wirkung structure transform -> cost model -> CPU/GPU executor`.
