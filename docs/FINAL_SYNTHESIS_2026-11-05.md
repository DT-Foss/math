# Final synthesis — 2026-11-05

The completed E168–E209 round establishes a broad structured-update runtime, but the strongest consequences appear only after composing the results.

## 1. The central law: compute scales with change description and demand

The useful cost variable is not total state size alone.

A better abstraction is:

`cost = f(delta description, surviving structure, requested output, future consumer, residency, horizon)`.

The measured system now includes:
- local dirty geometry;
- sparse rows/columns;
- low-rank factors;
- additive sufficient statistics;
- order certificates;
- response-basis state;
- requested-output masks;
- versioned deferred state.

## 2. Asymptotic hardware inversion

If a full algorithm has cost `F(N)` and a structured update has cost `D(N,delta)`, then for any finite constant hardware speed advantage `gamma`, whenever

`F(N) / D(N,delta) -> infinity`

there exists a size beyond which the slower-throughput device running the better algorithm beats the faster device running full recomputation.

Measured examples:
- local graph: full area grows as N^2 while fixed dirty work is nearly constant;
- moments: O(n) full versus O(e) update;
- pairwise kernel: O(n^2 d) full versus O(e n d) row/column update;
- factorized matrix functions: O(d^3) refactorization versus low-rank core work.

The direct resident-V100 measurements are empirical instances of this law.

## 3. Local-to-global closure

Spatial locality does not have to end at a global consumer.

A local nonlinear graph can terminate in an additive/global statistic:

`S' = S - S_D + S'_D`.

Therefore a global sum or narrow projection does not force dense output materialization.

This is a major closure property:
`local nonlinear delta -> global sufficient statistic`.

It removes a common boundary in prior incremental-CNN systems where a global-context operator forces full output reconstruction.

## 4. Minimal-state calculus: dense activation caches are not fundamental

For a strictly local deterministic graph with total receptive radius `R` and edit-patch side `p`:

- final dirty output side: `p + 2R`;
- input crop sufficient to reconstruct that dirty output: `p + 4R`.

For the measured architecture `p=8, R=12, channels=16`:
- crop side = 56;
- FP32 crop = 200,704 bytes;
- 2048^2 13-state activation cache = 3.49 GB;
- byte ratio = 17,387×.

David Tom Foss VM experiments recomputed old and new dirty outputs from two 56×56 crops with no full intermediate cache. The measured final dirty regions matched exactly.

This creates a second execution family:
- cached dirty delta: less compute, large persistent state;
- cacheless local reconstruction: slightly more bounded compute, almost no persistent activation state.

The planner chooses between them by memory pressure and expected update horizon.

## 5. Demand sparsity is an independent structure axis

For a fixed linear map `y=A x`, changed inputs `U`, and requested outputs `K`:

`delta y_K = A[K,U] delta x_U`.

Work can therefore scale with **edit size × requested-output size** rather than full input/output size.

Observed manifestations:
- requested Fourier coefficients;
- narrow linear consumers;
- selected low-rank blocks;
- requested normalized query rows.

This is the general law behind the direct FFT and normalized-query CPU/V100 routers.

## 6. Deferred reconciliation: streaming does not imply eager maintenance

A persistent state can be versioned.

For normalized bilinear state:
1. append K/V changes to a compact edit log;
2. leave unrequested query rows stale;
3. on access, apply only edits since that row's version;
4. if old log prefixes are discarded, invalidate affected rows and rebuild exactly on next access.

Thus unrequested outputs cost zero maintenance work.

This composes temporal sparsity, consumer sparsity and edit coalescing.

## 7. Representation superposition

A delta should retain every cheap descriptor that remains valid instead of committing early to one representation.

Example: a few RBF training points changing in only a few features simultaneously have:
- sparse point support;
- sparse feature support;
- row/column matrix support;
- rank <= 2e;
- possibly narrow downstream demand.

Different consumers may choose different views of the same logical delta.

The runtime should therefore hold a **portfolio of representations**, not a single delta type.

## 8. Whole-graph planning, not local routing

The optimal choice depends on future consumers.

The canonical example:
- RBF update -> dense K requested: V100 full wins after only a few edits;
- same RBF update -> Solve requested: keep rank<=2e; CPU beats V100 even at 16 edits.

Represent `(device, representation, residency)` as planning states and conversions/rewrite executions as weighted transitions. The consumer representation is the goal.

Implementation: `RepresentationPlanGraph`.

## 9. State lifetime is a hardware-planning dimension

The measured eager pinned GPU->CPU activation handoff amortizes in 6–8 updates.

But the new cacheless route can eliminate that multi-GB handoff for local-summary consumers. A third option is demand-paging only required local state/crops.

The planner must therefore reason over time:

`Total(strategy,H) = setup(strategy) + H * update(strategy)`.

A realistic lifecycle becomes:

`GPU cold/full -> CPU hot/incremental -> optional GPU rebase -> CPU hot again`.

Rebase is triggered when accumulated support/rank/state cost crosses the full-compute route.

## 10. State residency can be virtualized

When only a small local crop is required, the backing state need not reside in CPU DRAM.

For the 56×56×16 FP32 crop:
- bytes = 200,704;
- at 13.2 GB/s raw transfer time is ~0.015 ms before launch/gather overhead.

Therefore GPU memory, host DRAM, and NVMe can be treated as tiers of one logical state store. Tile layout and demand-paging become part of the execution planner.

## 11. Lazy Gram is the natural continuation of sufficient statistics

A row replacement produces:

`delta G = X_new^T X_new - X_old^T X_old = U V^T`.

If downstream consumers are Solve, Logdet, quadratic forms or selected Gram entries, rewriting the dense d×d Gram matrix is unnecessary.

Keep:

`G = G_base + U_all V_all^T`.

The streaming Woodbury core consumes the factors directly; selected Gram blocks are materialized only on demand.

Implementation: `LazyGramState`.

## 12. One prepared delta should fan out to every matrix-function consumer

The 67,012× prepared-delta result is a systems signal: preparation itself is a reusable state node.

The same low-rank update can feed:
- solve(s);
- logdet;
- inverse action;
- quadratic forms;
- selected entries;
- downstream projections.

Preparing `A^-1 U` separately for each consumer is redundant work.

## 13. Streaming Woodbury core closes the solver lifetime gap

The small Woodbury core itself is incrementally maintainable.

Appending a new rank-q block uses a Schur update of the old core inverse. Old `A^-1 U` columns are retained. The same Schur complement updates the determinant correction:

`det(M_new)=det(M_old) det(S)`.

Thus streaming Solve and Logdet share one evolving core.

## 14. Immediate application-level closures

### Dynamic Gaussian-process state
The measured RBF -> rank<=2e -> Solve/Logdet path already contains the expensive core of exact GP regression:
- kernel state;
- alpha solve;
- log determinant / marginal likelihood;
- sparse changed observations.

Requested predictions add a demand-sparse consumer.

### Dynamic graph / PDE state
An undirected Laplacian edge-weight change is rank one:

`delta L = delta w (e_i-e_j)(e_i-e_j)^T`.

The same solver/logdet infrastructure applies to dynamic networks, circuits and discretized systems.

### Second-order statistics / optimization
Incremental Gram/Fisher-like state plus solve/logdet gives a direct path to dynamic least squares, recursive second-order methods and covariance/preconditioner maintenance.

## 15. The missing third hardware regime: GPU delta

Current direct comparisons mostly measure:
- CPU structured delta;
- GPU full recomputation.

There is an obvious middle regime still to build:
- GPU structured delta.

The final hardware phase diagram should therefore contain at least:
`CPU delta <-> GPU delta <-> GPU full`.

The representation calculus remains the same; only executor cost changes.

## 16. Remath enables automatic discovery of the calculus

Remath already identifies mathematical meaning and execution structure.

The intended compiler is:

`program -> Remath graph -> structure portfolio -> rewrite closure -> whole-graph representation planner -> residency/horizon planner -> CPU/GPU executor`.

This turns the manually discovered E168–E209 laws into automatically recognized program transformations.

## 17. What is prior art and what the strongest contribution is

Prior work contains important individual ingredients:
- self-adjusting computation / dependence-graph change propagation;
- incremental view maintenance and factorized linear-algebra deltas;
- delta CNN inference;
- dynamic attention maintenance;
- event-based / dynamic Fourier updates.

The strongest contribution candidate in this project is the **closure and planner**:

1. multiple exact delta representations across local nonlinear, pairwise nonlinear, normalized, spectral and matrix-function families;
2. representation switching instead of forced materialization;
3. local-to-global closure through sufficient statistics;
4. demand-aware consumers;
5. persistent streaming state;
6. direct CPU↔resident-GPU crossover measurements;
7. lifetime/residency routing;
8. automatic semantic recognition through Remath.

That is substantially larger than any single optimized kernel.
