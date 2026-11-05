# Results registry

## Imported E168–E171 — CPU box, 2026-11-05

These measurements were supplied from the running CPU-box work and are retained as provenance.

### Normalized bilinear update

- n=2048, edits=1: 69.97×
- n=2048, edits=8: 15.01×
- n=2048, edits=64: 0.76×
- n=4096, edits=64: 2.43×
- n=8192, edits=64: 5.34×
- all 9 initial cases numerically matched; max absolute error about 1.5e-8.

Initial crossover map:

- n=1024: 64 edits = 6.25%
- n=2048: 64 edits = 3.125%
- n=4096: 256 edits = 6.25%
- n=8192: 512 edits = 6.25%

### Lazy Top-K update

After correction of the NumPy-view benchmark bug:

- n=2048: 256 edits = 12.5% remained profitable;
- n=4096: 512 edits = 12.5% remained profitable;
- n=8192: 512 edits remained profitable;
- observed Top-K depth was only 1–6.

Upper-bound search:

- n=4096, 12.5% edits: 1.67×
- n=4096, 25% edits: 0.37×
- n=8192, 6.25% edits: 1.08×
- n=8192, 12.5% edits: 0.57×
- n=8192, 25% edits: 0.37×

### Dirty/local operator composition

- 9/9 tested cases correct;
- best reported speedup: 301.679×.

### Low-rank update

- 12/12 tested cases correct;
- best reported speedup: 2.453×;
- errors about 1e-7.

## Executable verification on Sol VM — 2026-11-05

Machine-readable record: `results/sol_vm_2026-11-05.json`.

Environment: AMD EPYC 9V74, NumPy 2.3.5, Python 3.13.5, single-thread BLAS/OMP.

### E172 — mass-gated normalized state

At n=1024,d=64:

- 6.25% edits: 2.52×
- 12.5% edits: 1.19×
- 25% edits: 0.58×
- 50% edits: 0.32×
- zero fallback rows in this random workload;
- max absolute FP32 error <= 2.69e-7.

### E173 — sparse × low-rank

At n=4096,e=512,d=64, analytic operation-count crossover `r*=56.89`.

Recorded speedups:

- r=1: 7.02×
- r=2: 4.67×
- r=4: 5.07×
- r=8: 3.74×
- r=16: 4.16×
- r=32: 1.39×
- r=64: 1.08×

An earlier same-VM run put r=64 below parity, locating the executor transition around the predicted rank-57 region.

### E174 — factor propagation without intermediate materialization

Rank 4, n=4096,d=64, six linear layers, final materialization included:

- recorded run: 54.89×;
- preceding same-VM run: 113.18×;
- max absolute FP32 difference: 8.88e-7.

### E175 — normalization rewrites

- RMSNorm max absolute error: 1.78e-15
- LayerNorm max absolute error: 3.56e-15

### E176 — nonlinear rank barrier

A rank-4 delta became numerical rank 64 after SiLU at every tested amplitude.

- rank-4 energy retained: about 69–73%
- rank-8 energy retained: about 75–86%

This isolates elementwise nonlinearity as a measured structure-expansion boundary.

### E177 — factor compaction

32 rank-4 residual terms produced nominal rank 128 but shared an 8-dimensional latent space.

QR + small-core SVD recovered rank 8 without dense materialization:

- 16× factor-rank compression;
- 10.35× faster subsequent materialization in the recorded run;
- 12.46× in the preceding run;
- max absolute error: 2.56e-12.


## Imported E185–E187 dirty-graph results — CPU box, 2026-11-05

Machine-readable provenance: `results/imported_e185_e187.json`.

### E185 — connected-component dirty tiling

Across 24 nonlinear residual layers, preserving disconnected dirty components as separate cones produced:

- 2 separated changes: **196.6× vs full recomputation**, **3.68× vs one bounding rectangle**;
- 4 changes: **100.4× vs full**, **55.6× vs bounding rectangle**;
- 8 changes: **50.9× vs full**, **28.0× vs bounding rectangle**;
- all three cases correct.

The operational state is therefore geometric: a dirty set is represented by its connected components and propagated cones, rather than by a single enclosing rectangle.

### E186 — component merge scheduler

Measured two-component routing:

- gaps 0–128: merge;
- gaps 256–384: keep separate;
- at gap 128: 8.996 ms merged vs 9.002 ms tiled;
- at gap 256: 11.583 ms merged vs 9.054 ms tiled;
- at gap 384: 13.024 ms merged vs 9.749 ms tiled.

Fitted law:

`T_merge(g) = 5.8224 ms + 0.0204997 ms * g`

with R²=0.9596 and median tiled cost 9.2571 ms.

This yields an inferred marginal second-tile launch/scheduling cost of about 3.4347 ms and an automatic crossover gap of **167.55 cells**.

The runtime implementation generalizes this to arbitrary component sets with a dynamic-programming partitioner minimizing:

`launch_cost * tile_count + cell_cost * total_processed_hull_width`.

### E187 — local dirty graph into global consumer

The exact update law is:

`S' = S - sum(old_dirty) + sum(new_dirty)`.

For a narrow projection `G=sum_i x_i W`, the runtime uses:

`G' = G + sum(delta_dirty) W`.

Independent executable check on the Sol VM (AMD EPYC 9V74, Float64), n=131072, d=64, 256 dirty rows:

- cached global sum update: **649.36×** vs full reduction;
- cached 8-wide projected reduction: **448.87×** vs full reduction+projection;
- max absolute differences: 1.36e-12 and 9.09e-12 respectively.

This closes the local-to-global path without throwing away the dirty-graph gain at the first global consumer.


## Extended CPU-box breakthrough round — 2026-11-05

Machine-readable provenance: `results/imported_cpu_box_extended_2026-11-05.json`.

### Sparse × low-rank and lazy propagation

- 27/27 sparse×low-rank cases correct.
- Selected supplied cases: 16.20× (n=8192,e=64,r=4), 12.34× (e=256,r=16), 6.21× (e=1024,r=4), and 1.85× even at 50% changed rows for n=2048,r=4.
- Global 100%-row low-rank updates remained profitable for r<d; the materialized-output route crossed near r=d.
- Lazy factor propagation through eight linear operators reached 7.20× at n=8192,d=512 and 6.53× even at rank 64.
- With a narrow consumer, the composed linear path reached 105.22×.
- A paired 20-round ABBA replication of the strongest point measured median 75.35×.
- Against a fully precompiled full linear map, the additional low-rank-delta contribution remained up to 3.54×. Both gains remain part of the composed execution path.

### Dirty nonlinear graph

- Thread-envelope routing: 697× for patch 1, 587× for patch 8, 286× for patch 64.
- End-to-end through a global consumer: up to 601.5× / 502.8× / 171.5× for patches 1/8/64.
- FP64 cached global statistics retained 533.9× / 507.0× / 239.5× with max relative error 1.34e-11.

### Incremental sufficient statistics

Corrected same-input FP32/FP64 moment run: 9/9 correct, max error ~1.5e-14.

At n=1,048,576,d=64:
- 64 row edits: **9,117×**
- 1,024 edits: **1,066×**
- 16,384 edits: **73.3×**

Incremental Gram/covariance: 9/9 correct, relative errors ~1e-15.

At n=1,048,576,d=64:
- 64 edits: **4,874×**
- 1,024 edits: **485×**
- 16,384 edits: **33.3×**

### Low-rank matrix state

Woodbury inverse update:
- 9/9 correct;
- best 28×;
- d=2048,r=2..32 remained about 19–27×.

Direct cached-solve update:
- 18/18 correct, residuals ~1e-14;
- d=2048,r=8: **3,428×** for one RHS, **1,644×** for 16 RHS, **542×** for 64 RHS;
- rank 32 remained roughly 428–1,526×.

Low-rank logdet:
- 9/9 correct;
- d=2048: 309× at rank 2, 193× at rank 8, 144× at rank 32.

### Composed statistics pipeline

One row-change delta feeds Gram + solve + logdet.

Prepared-delta application:
- 6/6 correct;
- up to **67,012×**;
- n=1,048,576,d=256,e=16 still **22,475×**.

Full delta creation + all downstream consumers:
- 6/6 correct;
- n=262,144: 831× / 1,057× / 795× for e=1/4/16;
- n=1,048,576: **4,585× / 3,752× / 3,017×** for e=1/4/16.

### Sparse Fourier updates

- 9/9 correct, relative error 1e-14..1e-13.
- n=1,048,576: 18.1× for 4 edits and 11.2× for 16 edits.

### Nonlinear pairwise kernel

RBF matrix update:
- 9/9 correct;
- n=8192: **720×** for 1 changed point, 144× for 8, 15.3× for 64.

RBF -> dynamic solve:
- 9/9 correct, residuals ~1e-14;
- n=512: 22–71×;
- n=1024: 47–167×;
- n=2048: **382× / 291× / 132×** for e=1/4/16.
- n=4096: **801× / 640× / 230×** for e=1/4/16, all correct with residuals ~1e-13.


## Sol VM cross-check — dynamic normalized bilinear state

Machine-readable record: `results/sol_vm_dynamic_bilinear_2026-11-05.json`.

AMD EPYC 9V74, NumPy/OpenBLAS 5 threads, FP32, d=64. Cached old Q/K/V output and log-partitions are persistent state; setup/state copies were excluded from the timed update.

- n=1024: 51.37× / 15.80× / 1.86× for e=1/8/64.
- n=2048: 32.31× / 28.82× / 15.57×.
- n=4096: **86.57× / 38.46× / 11.81×**.
- zero clean-row fallbacks in all nine random cases.
- max absolute output error <= 2.95e-7.

This composes pairwise row/column change geometry with the normalized sufficient-statistic state rather than rebuilding the full n×n bilinear map.


## Streaming state + direct resident V100 comparisons — 2026-11-05

Machine-readable provenance: `results/imported_cpu_v100_streaming_2026-11-05.json` and `results/direct_v100_vs_cpu_delta_2026-11-05.json`.

### Persistent nonlinear dirty state

Thirty consecutive local edits were applied to the already-updated cache; every step was checked against full recomputation of the current state.

- median full: 90.438 ms
- median delta: 2.296 ms
- median paired speedup: **39.46×**
- worst step: **31.19×**
- max relative error across the stream: ~3.0e-10
- no growing error trend observed.

This establishes the dirty executor as a persistent state transition engine rather than a one-shot cache benchmark.

### Controlled dirty-area scaling

For channels=16, layers=12, patch=8, the final dirty region stays at 1024 cells while full area grows:

- 256²: 16.90×
- 512²: 70.92×
- 1024²: 521.92×
- 1536²: **1158.35×**

At the large sizes, measured executor speedup is roughly half the ideal area ratio.

### Direct CPU delta versus resident V100

GPU inputs/weights are resident and host transfer is excluded.

- local nonlinear residual graph: CPU delta wins **5.5× to 19×** in the supplied large-size comparisons;
- moments + full Gram: 0.0695 ms CPU delta vs 5.115 ms V100 full = **73.6×**;
- solve + logdet after low-rank change: 0.809 ms vs 18.27 ms = **22.6×**.

For the 12-layer patch-8 dirty graph, measured small-area points give a hardware crossover bracket of **320 < N* < 384**. Sol fits the 256–512 V100 points as

`T_V100(N) ~= 0.1505 ms + 1.2262e-5 ms * N²`

with R²≈0.99956, predicting N*≈358; direct interpolation of the 320/384 time difference gives N*≈354.6.

### Dense RBF materialization versus structured RBF solve

At n=8192,d=64 when the consumer demands the full K matrix:

- e=1: CPU changed-row/column materialization wins 3.75×;
- e=8: V100 full K wins;
- e=64: V100 full K wins decisively.

The current interpolated dense-materialization crossover is ~6.7 edits.

But at n=4096 with an 8-RHS solve, keeping the same kernel change as rank<=2e and feeding Woodbury directly gives:

- e=1: **16.57×** CPU over V100 full kernel+solve;
- e=4: **11.51×**;
- e=16: **4.54×**;
- solution differences are ~1e-13.

The router must therefore include the consumer's required representation; edit count alone is insufficient.


## Sol VM cross-structure checks — 2026-11-05

Machine-readable record: `results/sol_vm_cross_structure_2026-11-05.json`.
Executable benchmark: `experiments/sol_cross_structure_bench.py`.

### Sparse-feature RBF

A single changed point with only `s` changed feature coordinates updates its full RBF row from cached squared distances in `O(ns)` rather than `O(nd)`.

Controlled Float64, single-BLAS-thread results:

- n=32,768,d=256,s=1: **55.01×**
- n=32,768,d=256,s=4: **17.29×**
- n=65,536,d=256,s=1: **60.57×**
- n=65,536,d=256,s=4: **11.40×**
- n=65,536,d=256,s=16: **3.07×**

Maximum absolute errors were below 4e-23 in these runs. The resulting pairwise matrix change remains row/column-supported with rank <=2e, so this feature-sparse front-end composes directly with the existing Woodbury solver path.

### Associative reduction tree

For n=1,048,576 scalar values, an exact max reduction stored as a segment tree gave:

- full NumPy max median: 108.513 us
- point-update tree path: 5.308 us
- **20.44×**
- exact root aggregate.

This validates a separate incremental class for associative reductions that do not need an algebraic inverse.


### Sol streaming Woodbury-core check

Machine-readable record: `results/sol_vm_streaming_woodbury_2026-11-05.json`.

For d=1024, 8 RHS and sequential rank-1 updates on top of a cached Cholesky base solver, the incrementally maintained Woodbury core keeps old `A^-1 U` columns and updates only the new block:

- accumulated rank 16: solve **13.46×** vs full refactor+solve; update+solve **6.13×**;
- accumulated rank 32: solve **17.50×**; update+solve **8.36×**;
- append cost remains ~1.5–2.5 ms over the measured range;
- max absolute errors stay below 7e-16.

This closes the persistent-state gap for the low-rank solver family as well as for dirty-local execution.
