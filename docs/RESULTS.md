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

## Executable verification on David Tom Foss VM — 2026-11-05

Machine-readable record: `results/david_tom_foss_vm_2026-11-05.json`.

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

Independent executable check on the David Tom Foss VM (AMD EPYC 9V74, Float64), n=131072, d=64, 256 dirty rows:

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


## David Tom Foss VM cross-check — dynamic normalized bilinear state

Machine-readable record: `results/david_tom_foss_vm_dynamic_bilinear_2026-11-05.json`.

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

For the 12-layer patch-8 dirty graph, measured small-area points give a hardware crossover bracket of **320 < N* < 384**. A fit over the 256–512 V100 points gives

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


## David Tom Foss VM cross-structure checks — 2026-11-05

Machine-readable record: `results/david_tom_foss_vm_cross_structure_2026-11-05.json`.
Executable benchmark: `experiments/david_tom_foss_cross_structure_bench.py`.

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


### David Tom Foss streaming Woodbury-core check

Machine-readable record: `results/david_tom_foss_vm_streaming_woodbury_2026-11-05.json`.

For d=1024, 8 RHS and sequential rank-1 updates on top of a cached Cholesky base solver, the incrementally maintained Woodbury core keeps old `A^-1 U` columns and updates only the new block:

- accumulated rank 16: solve **13.46×** vs full refactor+solve; update+solve **6.13×**;
- accumulated rank 32: solve **17.50×**; update+solve **8.36×**;
- append cost remains ~1.5–2.5 ms over the measured range;
- max absolute errors stay below 7e-16.

This closes the persistent-state gap for the low-rank solver family as well as for dirty-local execution.


## Final E205–E209 closure — CPU box / resident V100

Machine-readable provenance: `results/imported_e205_e209_final_2026-11-05.json`.

### Final local-graph V100 crossover

For channels=16, layers=12, patch=8:

- 256²: GPU wins (0.928 ms vs 1.731 ms CPU delta)
- 320²: GPU wins (1.427 ms vs 1.727 ms)
- 352²: near parity (1.723 ms vs 1.791 ms)
- 384²: CPU delta **1.15×**
- 512²: **1.95×**
- 1024²: **5.51×**
- 1536²: **13.00×**
- 2048²: **18.97×**

The measured switching interval is 352–384 pixels; interpolation places parity at roughly 359 pixels.

### Streaming normalized sufficient state

Scores / normalization state / K/V state were maintained through 40 sequential updates:

- n=1024: median 5.52×, minimum 2.13×
- n=2048: median 10.69×, minimum 6.59×
- max error ~5.1e-10
- no state drift.

### GPU -> CPU state handoff

Pageable copies amortized after ~36–44 updates. Pinned D2H transfer reached ~13.2 GB/s and reduced the measured eager-cache break-even to **6–8 updates**:

- 1024²: 872 MB, 64.9 ms
- 1536²: 1.96 GB, 149.2 ms
- 2048²: 3.49 GB, 265.4 ms.

### Requested Fourier coefficients versus full V100 FFT

n=1,048,576, 16 changed input values:

- 1 coefficient: CPU **11.40×**
- 16: **11.67×**
- 256: **5.70×**
- 4,096: **3.02×**
- 16,384: **1.42×**
- 65,536: V100 wins
- 262,144: V100 wins.

The measured requested-output crossover lies between 16,384 and 65,536 coefficients.

### Requested normalized queries versus V100

n=8192, heads=2, d=64, 16 changed terms:

- 1 query: CPU **1.18×**
- 16: **1.93×**
- 32: **2.63×**
- 64: **1.65×**
- 128: **1.03×**
- 256: V100 wins
- 1024: V100 wins.

The measured crossover lies between 128 and 256 requested query rows.

## David Tom Foss follow-up: cacheless local reconstruction

Machine-readable record: `results/david_tom_foss_cacheless_local_2026-11-05.json`.

A 12-layer 3×3 Conv/SiLU/residual graph with patch=8 has total receptive radius R=12. The final dirty region has side 32, but it can be reconstructed exactly from an old/new input crop of side

`p + 4R = 56`.

No full intermediate activation cache is required for a consumer that only needs the dirty final output and an incrementally repairable downstream statistic.

Four-thread CPU measurements:

- 256²: 7.35× vs CPU full
- 512²: 77.62×
- 1024²: 197.78×
- 1536²: **533.48×**
- final dirty regions matched exactly in the measured runs
- incremental global-sum repair at 512² matched exactly.

The FP32 input crop is only 200,704 bytes. A 2048² full 13-state activation cache is 3.49 GB: **17,387× more bytes**. At the measured ~13.2 GB/s pinned bandwidth, transferring one such input crop is only ~0.015 ms before launch/gather overhead.

This exposes a new state-residency regime: cacheless dual-crop reconstruction can remove the eager multi-GB activation handoff entirely.

## David Tom Foss follow-up: lazy normalized query reconciliation

Machine-readable record: `results/david_tom_foss_lazy_queries_2026-11-05.json`.

K/V updates are appended to a versioned edit log; only requested query rows reconcile outstanding edits. At n=4096,d=64,e=16, compared with eagerly updating every CPU query state:

- 1 requested row: **44.90×** less update work
- 16: **33.48×**
- 32: **27.34×**
- 64: **22.08×**
- 128: **15.56×**
- 256: **9.51×**
- 512: **5.48×**

Maximum FP32 error in the controlled check was ~4.98e-8.

This composes streaming state with demand sparsity: unrequested outputs may remain stale at zero maintenance cost and are reconciled exactly when accessed.


## E210–E220 continuation — direct V100 scaling and >1000× point

Machine-readable provenance: `results/imported_e210_e220_2026-11-05.json`.

### Composed statistics pipeline versus resident V100

`X -> Gram -> 16-RHS Solve -> Logdet`, with the CPU consuming only changed rows through one shared low-rank delta:

- n=1,048,576: **50.48× / 48.24× / 34.07×** for 1/4/16 edits;
- n=4,194,304: **188.05×** for 1 edit and **130.48×** for 16;
- n=8,388,608, one edit: **225.93×**.

Relative errors in the supplied runs were around 1e-14.

### Local nonlinear residual graph scaling

With the optimized rolling two-state V100/cuDNN executor:

- 3072²: **42.20×**
- 4096²: **63.37×**
- 6144²: **150.25×**
- 8192²: **235.17×**

At 8192²: 755.07 ms V100 full versus 3.21 ms CPU dirty delta; relative error ~1.8e-7.

### Moments + Gram direct record

The GPU baseline was strengthened to fused `var_mean + Gram`.

At n=33,554,432,d=64,e=64:

- V100 full: 61.099 ms
- CPU delta: 0.0530 ms
- **CPU/V100 = 1152.60×**
- relative error ~3.9e-13.

A two-V100 row partition measured 70.55 ms versus 0.125 ms CPU delta, still **562.88×**.

With one changed row, the supplied generic full-run measurement reached **1627.18×** (61.092 ms versus 0.0375 ms). The later isolated specialized rank-1 NumPy path measured 21.43 us, but a full direct V100 validation of that specialization was not included in the supplied excerpt.

## David Tom Foss follow-up: midpoint/difference replacement algebra

Machine-readable record: `results/david_tom_foss_midpoint_state_quotient_2026-11-05.json`.

For row replacement, define `D=new-old`, `P=new+old`:

- `Delta S = sum(D)`
- `Delta Q = sum(D*P)`
- `Delta G = 1/2(D^T P + P^T D)`.

This replaces two Gram products with one GEMM plus symmetrization.

On the David Tom Foss VM, d=64, single-thread BLAS, the midpoint form was **1.40×–1.74×** faster over e=1..64 than the direct old/new quadratic difference.

In a conditioning stress test with baseline scale 1e6 and edit scale 1e-4, max Gram-delta error versus long-double dropped from 6.78e-2 to 3.98e-5: about **1700× less numerical error**.

## David Tom Foss follow-up: coherent low-rank replacement statistics

Machine-readable record: `results/david_tom_foss_lowrank_replacement_stats_2026-11-05.json`.

If `D=new-old=A B^T` has rank r, let `C=A^T(new+old)`. Then:

- `Delta S = B sum(A)`
- `Delta Q_j = <B[j,:], C[:,j]>`
- `Delta G = 1/2(B C + C^T B^T)`
- `rank(Delta G) <= 2r`, independent of the number of changed rows.

Measured examples versus direct midpoint delta statistics:

- e=1024,d=256,r=4: **13.7×** with dense Gram materialized; **36.5×** when keeping the factor;
- e=4096,d=256,r=4: **23.4× / 41.0×**;
- e=4096,d=512,r=4: **22.6× / 49.0×**.

If the update already arrives as factors A,B, the runtime can avoid materializing both D and new rows. That delta-native path was a further **2.6×–7.1×** faster than materializing the factored delta first in the controlled follow-up.

## David Tom Foss follow-up: native rank-1 persistent statistics state

Machine-readable record: `results/david_tom_foss_native_rank1_stats_2026-11-05.json`.

A specialized native midpoint kernel updates persistent `S,Q,G` in place for one row. On the David Tom Foss VM with d=64, pre-bound pointers:

- dense Gram state: ~**1.74 us median**
- packed-upper Gram variant: ~**1.58 us median**.

These are microkernel measurements on a different CPU from the V100 box and are therefore targets for direct box validation, not direct V100 speedup claims.

## Hot-state quotient

For the 33,554,432×64 FP64 statistics workload:

- raw data: exactly **16 GiB**;
- sum + sumsq + dense 64×64 Gram: **33 KiB**;
- hot-state reduction: **508,400×**.

Only 32 KiB of old rows are required for a 64-row replacement. A cold-row store at 1 GB/s would add only ~0.033 ms to the measured 0.053 ms delta compute; against 61.099 ms V100 full this remains over **700×** in the simple bandwidth model.

This means sufficient-state execution can collapse both compute and hot-memory footprint.
