# Wirkung Math

**Structured delta algebra and execution research.**

The goal is not to make a CPU imitate a GPU kernel-by-kernel. The goal is to replace dense recomputation with exact state transitions whose work scales with the *intrinsic structure of the change*.

## What is implemented now

- exact normalized weighted-sum updates with cancellation-gated rebasing;
- sparse × low-rank delta algebra and closed crossover rule;
- factor propagation through linear maps without dense materialization;
- exact structured RMSNorm and LayerNorm rewrites;
- local operator composition, connected-component dirty cones, and geometry-aware tile scheduling;
- factor-ledger compaction without dense reconstruction;
- a cost router for full vs sparse vs sparse×low-rank execution;
- a `DirtyGraphCompiler` that compiles dirty support into optimal local tiles;
- exact cached global-reduction / projected-reduction delta consumers;
- FP64 sufficient-statistics state for moments, Gram/covariance and normal equations;
- shared Woodbury state for solve/logdet/inverse-action fanout;
- implicit row/column low-rank updates for pairwise kernels;
- Cholesky update/downdate and Schur block insert/delete routes;
- sparse DFT and dynamic butterfly-DAG reference paths;
- a typed representation-switching delta IR;
- a consumer-aware CPU↔GPU hardware router calibrated against resident V100 baselines;
- output-demand routing (`edit support × requested outputs`);
- cacheless local reconstruction and lifetime/residency planning;
- lazy normalized query reconciliation;
- lazy Gram state with streaming Solve/Logdet consumers;
- whole-graph representation/device shortest-path planning;
- Remath portability/observability integration receipts under `tools/remath/`;
- E172–E177 plus E186–E187 executable experiments, with E185 CPU-box provenance;
- regression tests for every exact rewrite;
- machine-readable result records.

## Repository map

- `src/wirkung_math/` — executable algebra/runtime primitives
- `experiments/` — numbered reproducible experiments
- `tests/` — correctness suite
- `docs/ALGEBRA.md` — indexed algebra with code pointers
- `docs/ARCHITECTURE.md` — runtime model
- `docs/RESULTS.md` — chronological measured results
- `docs/HARDWARE_ROUTER.md` — measured CPU↔GPU routing rules
- `docs/FINAL_SYNTHESIS_2026-11-05.md` — final breakthrough map and overlooked levers
- `tools/remath/` — leak-safe Remath portability integration
- `results/` — machine-readable imported/generated records
- `scripts/run_and_record.py` — reproducible experiment recorder

## Run

```bash
python -m pip install -e '.[dev]'
pytest -q
python scripts/run_all.py
python scripts/run_and_record.py --out results/my_run.json
```

## Central cost law

For sequence length `n`, edit count `e`, width `d`, rank `r`:

- dense full update: `n²d`
- direct sparse update: `ned`
- sparse × low-rank update: `nr(e+d)`

The factorized route beats direct sparse exactly when

`r < ed/(e+d)`.

The runtime treats this as a routing law, not as commentary.

## Current measured highlights

Imported CPU-box work:

- normalized update: up to 69.97× in the supplied E168 cases;
- dirty/local operator composition: 301.679× best supplied case;
- low-rank path: 2.453× best supplied E168 case.

Executable verification in this repository:

- E174 factor propagation across six linear layers: 54.89× in the recorded VM run, with a preceding run at 113.18×;
- E175 RMSNorm and LayerNorm rewrites agree with full recomputation to ~1e-15;
- E177 compacts nominal rank 128 to rank 8 without dense materialization;
- E185 preserves disconnected cones for 50.9×–196.6× vs full recomputation in the supplied CPU-box cases;
- E186 learns a 167.55-cell merge/split crossover and drives an optimal component partitioner;
- E187 preserves the dirty-path gain through global consumers: 649.36× cached sum update and 448.87× projected update on the recorded VM run.

See `docs/FINAL_SYNTHESIS_2026-11-05.md` for the final synthesis, `docs/INCREMENTAL_ALGEBRA.md` for the algebra, and `docs/RESULTS.md` / `results/` for full provenance.


## Hardware replacement status

The repository now contains direct resident-V100 comparisons, not only CPU-vs-CPU baselines. Current measured classes include:

- nonlinear local residual graph: CPU dirty execution beats V100/cuDNN above the measured size crossover;
- moments + Gram: CPU delta 73.6× over V100 full;
- low-rank solve + logdet: CPU delta 22.6× over V100 full;
- RBF full matrix: hybrid crossover — CPU only for extremely thin edits;
- RBF -> solve: structured CPU rank-delta path wins 16.57× / 11.51× / 4.54× for 1/4/16 edits at n=4096.

The router is consumer-aware: requiring a dense intermediate and consuming a structured delta are different execution problems.


## Final run closure

The E168–E209 round established direct resident-V100 crossover maps for local nonlinear graphs, pairwise kernels, spectral demand and normalized-query demand, plus persistent streaming state and GPU->CPU handoff economics.

The post-run David Tom Foss follow-up adds two major execution regimes:

- **cacheless local reconstruction:** exact old/new dirty outputs from bounded input crops, eliminating full intermediate activation caches for compatible consumers;
- **lazy query reconciliation:** unrequested normalized outputs incur zero maintenance work and reconcile exact edit history only when queried.

The runtime now plans over change structure, requested output, future consumer, state residency, expected horizon and hardware.
