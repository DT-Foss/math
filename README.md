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

See `docs/INCREMENTAL_ALGEBRA.md` for the unified model, and `docs/RESULTS.md` / `results/` for full provenance.
