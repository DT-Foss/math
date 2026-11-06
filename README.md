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
- an **Origami Temporal Delta Machine** for token-to-token state transitions;
- exact temporal Linear, RMSNorm→multi-projection and quantized activation-code updates;
- a delta-native input-direction weight layout and compressed-codec callback ABI;
- certified MoE top-k routing;
- flat and tensor-train compiled partition-function attention;
- cross-lane/speculative low-rank factorization;
- a shipped Origami/Beast temporal-delta census CLI;
- Remath semantic auto-compilation into temporal rewrite families;
- Remath portability/observability integration receipts under `tools/remath/`;
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
- `docs/ORIGAMI_TEMPORAL_DELTA_MACHINE_2026-11-05.md` — transformer state-machine architecture
- `docs/ORIGAMI_DELTA_NATIVE_LAYOUT_2026-11-05.md` — direction-major codec layout and 160–365 tok/s architecture budgets
- `docs/ORIGAMI_TEMPORAL_IP_CLAIM_MAP_2026-11-05.md` — technical invention-disclosure map
- `docs/REAL_MODEL_TEMPORAL_VALIDATION_2026-11-06.md` — real SmolLM2/TinyStories decode validation and measured boundaries
- `tools/remath/` — leak-safe Remath portability integration
- `results/` — machine-readable imported/generated records
- `scripts/origami_temporal_census.py` — actual Origami/Beast temporal geometry census
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

The temporal transformer extension adds two more independent axes:

- changed activation/code directions;
- outputs/queries actually requested.

Its core execution law is:

`work ∝ temporal information change × requested output × surviving structure`.

## Current measured highlights

Direct resident-V100 records include:

- nonlinear local residual graph: **235.17×** at 8192²;
- moments + Gram: **1152.60×** for 64 changed rows at 33,554,432×64 FP64;
- supplied one-row full run: **1627.18×**;
- composed Gram + 16-RHS Solve + Logdet: **225.93×**;
- RBF→Solve: CPU structured route still wins at 16 changed points.

Origami Temporal Delta Machine controlled prototypes include:

- direction-major exact RMSNorm→Linear: **60–121×** for 4–32 changed directions at d=2560;
- exact certified Beast-sized MoE router: **18.67×**;
- exact compiled partition attention: **201.10×** on a 131k-context discrete-signature workload;
- quantized SwiGLU→down: **11.73×** when 0.25% of activation codes change;
- 80-lane shared temporal low rank: **8.60×** at rank 1.

The measured Beast ctx512 operation budget is 134.13 ms/token (~7.46 tok/s). Composition of controlled temporal routes yields architecture targets of **163–365 tok/s**. `origami_temporal_census.py` replaces route assumptions with actual Origami layer distributions.

## Hardware replacement status

The repository contains direct resident-V100 comparisons, not only CPU-vs-CPU baselines. The router is consumer-aware: requiring a dense intermediate and consuming a structured delta are different execution problems.

The next hardware phase diagram is:

`CPU_delta ↔ GPU_delta ↔ GPU_full`.


## Real decoder validation

The temporal runtime has now been exercised on two public causal decoders rather than synthetic operators only:

- **SmolLM2-135M-Instruct** — 134.5M parameters, 30 Llama layers;
- **TinyStories-1M** — 3.75M parameters, 8 GPT-Neo layers.

Measured conclusions:

- raw FP32 / high-bit coordinate deltas are dense on ordinary checkpoints;
- simple fixed temporal PCA, static short-block response VQ, absolute signed-digit planes, and loose vocabulary Cauchy bounds do not provide a profitable universal route;
- deterministic response-quantized integer accumulators make temporal and full compiled arithmetic identical;
- a complete TinyStories 2-bit event executor gives **1.043× end-to-end temporal/full speedup** inside identical compiled semantics;
- a mixed per-operator Sigma-Delta compiler preserves a separate **128/128-token FP32 greedy sequence exactly**, but ~90.6% event density leaves it below the full FP32 crossover;
- real low-bit block transitions can repeat above 90% on selected TinyStories operators;
- exact vocabulary demand can collapse 50,257 rows to a median **2 final candidates**, although the tiny 64-wide head remains below the hardware crossover for staged certification.

The resulting compiler target is therefore per-operator **precision × event representation × hardware route**, with deterministic full execution as the cost/quality fallback.
