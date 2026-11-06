# Real-model temporal validation — 2026-11-06

## Models

Two public causal decoders were executed directly on Hugging Face CPU jobs:

1. `HuggingFaceTB/SmolLM2-135M-Instruct`
   - 134,515,008 parameters
   - Llama architecture
   - hidden 576
   - MLP 1536
   - 30 layers
   - 9 query heads / 3 KV heads
2. `roneneldan/TinyStories-1M`
   - 3,745,984 parameters
   - GPT-Neo architecture
   - hidden 64
   - 8 layers

The purpose was to replace synthetic structure assumptions with measured token-to-token state geometry.

## SmolLM2 baseline

Prompt: `The future of efficient AI is`.

Transformers `generate`, 16 greedy tokens:
- 1.60285 s
- 9.98 tok/s including generation-framework overhead.

Manual KV-cached decode on the same CPU-performance runner:
- approximately 20–24 tok/s depending on run/warm state.

## SmolLM2: naive activation-code temporal routing

Every Linear input was assigned fixed per-coordinate activation codes, and a temporal state retained the previous code vector and Linear output.

The temporal replay was exact relative to its fixed-scale quantized execution when the sparse route was not recursively accumulated. The observed support was not sparse:

| bits | q/k/v median changed | down median changed | temporal/full-quant |
|---:|---:|---:|---:|
| 3 | 420 / 576 | 883 / 1536 | 0.680× |
| 4 | 500 / 576 | 1172 / 1536 | 0.740× |
| 5 | 541 / 576 | 1368 / 1536 | 0.758× |

Conclusion: raw coordinate/code sparsity is not the transformer invariant.

## SmolLM2: 64-token temporal census

Across all 30 layers:

### Q input
- rank-90 median: 26
- rank-99 median: 47
- rank-99.9 median: 57
- held-out rank-32 PCA residual: ~0.72
- 2-bit coordinate change median: 30.1%
- 2-bit absolute block-4 repeat hit: 75.3%

### Gate input
- rank-90: 28
- rank-99: 48
- held-out rank-32 residual: ~0.78
- 2-bit change: 29.9%
- 2-bit absolute block-4 hit: 75.1%

### MLP down input
- rank-90: 29
- rank-99: 49
- held-out rank-32 residual: ~0.87
- 2-bit change: 17.7%
- 2-bit absolute block-4 hit: **84.3%**

### LM head
- rank-90: 25
- rank-99: 47
- held-out rank-32 residual: ~0.70
- 2-bit change: 30.4%
- 2-bit absolute block-4 hit: 75.9%

A single fixed low-rank temporal basis therefore does not explain the trajectory. Repeated discrete block states are substantially stronger.

## SmolLM2: deterministic integer delta state

A real layer-15 MLP `down_proj` (1536 -> 576) was compiled into response-quantized direction-major integer state.

For all 64 real token inputs, the temporal int accumulator was exact to the corresponding full compiled integer arithmetic.

| activation bits | changed coordinates | Full compiled seq | Delta seq | delta/full |
|---:|---:|---:|---:|---:|
| 3 | 54.6% | 5.064 ms | 3.958 ms | **1.279×** |
| 4 | 76.2% | 3.982 ms | 4.412 ms | 0.902× |
| 5 | 88.1% | 5.150 ms | 4.854 ms | 1.061× |

This removes recursive FP32 drift as an execution-semantic problem. The remaining question is representation quality / event density.

## SmolLM2: FP32 quality boundary

Quantizing only MLP `down_proj` activation inputs:

- 3-bit: FP32 greedy prefix 0/32
- 4-bit: 1/32
- 5-bit: 1/32
- 6-bit: 18/32
- **7-bit: 32/32**
- **8-bit: 32/32**

For this prompt, selective down-projection coding therefore needs roughly 7 bits to preserve the complete 32-token FP32 greedy sequence.

This is a prompt-level compiler calibration result, not a global model-quality claim.

## SmolLM2: routes ruled out by real traces

### Static response quotient VQ

Block response codebooks trained on a separate 512-token calibration text and tested on held-out generation remained too inaccurate:
- Q/K median relative response error remained ~17–21% even at K=128;
- V ~43%;
- Down ~48%.

### Simple fixed temporal PCA

Held-out residuals remain 70–90% at rank 32, so fixed low-rank basis alone is not the route.

### Simple Cauchy-certified vocabulary argmax

A 64-token exact test retained all 49,152 candidates at every step. The bound is correct but too loose to remove LM-head work.

### Layer-0 token Q/K/V compilation

Layer-0 `RMSNorm(embedding[token]) -> Q/K/V` is context-independent and was compiled exactly for all vocabulary tokens.

- 128/128 tokens identical to full decode.
- table size: 188.7 MB.
- throughput: 23.17 tok/s versus 23.38 tok/s full.

The lookup state is larger than the three small layer-0 weight matrices, so it does not pay on this model.

## TinyStories-1M baseline

Prompt: `Once upon a time`.

128 greedy tokens:
- median decode latency: 3.968 ms/token
- **252.0 tok/s** on the first CPU-performance census run.

## TinyStories: temporal event geometry

The small GPT-Neo model exhibits strong low-bit transition reuse.

Examples:

- layer-0 Q input, 2-bit block-2 transition hit: **90.4%**
- layer-1 Q input: **91.9%**
- later Q inputs: typically **92–93%**
- layer-1 attention-out, 2-bit block-4 hit: **91.1%**
- layer-3 attention-out: **88.2%**
- layer-0 MLP c-proj, 2-bit block-4 hit: **85.5%**
- layer-7 MLP c-proj: **84.5%**

This is direct evidence for a Temporal Event ISA on a real decoder.

## TinyStories: complete event executor

All internal Linear operators except `lm_head` were compiled into block-state response tables with deterministic integer persistent accumulators.

128-token full-event versus temporal-event execution:

### 2-bit
- Full event: 168.05 tok/s
- Temporal event: **175.23 tok/s**
- End-to-end temporal gain: **1.043×**
- temporal tokens exactly equal to full-event tokens: yes
- median changed block fraction: 25%
- response tables: 14.16 MB

### 3-bit
- Full event: 164.66 tok/s
- Temporal event: 159.42 tok/s
- 0.968×
- exact to full-event tokens
- changed block fraction: 90.6%

The 2-bit executor is the first measured complete real-decoder temporal speedup. Its model semantics are not FP32-equivalent; global 2-bit coding collapses the generated sequence. The next compiler stage is therefore mixed per-operator precision/event selection.

## Native fused Event ISA kernel

Removing the Python/ctypes call per token exposes the actual transition kernel.

On a real TinyStories layer-3 attention `out_proj`:

- 2-bit block-2 full native state sequence: 0.0677 ms / 128 transitions
- temporal event sequence: **0.0578 ms**
- event/full native: **1.172×**
- relative to sequential Python/BLAS GEMVs: **3.03×**

The MLP `c_proj` does not benefit at the same code density, confirming that event execution is operator-specific and must be cost-routed.

## Runtime conclusion

Real-model evidence changes the target representation:

`dense hidden delta -> fixed sparse coordinates` is rejected.

The surviving execution hierarchy is:

1. deterministic integer response state;
2. repeated block-transition events where observed;
3. per-operator precision selection;
4. full route at dense/high-precision boundaries;
5. richer union-of-subspaces/dictionary representation only when held-out traces justify it.

The compiler, not a universal sparse kernel, chooses the representation.


## TinyStories: per-operator precision is a runtime state

A 64-token sensitivity map was measured by changing exactly one operator at a time and requiring the full FP32 greedy sequence to remain identical.

Examples of the minimum individually safe activation precision on this trace:

- layer 0: Q 3-bit, K 4-bit, V 4-bit, attention-out 5-bit, MLP c-proj 8-bit;
- layer 1: K 6-bit, V 5-bit, attention-out 8-bit, MLP c-proj 6-bit;
- layer 2: Q 4-bit, attention-out 6-bit, MLP c-proj 7-bit;
- layer 4: Q 4-bit, V 5-bit, attention-out 6-bit;
- layer 5: Q 4-bit, K/V 5-bit, attention-out 6-bit, MLP c-proj 7-bit;
- layer 7: V 4-bit, Q 5-bit, attention-out 6-bit.

Several K projections and MLP c-projections remain unsafe even at the tested upper precision.

Therefore precision is not a model-wide scalar. It is part of the per-operator execution state.

Machine-readable record:
`results/david_tom_foss_tinystories_precision_sensitivity_2026-11-06.json`.

## Absolute signed-digit planes: measured boundary

High precision was decomposed into 4–6 balanced ternary planes, each preserving a small block-state alphabet.

This removes the exponential table growth of a single high-bit code, but absolute planes still changed too often:

| planes | effective bits | temporal/full | FP32 prefix | dirty plane-blocks |
|---:|---:|---:|---:|---:|
| 4 | 6.34 | 0.965× | 14/128 | 82.8% |
| 5 | 7.92 | 0.950× | 16/128 | 84.1% |
| 6 | 9.51 | 0.953× | 30/128 | 84.9% |

The route is therefore rejected in absolute-state form.

The surviving transform is **error-feedback temporal coding**: encode level crossings relative to the reconstructed previous state instead of repeatedly encoding the absolute activation.

Implementation:
`SigmaDeltaResponseLinearState`.

## Quality-preserving mixed Sigma-Delta compiler

A real compiler pass used the measured per-operator sensitivity map as candidates.

Procedure:

1. fit activation step sizes on two compile prompts;
2. greedily add a Sigma-Delta route only while the *combined* route set preserves every compile token;
3. prune the combined plan against a separate validation prompt;
4. evaluate on a third 128-token prompt.

The compile pass initially accepted 26 operators. Validation reduced this to six early Q/K/V operators:

- layer-0 Q: 4-bit;
- layer-0 K: 4-bit;
- layer-0 V: 3-bit;
- layer-0 attention-out: 6-bit;
- layer-1 K: 5-bit;
- layer-1 V: 6-bit.

### Separate validation prompt

`A little dog was playing outside`, 64 tokens:

- FP32: 288.96 tok/s;
- Full Sigma-Delta semantics: 283.68 tok/s;
- Temporal Sigma-Delta: 272.31 tok/s;
- FP32 greedy sequence: **64/64 identical**;
- temporal/full-Sigma sequence: **64/64 identical**;
- median event fraction: 90.6%.

### Held-out test prompt

`One day the sun was shining`, 128 tokens:

- FP32: 299.94 tok/s;
- Full Sigma-Delta semantics: 277.88 tok/s;
- Temporal Sigma-Delta: 276.25 tok/s;
- FP32 greedy sequence: **128/128 identical**;
- temporal/full-Sigma sequence: **128/128 identical**;
- median event fraction: 90.6%.

This closes the quality question for the compiled mixed plan: temporal error-feedback can preserve the complete FP32 greedy path on a separate test prompt. It does not yet accelerate this model because the selected operators remain event-dense.

Machine-readable record:
`results/david_tom_foss_tinystories_mixed_sigma_delta_2026-11-06.json`.

## Exact vocabulary demand: certified subspace head

The 50,257×64 LM head was tested as an exact maximum-inner-product search problem.

A basis learned from 256 real decoder hidden states provides, for every vocabulary row, an exact Cauchy upper bound from its projected component plus residual norm.

All tested greedy tokens remained exact.

At rank 48, the one-shot certificate reduced the median exact candidate count from 50,257 to:

- 32 on `Once upon a time`;
- 32 on `A small dog went to the park`;
- 32 on `One day the sun was shining`.

Progressive stages tighten the certificate further. With stages 4→12→24→40→56, final median candidate counts are **2** on all three prompts.

For example:

- `Once upon a time`: 50,257 → 50,257 → 26,804 → 85 → **2**;
- `A small dog went to the park`: 50,257 → 50,257 → 50,256 → 174 → **2**;
- `One day the sun was shining`: 50,257 → 50,242 → 15,043 → 163 → **2**.

This is a real exact demand collapse of more than four orders of magnitude at the final scoring stage.

### Performance boundary

The Python/Torch implementation and a first scalar native implementation remain slower than the full 64D SGEMV.

Native staged execution reaches roughly 0.44–0.54× of full-head throughput on this model despite the final two-candidate set.

The reason is specific and measured: hidden width 64 makes the dense head extremely cheap, while early bound stages still touch much of the vocabulary.

This route remains structurally important for wide vocabulary heads / wider hidden states, but TinyStories is below its hardware crossover.

Machine-readable records:

- `results/david_tom_foss_tinystories_certified_subspace_head_2026-11-06.json`;
- `results/david_tom_foss_tinystories_progressive_head_2026-11-06.json`;
- `results/david_tom_foss_tinystories_native_certified_head_2026-11-06.json`.

## Real-model conclusion

The real decoder campaign establishes five separate facts.

### 1. Temporal execution itself works end-to-end

TinyStories demonstrates a complete persistent event executor with a measured 1.043× temporal/full gain inside identical 2-bit compiled semantics.

### 2. Deterministic accumulators solve temporal arithmetic identity

SmolLM2 and TinyStories response-state experiments show that full and temporal compiled arithmetic can be made identical through integer response accumulators rather than recursively accumulating FP32 outputs.

### 3. FP32-quality preservation is compiler-selective

The mixed Sigma-Delta plan preserves a separate 128-token FP32 greedy sequence exactly. Quality preservation therefore does not require every operator to use the same temporal representation.

### 4. Standard dense checkpoints are not naturally event-native

On SmolLM2 and TinyStories, high-precision activation states change densely. A runtime that simply waits for sparse FP32/7-bit coordinate changes is not the moonshot.

The model representation must expose temporally stable response/event coordinates deliberately.

### 5. The Origami target becomes sharper

The required model/runtime co-design is now:

`checkpoint -> temporal census -> per-operator precision/event representation -> deterministic response state -> cost crossover -> full/delta route`.

Origami/HCGM is a better target than an ordinary FP32 checkpoint because its compressed code representation can be laid out directly around response events rather than retrofitting event semantics after training.

The next Origami measurement is no longer an open-ended search. It must report, per operator:

- event rate at the model's native code precision;
- response-page bytes touched per token;
- integer-state reuse;
- full/delta crossover;
- cross-lane event sharing;
- resulting end-to-end tok/s.

