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
