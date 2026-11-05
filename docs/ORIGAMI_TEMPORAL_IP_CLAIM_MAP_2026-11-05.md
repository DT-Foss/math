# Origami Temporal Delta Machine — technical invention disclosure map

## Core invention

A stateful neural-program runtime that executes each inference step as an exact or certified transition from a prior program state, selecting among multiple delta representations and hardware routes while operating directly on compressed model pages.

## Independent technical pillars

### P1 — Temporal multi-representation linear execution

For a static linear form `y=W x`, retain previous input/output and route `x_t-x_(t-1)` through one of:

- coordinate support;
- codec block/bitplane support;
- learned temporal basis plus exact residual;
- shared cross-lane low rank;
- full evaluation.

Selection is based on measured execution cost and certificate validity.

### P2 — Delta-native compressed weight layout

Expose model weights by temporal input direction so that a changed input coordinate, block, basis direction or bitplane maps to contiguous compressed response pages.

The layout supports:

- sparse column response without dense decode;
- precomputed temporal-basis images;
- cross-lane factor images;
- full route from the same compressed representation.

### P3 — Exact normalization-to-multiple-projection transition

Maintain normalization statistics and previous projection outputs. For sparse temporal input changes, update multiple Q/K/V/router or gate/up projections from one norm transition without materializing the normalized vector or reading unchanged weight directions.

### P4 — Quantized nonlinear temporal support

Compute nonlinear gate/up updates, quantize the resulting activation, and use only changed activation codes to update a downstream projection. Full floating support is not treated as execution support.

### P5 — Certified MoE routing and expert residency

Maintain selected experts, exact selected logits and bounds for all unselected logits. Reuse the top-k set while a mathematically sufficient separation certificate holds; perform full refresh on certificate failure. Stable certification controls expert page residency and prefetch.

### P6 — Compiled partition-function attention

For temporal query regimes with discrete key signatures, aggregate softmax mass and value numerators by signature. Evaluate future normalized attention by a partition function over signatures rather than tokens.

Embodiments:

- flat occupied-signature table;
- support-specific partition bank;
- tensor-train / tensor-network partition state;
- incremental KV append;
- exact or controlled-rank compilation.

### P7 — Cross-lane/speculative delta factorization

Factor the matrix of temporal deltas across concurrent requests, speculative positions or M-Rail paths and share one weight image across all lanes.

### P8 — Semantic auto-compilation

Use a program observability/semantic system to recognize formulas and automatically produce persistent temporal states, delta rewrites, certificates, codec callbacks and hardware routes.

## Strong combined claim

A compiler/runtime combination that:

1. recognizes a static neural-program operator;
2. records the previous operator state;
3. measures or derives the temporal delta structure;
4. retains a portfolio of valid representations;
5. selects a representation using downstream demand, state residency and hardware cost;
6. executes directly against a compressed direction-addressable model representation;
7. updates downstream persistent states without dense intermediate materialization;
8. falls back or rebases when an exact certificate/cost boundary is crossed.

## Why the combination matters

Individual ingredients such as low-rank inference, sparse attention, KV caching, speculative decoding and quantization optimize one tensor or one operator. The combined machine changes the execution unit from a layer invocation to a certified state transition of the whole neural program.

## Concrete embodiments already implemented

- exact temporal Linear and RMSNorm→Linear;
- exact RMSNorm→multiple projections;
- direction-major weight layout;
- quantized-code Linear and SwiGLU→down updates;
- certified top-k MoE router;
- flat and tensor-train compiled partition attention;
- cross-lane low-rank Linear;
- learned temporal-delta basis with exact residual;
- compressed-codec callback ABI;
- Remath temporal rewrite annotations;
- Origami temporal census and route planner.

## Evidence package

Machine-readable prototype receipts:

- `results/david_tom_foss_temporal_delta_machine_2026-11-05.json`
- `results/david_tom_foss_delta_native_layout_2026-11-05.json`
- `results/david_tom_foss_quantized_swiglu_delta_2026-11-05.json`
- `results/david_tom_foss_origami_delta_budget_2026-11-05.json`

The actual Origami trace census is the next evidence layer and determines how often every exact route applies in the released model.
