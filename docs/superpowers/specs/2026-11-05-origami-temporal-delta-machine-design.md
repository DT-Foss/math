# Origami Temporal Delta Machine design

## Intent

Convert Origami/Beast decode from repeated dense layer execution into persistent exact/certified token-to-token state transitions. The runtime must attack Beast's measured dominant classes together: static linear forms, normalized reductions and codec/I/O movement.

## Goals

- exact temporal updates whenever a certificate/representation applies;
- direct operation on Origami/HCGM/fx7 compressed pages;
- route among coordinate, block-code, learned-basis, cross-lane low-rank and full forms;
- compile attention partition state from real key-code signatures;
- certify MoE top-k reuse;
- measure actual route occupancy from Origami traces;
- retain full execution as an explicit crossover/rebase route.

## Non-goals

- changing model weights or training a replacement architecture;
- depending on approximate hidden-state reuse for correctness;
- decoding a dense weight matrix before every delta operation;
- claiming an end-to-end Origami throughput before the actual trace census/runtime integration.

## Approaches considered

### Dense-kernel optimization

Continue optimizing every GEMV/softmax individually. This cannot remove the 134.13 ms/token Beast operation budget because all dense weight/state traffic remains.

### Approximate subspace/gating

Predict whether low-rank or sparse paths are accurate enough. Useful as an optional approximate mode, but it does not exploit Origami's existing exact quantized/code structure or provide an exact general fallback discipline.

### Chosen: exact temporal differential machine

Store prior operator state and propagate a portfolio of exact representations. Compile model layout and attention state around temporal directions. Use certificates and measured cost crossovers to select exact delta or full routes.

## Architecture

1. Remath recognizes formula semantics.
2. Origami temporal census records coordinate/block/code/basis/rank geometry.
3. The compiler builds a portfolio of temporal states.
4. Direction-major compressed pages serve sparse/basis weight responses.
5. RMSNorm updates all dependent projections from one state transition.
6. Quantized nonlinear code changes update downstream projections.
7. Certified router preserves expert sets/residency when valid.
8. Partition-function attention evaluates grouped/tensor-network signature state.
9. Cross-lane factorization shares weight images across M-Rail/speculative lanes.
10. Whole-graph planning routes CPU delta, GPU delta or full execution.

## Verification

- exact algebra tests for every state transition;
- fallback/certificate-failure tests;
- reproducible synthetic structure benchmarks;
- GitHub Actions full suite;
- E235 actual Origami trace census before claiming end-to-end throughput.
