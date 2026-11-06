# Consumer-Relative Incremental Computation Compiler — Design

## Intent

Build a compiler that discovers when a program's full state contains information no future consumer can observe, derives a smaller persistent state, attaches exact certificates or empirical validation evidence, benchmarks competing representations, and emits the cheapest semantically valid execution plan.

The compiler is a discovery engine, not a catalogue of hand-written speedups. Existing Wirkung rewrites remain candidate generators; the new layer decides what information is actually required by the future consumer and searches for a smaller state/transition program.

## Exactness contract

Every candidate is labelled PROVEN, VALIDATED, or HEURISTIC.
- PROVEN: algebra/certificate proves consumer semantics for the declared domain.
- VALIDATED: Golden Reference traces agree; usable only when validation mode is explicitly enabled.
- HEURISTIC: discovery lead only; never selected for execution.

Default compilation is exact_only=True and may select only PROVEN candidates. No empirical correlation is silently promoted to an exact rewrite.

## Core IR

A ConsumerProgram is a directed operator graph with named nodes, semantic family, optional operator payload/shape metadata, changed source nodes, and requested consumer nodes.

The compiler first applies Desc(changed) intersect Anc(requested) to remove irrelevant graph regions.

## Discovery engines

### Exact linear consumer quotient

For z=Cx, compute an orthonormal basis Q for row(C). The minimal consumer state is q=Q^T x and z=(C Q)q. Any component in ker(C) is exactly irrelevant. For x=A u, emit q=(Q^T A)u without materializing x.

### Exact observable-state quotient

For x_(t+1)=A x_t+B u_t, y_t=C x_t, build O=[C;CA;CA^2;...]. A basis Q for row(O) yields z=Q^T x with Aq=Q^T A Q, Bq=Q^T B, Cq=C Q. The compiler verifies the intertwining residual before emitting a PROVEN quotient.

### Empirical consumer-equivalence discovery

Given state traces and consumer signatures, measure class compression, transition entropy and within-class transition reuse. This emits VALIDATED/HEURISTIC leads only.

### Certificate synthesis

Built-in exact certificates cover linear argmax/top-k margin bounds, quantization-cell stability, threshold/sign stability and zero additive contribution. Certificates may form refinement ladders.

## Planner

Each route carries state bytes, setup/update/query cost, representation, device and evidence. Candidates become edges in RepresentationPlanGraph. The planner minimizes expected cost over a horizon while respecting exactness policy.

## First breakthrough demonstrations

1. Automatic observable quotient: discover a tiny exact observable state inside a high-dimensional linear state machine and benchmark it against full state updates.
2. Consumer-sensitive path: eliminate a dense intermediate automatically when its downstream linear consumer has low row rank.
3. Real-model receipt ingestion: classify TinyStories/SmolLM routes as accepted/rejected from measured semantics and cost.

## Success criteria

- exact quotient tests pass;
- empirical candidates cannot leak into exact-only plans;
- benchmark shows large measured full-vs-quotient speedup;
- real decoder receipts are ingested automatically;
- complete test suite stays green.