# Consumer-Relative Incremental Computation Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an exact-by-default compiler that discovers consumer quotients, persistent sufficient state, certificates and globally cheapest incremental execution plans.

**Architecture:** Add a consumer-program IR, exact linear/observable quotient engines, empirical equivalence discovery, certificate primitives, and a compiler that lowers candidates into the existing representation planner. Add a benchmark and real-result ingester so measured hardware evidence can accept/reject candidate routes.

**Tech Stack:** Python 3.11+, NumPy, existing wirkung_math planners, pytest.

**Spec:** docs/superpowers/specs/2026-11-06-consumer-relative-compiler-design.md

## Global Constraints
- exact-only compilation selects PROVEN candidates only;
- empirical trace agreement alone never becomes PROVEN;
- reuse Remath semantics and RepresentationPlanGraph;
- no new runtime dependency beyond NumPy;
- plans carry evidence and cost provenance.

## Tasks

### 1. Consumer IR and evidence policy
Create src/wirkung_math/consumer_ir.py and tests/test_consumer_ir.py.
Test graph slicing, evidence ordering and exact-only filtering before implementation.

### 2. Exact linear quotient discovery
Create src/wirkung_math/linear_quotient.py and tests/test_linear_quotient.py.
Implement LinearConsumerQuotient.fit(C) and ObservableLinearQuotient.fit(A,B,C). Test rank-deficient quotient and exact trajectory equivalence.

### 3. Empirical quotient and certificate discovery
Create src/wirkung_math/quotient_discovery.py, src/wirkung_math/certificates.py and tests.
Measure consumer classes/transition entropy; add exact argmax, quantization-cell and threshold certificates.

### 4. Whole-graph quotient compiler
Create src/wirkung_math/quotient_compiler.py and tests/test_quotient_compiler.py.
Lower proven candidates into RepresentationPlanGraph and enforce evidence policy.

### 5. Real receipt ingester
Create src/wirkung_math/result_discovery.py and tests/test_result_discovery.py.
Parse result fields generically into semantic-success/cost-win candidate routes.

### 6. Benchmark + CLI
Create experiments/quotient_compiler_bench.py and scripts/discover_compute_quotients.py.
Benchmark automatic observable-state reduction and ingest existing real-model receipts.

### 7. Verification/integration
Update docs and README, run full pytest, run benchmark/CLI, request review, PR, final-head CI, merge, verify main.