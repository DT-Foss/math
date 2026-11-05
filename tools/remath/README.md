# Remath integration

This directory records the portable integration of the user-supplied **wirkung-remath 0.1.0** toolchain used for computational observability, formula recognition, pattern mining and native kernel synthesis.

## Provenance

- imported source snapshot HEAD: `7fed349`
- the raw uploaded archive was **not** copied into this public repository
- excluded deliberately: nested `.git`, caches, generated reports, SQLite traces, compiled `.dylib` files, build directories and other non-source artifacts
- source allowlist secret scan: no credential/token material found

## Sol portability work

The supplied snapshot was macOS/Accelerate-oriented. Sol repaired the runtime for Linux/x86 without weakening the existing macOS path:

- platform shared-library suffixes (`.dylib` / `.so` / `.dll`)
- compiler discovery through `CXX`, `clang++`, `g++`, `c++`
- Apple Accelerate on macOS; CBLAS/`-lblas` on Linux
- native CPUDA runtime is rebuilt from source instead of loading a foreign-platform binary
- Morph/Holo/Stage-5/Stage-6 synthesis uses the common portable build helper
- Stage-6 keeps the ARM NEON path and adds an exact scalar x86 fallback
- release receipts build/install locally without requiring network dependency resolution
- npm local-wheel verification can run with `--no-deps`
- `remath doctor` recognizes Linux BLAS and reports native synthesis ready correctly

Full patch: `PORTABILITY_PATCH.diff`.

## Verification

On the Sol Linux VM:

- **22/22 tests passed**
- release-receipt test passed offline
- installed-wheel `doctor`: `native_synthesis_ready: true`

Clean release artifacts generated from the repaired tree:

| Artifact | SHA-256 |
|---|---|
| `wirkung_remath-0.1.0-py3-none-any.whl` | `2deebfcc0e62cacf1afba20707d6ad667a07699977a7548340480e54ea54d61f` |
| `wirkung_remath-0.1.0.tar.gz` | `5df3426cb95c56cd600ea460182c0631364e3d8d44a1305263cab7b34c92a7ec` |
| npm `wirkung-remath-0.1.0.tgz` | `5bcdc428124147774182b5e359b918631ca589b4e9d935e6a6ffeda025283d1d` |

The binary release artifacts are intentionally not committed here. The public repository stores the reproducible source transformation and its verification receipt instead.

## Why Remath belongs next to Wirkung Math

Remath provides the observability layer needed by the delta compiler:

`program -> execution graph -> recognized mathematical operator -> delta representation -> rewrite/router decision`

The next integration boundary is therefore operator recognition and cost annotation, not copying raw build artifacts.


## Live Wirkung bridge

`src/wirkung_math/remath_bridge.py` consumes Remath `FormulaSpec` / `GraphNode` objects (or their dictionary form) without requiring Remath as a runtime dependency.

Current semantic routes include:

- Scaled Dot-Product Attention -> dynamic normalized bilinear state;
- Stable Softmax -> mass-gated normalized sufficient state;
- RMSNorm/LayerNorm -> structured normalization rewrite;
- GEMM/GEMV / matrix contraction -> lazy factor propagation;
- SiLU/SwiGLU and other activations -> preserve support geometry while allowing rank structure to change;
- additive/order/spectral reductions -> their corresponding incremental state families.

This is the intended integration point: Remath discovers mathematical meaning; Wirkung chooses the state-transition algebra and hardware route.
