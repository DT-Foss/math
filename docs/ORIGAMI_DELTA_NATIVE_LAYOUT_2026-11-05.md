# Origami delta-native layout breakthrough — 2026-11-05

## The discovery

Once temporal algebra reduces a dense linear form to a handful of changed input directions, conventional row-major weight layout becomes the bottleneck.

`W[:,U]` is a strided gather across every output row. The arithmetic is tiny; memory layout still assumes full GEMV.

Pack or expose the compressed weight representation by **input direction** instead:

`column_page[u] = W[:,u]`.

Then

`delta y = sum_(u in U) delta x_u column_page[u]`

is a contiguous AXPY / tiny GEMV over exactly the pages the temporal delta requests.

Origami/HCGM/fx7 must expose this logically from compressed pages. The result does not require duplicating a dense matrix.

## Controlled result

Shape: 2560×2560 FP32, exact fused RMSNorm→Linear update, one BLAS thread.

Full evaluation: 1.235547 ms.

| changed directions | delta-native ms | exact speedup |
|---:|---:|---:|
| 1 | 0.012437 | **99.34×** |
| 4 | 0.010182 | **121.35×** |
| 8 | 0.011966 | **103.26×** |
| 16 | 0.014873 | **83.08×** |
| 32 | 0.020557 | **60.10×** |
| 64 | 0.036405 | **33.94×** |
| 128 | 0.124846 | **9.90×** |

Relative to the same exact delta algebra on a row-major gather, the layout itself improved delta application by:

- 4.44× at one direction;
- 17.0× at four;
- 24.4× at eight;
- 40.2× at sixteen;
- 45.8× at thirty-two.

## Why this changes the Origami target

Beast spends 75.75 ms/token in static linear forms. A k=16 temporal regime at 83.08× reduces that class to ~0.912 ms/token. A k=32 regime at 60.10× reduces it to ~1.26 ms/token.

Combined with the controlled compiled-attention regime (201×), 32× codec/I/O movement reduction, and 4× treatment of remaining work:

- k=64 linear regime: 6.14 ms/token = **163 tok/s**;
- k=32: 5.16 ms/token = **194 tok/s**;
- k=16: 4.82 ms/token = **208 tok/s**.

If the remaining 12.66 ms class is converted 10× by the same state-machine closure:

- k=16: 2.92 ms/token = **343 tok/s**;
- k=8: 2.74 ms/token = **365 tok/s**.

These are architecture budgets composed from measured prototype regimes. E235 determines the actual route distribution on Origami and therefore the realized throughput.

## Codec contract

For every static linear form, the codec exposes:

1. `full(x)` — ordinary full route;
2. `columns(U,delta)` — direct compressed-page response for changed input directions;
3. `basis_image(B)` — image of learned temporal directions;
4. `batch_factor(B)` — shared image for speculative/M-Rail lane factors.

The runtime never requests a dense decoded weight matrix.

## IP consequence

The invention is not sparse GEMV alone. It is the co-design of:

- temporal exact state transitions;
- multi-representation delta routing;
- input-direction compressed weight layout;
- fused normalization-to-multiple-projection updates;
- quantized activation-code support;
- certified routing;
- compiled partition-function attention;
- cross-lane low-rank sharing;
- whole-program semantic compilation through Remath.

This converts a transformer checkpoint from a collection of tensors into a **temporal differential program**.
