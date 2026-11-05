# Results registry

## Imported E168–E171 — CPU box, 2026-11-05

These measurements were supplied from the running CPU-box work and are retained as provenance.

### Normalized bilinear update

- n=2048, edits=1: 69.97×
- n=2048, edits=8: 15.01×
- n=2048, edits=64: 0.76×
- n=4096, edits=64: 2.43×
- n=8192, edits=64: 5.34×
- all 9 initial cases numerically matched; max absolute error about 1.5e-8.

Initial crossover map:

- n=1024: 64 edits = 6.25%
- n=2048: 64 edits = 3.125%
- n=4096: 256 edits = 6.25%
- n=8192: 512 edits = 6.25%

### Lazy Top-K update

After correction of the NumPy-view benchmark bug:

- n=2048: 256 edits = 12.5% remained profitable;
- n=4096: 512 edits = 12.5% remained profitable;
- n=8192: 512 edits remained profitable;
- observed Top-K depth was only 1–6.

Upper-bound search:

- n=4096, 12.5% edits: 1.67×
- n=4096, 25% edits: 0.37×
- n=8192, 6.25% edits: 1.08×
- n=8192, 12.5% edits: 0.57×
- n=8192, 25% edits: 0.37×

### Dirty/local operator composition

- 9/9 tested cases correct;
- best reported speedup: 301.679×.

### Low-rank update

- 12/12 tested cases correct;
- best reported speedup: 2.453×;
- errors about 1e-7.

## Executable verification on ChatGPT VM — 2026-11-05

Machine-readable record: `results/chatgpt_vm_2026-11-05.json`.

Environment: AMD EPYC 9V74, NumPy 2.3.5, Python 3.13.5, single-thread BLAS/OMP.

### E172 — mass-gated normalized state

At n=1024,d=64:

- 6.25% edits: 2.52×
- 12.5% edits: 1.19×
- 25% edits: 0.58×
- 50% edits: 0.32×
- zero fallback rows in this random workload;
- max absolute FP32 error <= 2.69e-7.

### E173 — sparse × low-rank

At n=4096,e=512,d=64, analytic operation-count crossover `r*=56.89`.

Recorded speedups:

- r=1: 7.02×
- r=2: 4.67×
- r=4: 5.07×
- r=8: 3.74×
- r=16: 4.16×
- r=32: 1.39×
- r=64: 1.08×

An earlier same-VM run put r=64 below parity, locating the executor transition around the predicted rank-57 region.

### E174 — factor propagation without intermediate materialization

Rank 4, n=4096,d=64, six linear layers, final materialization included:

- recorded run: 54.89×;
- preceding same-VM run: 113.18×;
- max absolute FP32 difference: 8.88e-7.

### E175 — normalization rewrites

- RMSNorm max absolute error: 1.78e-15
- LayerNorm max absolute error: 3.56e-15

### E176 — nonlinear rank barrier

A rank-4 delta became numerical rank 64 after SiLU at every tested amplitude.

- rank-4 energy retained: about 69–73%
- rank-8 energy retained: about 75–86%

This isolates elementwise nonlinearity as a measured structure-expansion boundary.

### E177 — factor compaction

32 rank-4 residual terms produced nominal rank 128 but shared an 8-dimensional latent space.

QR + small-core SVD recovered rank 8 without dense materialization:

- 16× factor-rank compression;
- 10.35× faster subsequent materialization in the recorded run;
- 12.46× in the preceding run;
- max absolute error: 2.56e-12.
