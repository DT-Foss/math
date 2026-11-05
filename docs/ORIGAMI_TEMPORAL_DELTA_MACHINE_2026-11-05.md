# Origami Temporal Delta Machine — 2026-11-05

## Thesis

Origami must not execute a dense transformer forward pass for every token.

Each decode step is a state transition:

`previous model state + compact token delta + requested outputs -> next model state`.

The runtime propagates a portfolio of exact delta representations and reads only the weight/state fragments required by the cheapest valid representation.

## Measured Beast target

Current ctx512 census:

| class | ms/token | share |
|---|---:|---:|
| static linear forms | 75.75 | 56.5% |
| normalized reductions | 26.24 | 19.6% |
| I/O / codec movement | 19.48 | 14.5% |
| other | 12.66 | 9.4% |
| total | 134.13 | 100% |

Current throughput: ~7.46 tok/s.

100 tok/s requires <=10 ms/token, an end-to-end 13.413× reduction. No isolated kernel can deliver this. The three dominant classes must be converted together.

## Runtime state

For every static linear form, retain:

- previous input code/state;
- previous output;
- coordinate/block/code support of the temporal delta;
- temporal-basis coefficients and exact residual;
- optional cross-lane/speculative low-rank factors;
- zero-copy callbacks into the Origami/HCGM/fx7 codec.

For attention, retain:

- base query regime;
- quantized key signatures in active temporal directions;
- grouped partition mass and value numerators;
- optional tensor-train partition state;
- KV append overlay.

For MoE routing, retain:

- selected experts;
- exact selected logits;
- interval bounds for all other logits;
- certificate margin;
- resident expert/code pages.

## Exact linear transition

For `y_t = W x_t`:

`y_t = y_(t-1) + W delta_x_t`.

Routes:

1. coordinate support: `W[:,U] delta_x[U]`;
2. changed codec blocks;
3. temporal basis: `(W B) delta_c + W residual`;
4. cross-lane rank r: `delta_Y = A (B^T W^T)`;
5. full evaluation.

The route is chosen from measured delta geometry, not assumed sparsity.

## Exact RMSNorm -> multi-linear transition

If `x' = x + delta_U` and `rho,rho'` are RMS scales:

`norm(x') = (rho/rho') norm(x) + gamma_U delta_U/rho'`.

Therefore every projection sharing that normalized input updates as

`W_j norm(x') = (rho/rho') y_j + W_j[:,U](gamma_U delta_U/rho')`.

Q/K/V/router or gate/up projections share one norm update and read only changed weight columns.

David Tom Foss VM, d=2560, one projection, one BLAS thread:

| changed coordinates | exact speedup |
|---:|---:|
| 1 | 34.49× |
| 4 | 14.26× |
| 8 | 7.52× |
| 16 | 3.45× |
| 32 | 1.79× |

The actual Origami trace census selects the operating region.

## Quantized activation transition

Dense floating drift is not the execution support. The runtime support is the set of activation codes that actually change.

For a quantized SwiGLU state:

1. update gate/up values;
2. quantize the new SwiGLU output;
3. find changed codes;
4. update the down projection from those columns only.

Synthetic exact-code benchmark, hidden=14336, output=2560, 33 levels:

| gate/up delta std | code-change fraction | down-projection speedup |
|---:|---:|---:|
| 0.001 | 0.25% | 11.73× |
| 0.003 | 0.74% | 4.95× |
| 0.010 | 2.27% | 2.59× |

The Origami code trace determines the real distribution; the full route remains the measured crossover fallback.

## Certified MoE routing

For router logits `z=W_r x`, a sparse delta gives exact selected-logit updates and per-expert Cauchy bounds.

The old top-k remains exact when

`min lower(selected) > max upper(unselected)`.

Then:

- no full router matvec;
- no global top-k search;
- expert residency remains stable;
- only selected gate weights are recomputed.

Synthetic Beast-sized router (384 experts, d=2560, top-6, eight changed coordinates):

- full router: 0.488 ms;
- certificate path: 0.0261 ms;
- 18.67×.

Certificate failure performs one exact full refresh.

## Cross-lane / speculative low rank

For 80 parallel instances or speculative positions:

`delta_X = A B^T`, `delta_Y = A(B^T W^T)`.

The weight image is computed for r shared temporal directions, not 80 independent vectors.

Controlled 80-lane, 2048x2048 benchmark:

| delta rank | speedup |
|---:|---:|
| 1 | 8.60× |
| 4 | 3.16× |
| 8 | 2.69× |
| 16 | 2.49× |

The ideal large-matrix ratio approaches lane_count/rank. M-Rail/speculative blocks and Beast's many parallel instances provide the natural batch axis.

## Compiled partition-function attention

Let a temporal query regime satisfy

`delta score_i = signature_i^T alpha`,

where signatures are exact discrete key projections in the active temporal directions.

Group equal signatures c:

`H_c = sum exp(base_score_i)`

`M_c = sum exp(base_score_i) value_i`.

Then

`Z(alpha) = sum_c exp(c^T alpha) H_c`

`N(alpha) = sum_c exp(c^T alpha) M_c`.

Attention cost depends on occupied signatures, not context length.

Controlled exact benchmark:

- context: 131,072;
- value width: 64;
- four discrete temporal signature dimensions;
- occupied groups: 1,520;
- dense: 4.962 ms;
- compiled: 0.0247 ms;
- 201.10×;
- max error: 1.11e-16.

This is a structure benchmark. `origami_temporal_census.py` determines real signature occupancy and support-pattern frequency before the route is enabled.

## Tensor-train partition attention

When active temporal width grows, the grouped partition tensor is

`H[c1,...,cr]`, `M[c1,...,cr,v]`.

A query is a tensor contraction with one vector per temporal dimension. Tensor-train compilation changes query cost from context length / full code-tuple count to TT rank:

`O(sum_j level_count_j * rank_(j-1) * rank_j)`.

This is the long-context path beyond flat signature grouping.

## Codec-native execution

`CallbackDeltaLinearOperator` is the zero-copy boundary for Beast:

- `full(x)`;
- `columns(U,delta)` directly from compressed pages;
- `basis_image(B)` directly from the Origami/HCGM/fx7 representation.

No dense weight materialization is part of the temporal runtime contract.

## Remath compiler path

Remath formulas are now temporalized:

| recognized family | temporal rewrite |
|---|---|
| GEMM/GEMV | temporal linear portfolio |
| RMSNorm | temporal RMSNorm multi-linear |
| Attention | compiled partition or dynamic normalized state |
| Softmax | partition state / lazy reconciliation |
| Top-k | certified router |
| activation | quantized temporal activation |
| reductions | streaming sufficient state |
| FFT | requested response basis |

Compiler:

`program -> Remath graph -> temporal delta census -> representation portfolio -> whole-graph plan -> codec-native CPU/GPU executor`.

## E235: actual Origami census

The shipped CLI accepts NPZ traces with arrays `[tokens,width]` and optional code arrays / bases:

```bash
python scripts/origami_temporal_census.py trace.npz \
  --basis origami_bases.npz \
  --block-size 32 \
  --out origami-temporal-census.json
```

It reports for every static linear-form input:

- exact coordinate and block support;
- activation-code Hamming support;
- top-99/top-99.9 energy support;
- temporal-basis concentration and exact residual;
- cross-token effective rank;
- recurring support patterns for partition compilation.

## System budget

Using only prototype-measured structural regimes:

- linear 14.26× (four-coordinate RMSNorm-linear regime),
- normalized reduction 201× (compiled four-dimensional signature regime),
- codec/I/O 10×,
- other work 3×,

projects 11.61 ms/token = 86.1 tok/s.

At the one-coordinate linear regime (34.49×), the same budget is 8.49 ms/token = 117.7 tok/s.

With 20× codec/I/O and 4× remaining-work reduction, the corresponding projections are 104–155 tok/s.

These are route-composition budgets, not Origami measurements. E235 replaces each assumed regime with observed layer distributions and emits the real end-to-end plan.

## IP claim clusters

1. **Temporal multi-representation transformer execution**: exact token-to-token updates routed among coordinate, block-code, basis, low-rank and full forms.
2. **Exact RMSNorm-to-multiple-linear state transition** without normalized-vector materialization.
3. **Compiled partition-function attention** over discrete temporal key signatures.
4. **Tensor-network partition attention** whose cost is governed by code-tensor rank rather than context length.
5. **Certified MoE routing** with interval-stable expert sets and codec residency.
6. **Cross-lane/speculative delta factorization** sharing one weight image across M-Rail paths or parallel instances.
7. **Codec-native delta operator ABI** that executes directly on Origami/HCGM/fx7 pages.
8. **Remath semantic auto-compilation** from program formulas into persistent temporal state machines.

## End state

Origami is no longer treated as a sequence of dense layer calls.

It becomes a compiled state-transition machine whose work is proportional to:

`temporal information change x requested output x surviving structure`.
