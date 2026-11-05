# Experiment map

| ID | Question | Executable |
|---|---|---|
| E172 | How far can normalized state updates go when routing on probability mass/cancellation rather than maximum identity? | `experiments/e172_mass_gated_normalized.py` |
| E173 | Does the closed sparse×low-rank crossover `r*=ed/(e+d)` predict measured executor crossover? | `experiments/e173_sparse_lowrank_phase.py` |
| E174 | What is gained by propagating factors through linear layers instead of materializing each delta? | `experiments/e174_factor_chain.py` |
| E175 | Do RMSNorm and LayerNorm admit exact structured low-rank rewrites? | `experiments/e175_normalization_chain.py` |
| E176 | How violently does SiLU expand numerical rank as delta amplitude grows? | `experiments/e176_nonlinear_barrier.py` |
| E177 | Can accumulated residual factors be rank-revealed and compacted without dense materialization? | `experiments/e177_factor_compaction.py` |

Every experiment prints one JSON record so outputs can be stored directly without prose parsing.
