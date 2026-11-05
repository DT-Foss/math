# Experiment map

| ID | Question | Executable / provenance |
|---|---|---|
| E172 | How far can normalized state updates go when routing on probability mass/cancellation rather than maximum identity? | `experiments/e172_mass_gated_normalized.py` |
| E173 | Does the closed sparse×low-rank crossover `r*=ed/(e+d)` predict measured executor crossover? | `experiments/e173_sparse_lowrank_phase.py` |
| E174 | What is gained by propagating factors through linear layers instead of materializing each delta? | `experiments/e174_factor_chain.py` |
| E175 | Do RMSNorm and LayerNorm admit exact structured low-rank rewrites? | `experiments/e175_normalization_chain.py` |
| E176 | How violently does SiLU expand numerical rank as delta amplitude grows? | `experiments/e176_nonlinear_barrier.py` |
| E177 | Can accumulated residual factors be rank-revealed and compacted without dense materialization? | `experiments/e177_factor_compaction.py` |
| E185 | Does preserving disconnected dirty cones beat one bounding rectangle through a deep nonlinear local graph? | imported CPU-box record: `results/imported_e185_e187.json` |
| E186 | When should dirty components remain tiled versus be merged? | `experiments/e186_dirty_component_scheduler.py` + imported CPU-box calibration |
| E187 | Can a global reduction / narrow projection consume only the dirty terminal regions? | `experiments/e187_dirty_to_global_reduction.py` |

Every executable experiment prints one JSON record so outputs can be stored directly without prose parsing.

## Compiler path

E185–E187 are represented in runtime code, not only as benchmark scripts:

`dirty support -> connected components -> receptive-field cones -> optimal tile partition -> local execution -> delta global reduction`

The first four scheduling stages are implemented by `DirtyGraphCompiler`.
