"""Wirkung Math: structured delta algebra for stateful computation."""
from .lowrank import FactorDelta, crossover_rank, effective_edits, sparse_lowrank_update
from .normalized import NormalizedState, full_normalized, mass_gated_update, topk_miss_probability
from .norms import rmsnorm_lowrank_update, layernorm_lowrank_update
from .router import route_linear_delta
from .dirty import (
    GeometryCostModel,
    Interval,
    connected_components,
    dirty_cone_components,
    schedule_components_optimal,
    update_global_sum,
    update_projected_global_sum,
)

__all__ = [
    "FactorDelta", "crossover_rank", "effective_edits", "sparse_lowrank_update",
    "NormalizedState", "full_normalized", "mass_gated_update", "topk_miss_probability",
    "rmsnorm_lowrank_update", "layernorm_lowrank_update", "route_linear_delta",
    "GeometryCostModel", "Interval", "connected_components", "dirty_cone_components",
    "schedule_components_optimal", "update_global_sum", "update_projected_global_sum",
]
