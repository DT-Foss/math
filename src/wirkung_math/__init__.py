"""Wirkung Math: structured delta algebra for stateful computation."""
from .lowrank import FactorDelta, crossover_rank, effective_edits, sparse_lowrank_update
from .normalized import NormalizedState, full_normalized, mass_gated_update, topk_miss_probability
from .norms import rmsnorm_lowrank_update, layernorm_lowrank_update
from .router import route_linear_delta
from .dirty import (
    GeometryCostModel, Interval, connected_components, dirty_cone_components,
    schedule_components_optimal, update_global_sum, update_projected_global_sum,
)
from .compiler import DirtyGraphCompiler, DirtyPlan
from .statistics import SufficientStatisticsState, NormalEquationsState
from .symmetric import SymmetricLowRankDelta, WoodburyCache
from .kernel import RowColumnDelta, RowColumnWoodbury, rbf_kernel, rbf_replacement_delta
from .spectral import SparseDFTState, ButterflyFFTState
from .cholesky import CholeskyState
from .block import BlockAppendCache, delete_trailing_block_from_inverse
from .delta_ir import DeltaDescriptor, Structure
from .bilinear import DynamicNormalizedBilinearState, full_normalized_bilinear

__all__ = [
    "FactorDelta","crossover_rank","effective_edits","sparse_lowrank_update",
    "NormalizedState","full_normalized","mass_gated_update","topk_miss_probability",
    "rmsnorm_lowrank_update","layernorm_lowrank_update","route_linear_delta",
    "GeometryCostModel","Interval","connected_components","dirty_cone_components",
    "schedule_components_optimal","update_global_sum","update_projected_global_sum",
    "DirtyGraphCompiler","DirtyPlan",
    "SufficientStatisticsState","NormalEquationsState",
    "SymmetricLowRankDelta","WoodburyCache",
    "RowColumnDelta","RowColumnWoodbury","rbf_kernel","rbf_replacement_delta",
    "SparseDFTState","ButterflyFFTState","CholeskyState",
    "BlockAppendCache","delete_trailing_block_from_inverse",
    "DeltaDescriptor","Structure",
    "DynamicNormalizedBilinearState","full_normalized_bilinear",
]
