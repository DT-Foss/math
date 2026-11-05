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
from .kernel import RowColumnDelta, RowColumnWoodbury, RowColumnSolverWoodbury, rbf_kernel, rbf_replacement_delta
from .spectral import SparseDFTState, ButterflyFFTState
from .cholesky import CholeskyState
from .block import BlockAppendCache, delete_trailing_block_from_inverse
from .delta_ir import DeltaDescriptor, Structure
from .bilinear import DynamicNormalizedBilinearState, full_normalized_bilinear
from .graph import laplacian_edge_delta, laplacian_edges_delta
from .reduction import SegmentReductionTree
from .coalesce import RowReplacementLog
from .woodbury import GeneralLowRankUpdate, GeneralWoodburyCache, SolverWoodburyCache
from .remath_bridge import RemathWirkungHint, classify_remath_formula, annotate_remath_trace
from .dirty2d import AreaCostModel, DirtyGraphCompiler2D, DirtyPlan2D, Rect, schedule_rectangles_optimal
from .update_ledger import LowRankUpdateLedger, IncrementalWoodburyLedger
from .cacheless_local import LocalGraphDependency, LocalResidencyChoice, LocalResidencyPlanner
from .lazy_normalized import LazyNormalizedKVState
from .lifetime import ExecutionRegime, HorizonPlanner
from .demand_linear import sparse_requested_linear_delta
from .plan_graph import PlanTransition, PlanResult, RepresentationPlanGraph
from .lazy_gram import LazyGramState
from .replacement import replacement_statistics_delta, replacement_gram_delta, replacement_cross_delta
from .demand_slice import DemandSlice, delta_demand_slice
from .state_quotient import AdditiveStatsFootprint, StateQuotient
from .native_rank1 import NativeRank1StatsState, build_rank1_stats_kernel
from .hardware_router import (
    ConsumerAwareKernelRouter, ConsumerRequirement, DirtyV100Calibration,
    KernelMaterializationCalibration, KernelSolveCalibration,
    SparseSpectrumV100Calibration, NormalizedQueryV100Calibration,
)

__all__ = [
    "FactorDelta","crossover_rank","effective_edits","sparse_lowrank_update",
    "NormalizedState","full_normalized","mass_gated_update","topk_miss_probability",
    "rmsnorm_lowrank_update","layernorm_lowrank_update","route_linear_delta",
    "GeometryCostModel","Interval","connected_components","dirty_cone_components",
    "schedule_components_optimal","update_global_sum","update_projected_global_sum",
    "DirtyGraphCompiler","DirtyPlan",
    "SufficientStatisticsState","NormalEquationsState",
    "SymmetricLowRankDelta","WoodburyCache",
    "RowColumnDelta","RowColumnWoodbury","RowColumnSolverWoodbury","rbf_kernel","rbf_replacement_delta",
    "SparseDFTState","ButterflyFFTState","CholeskyState",
    "BlockAppendCache","delete_trailing_block_from_inverse",
    "DeltaDescriptor","Structure",
    "DynamicNormalizedBilinearState","full_normalized_bilinear",
    "laplacian_edge_delta","laplacian_edges_delta",
    "SegmentReductionTree","RowReplacementLog",
    "GeneralLowRankUpdate","GeneralWoodburyCache","SolverWoodburyCache",
    "RemathWirkungHint","classify_remath_formula","annotate_remath_trace",
    "AreaCostModel","DirtyGraphCompiler2D","DirtyPlan2D","Rect","schedule_rectangles_optimal",
    "LowRankUpdateLedger","IncrementalWoodburyLedger",
    "LocalGraphDependency","LocalResidencyChoice","LocalResidencyPlanner",
    "LazyNormalizedKVState","ExecutionRegime","HorizonPlanner","sparse_requested_linear_delta",
    "PlanTransition","PlanResult","RepresentationPlanGraph","LazyGramState",
    "replacement_statistics_delta","replacement_gram_delta","replacement_cross_delta",
    "DemandSlice","delta_demand_slice","AdditiveStatsFootprint","StateQuotient",
    "NativeRank1StatsState","build_rank1_stats_kernel",
    "ConsumerAwareKernelRouter","ConsumerRequirement","DirtyV100Calibration",
    "KernelMaterializationCalibration","KernelSolveCalibration",
    "SparseSpectrumV100Calibration","NormalizedQueryV100Calibration",
]
