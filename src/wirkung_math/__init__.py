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
from .remath_bridge import RemathWirkungHint, classify_remath_formula, temporalize_hint, annotate_remath_trace
from .dirty2d import AreaCostModel, DirtyGraphCompiler2D, DirtyPlan2D, Rect, schedule_rectangles_optimal
from .update_ledger import LowRankUpdateLedger, IncrementalWoodburyLedger
from .cacheless_local import LocalGraphDependency, LocalResidencyChoice, LocalResidencyPlanner
from .lazy_normalized import LazyNormalizedKVState
from .lifetime import ExecutionRegime, HorizonPlanner
from .demand_linear import sparse_requested_linear_delta
from .plan_graph import PlanTransition, PlanResult, RepresentationPlanGraph
from .lazy_gram import LazyGramState
from .replacement import replacement_statistics_delta, replacement_gram_delta, replacement_cross_delta, lowrank_replacement_statistics, lowrank_delta_statistics_from_base
from .demand_slice import DemandSlice, delta_demand_slice
from .state_quotient import AdditiveStatsFootprint, StateQuotient
from .native_rank1 import NativeRank1StatsState, build_rank1_stats_kernel
from .cold_state import ColdRowFetchModel, ColdBackedDeltaPlan
from .adaptive_stats import AdaptiveStatsState
from .covariance import covariance_replacement_delta
from .local_demand import LayerDemandRegion, local_chain_delta_demand, required_area_sum
from .hardware_router import (
    ConsumerAwareKernelRouter, ConsumerRequirement, DirtyV100Calibration,
    KernelMaterializationCalibration, KernelSolveCalibration,
    SparseSpectrumV100Calibration, NormalizedQueryV100Calibration,
)
from .temporal_linear import (
    sparse_linear_delta, TemporalLinearState, BasisDeltaLinearState,
    BatchLowRankLinearState, TemporalRMSNormLinearState,
)
from .temporal_multilinear import TemporalRMSNormMultiLinearState
from .temporal_quantized import QuantizedVectorState, QuantizedTemporalLinearState
from .temporal_activation import QuantizedSwiGLUDownState, silu, nearest_codes
from .temporal_basis import TemporalDeltaBasis, EncodedTemporalDelta, exact_batch_factorization
from .delta_layout import (
    ColumnPackedLinearOperator, ColumnPackedTemporalLinearState,
    ColumnPackedRMSNormLinearState, ColumnPackedRMSNormMultiLinearState,
)
from .router_certificate import CertifiedTopKRouterState
from .partition_attention import CompiledPartitionAttention, PartitionAttentionBank
from .tensor_train import TensorTrain
from .tensor_partition_attention import TensorTrainPartitionAttention
from .delta_operator import (
    DeltaLinearOperator, DenseDeltaLinearOperator, CallbackDeltaLinearOperator,
    TemporalOperatorState, BasisOperatorState,
)
from .temporal_census import (
    TemporalDeltaRecord, TemporalDeltaCensus, measure_temporal_delta,
    support_for_energy, effective_rank,
)
from .event_response import IntegerResponseLinearState
from .signed_digit_response import SignedDigitResponseLinearState, balanced_ternary_scales, balanced_ternary_encode
from .sigma_delta_response import SigmaDeltaResponseLinearState
from .consumer_ir import (
    EvidenceLevel, ProgramNode, CandidateRoute, CompilationPolicy, ConsumerProgram,
)
from .linear_quotient import LinearConsumerQuotient, ObservableLinearQuotient
from .linear_graph_quotient import LinearNodeQuotient, discover_linear_graph_quotients
from .quotient_discovery import EmpiricalQuotient, discover_empirical_quotient
from .certificates import (
    LinearOrderCertificate, QuantizationCellCertificate, ThresholdCertificate,
)
from .quotient_compiler import CompilationResult, QuotientCompiler
from .result_discovery import ingest_result_receipt
from .temporal_delta_machine import (
    LinearDeltaRoute, DeltaPortfolio, LinearRouteCost, LinearDeltaPlanner,
    BeastDecodeBudget, ProjectedDecodeBudget, OrigamiDeltaMachineTarget,
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
    "RemathWirkungHint","classify_remath_formula","temporalize_hint","annotate_remath_trace",
    "AreaCostModel","DirtyGraphCompiler2D","DirtyPlan2D","Rect","schedule_rectangles_optimal",
    "LowRankUpdateLedger","IncrementalWoodburyLedger",
    "LocalGraphDependency","LocalResidencyChoice","LocalResidencyPlanner",
    "LazyNormalizedKVState","ExecutionRegime","HorizonPlanner","sparse_requested_linear_delta",
    "PlanTransition","PlanResult","RepresentationPlanGraph","LazyGramState",
    "replacement_statistics_delta","replacement_gram_delta","replacement_cross_delta","lowrank_replacement_statistics","lowrank_delta_statistics_from_base",
    "DemandSlice","delta_demand_slice","AdditiveStatsFootprint","StateQuotient",
    "NativeRank1StatsState","build_rank1_stats_kernel","ColdRowFetchModel","ColdBackedDeltaPlan","AdaptiveStatsState","covariance_replacement_delta",
    "LayerDemandRegion","local_chain_delta_demand","required_area_sum",
    "ConsumerAwareKernelRouter","ConsumerRequirement","DirtyV100Calibration",
    "KernelMaterializationCalibration","KernelSolveCalibration",
    "SparseSpectrumV100Calibration","NormalizedQueryV100Calibration",
    "sparse_linear_delta","TemporalLinearState","BasisDeltaLinearState",
    "BatchLowRankLinearState","TemporalRMSNormLinearState",
    "TemporalRMSNormMultiLinearState","QuantizedVectorState","QuantizedTemporalLinearState",
    "QuantizedSwiGLUDownState","silu","nearest_codes",
    "TemporalDeltaBasis","EncodedTemporalDelta","exact_batch_factorization",
    "ColumnPackedLinearOperator","ColumnPackedTemporalLinearState",
    "ColumnPackedRMSNormLinearState","ColumnPackedRMSNormMultiLinearState",
    "CertifiedTopKRouterState","CompiledPartitionAttention","PartitionAttentionBank",
    "TensorTrain","TensorTrainPartitionAttention",
    "DeltaLinearOperator","DenseDeltaLinearOperator","CallbackDeltaLinearOperator",
    "TemporalOperatorState","BasisOperatorState",
    "TemporalDeltaRecord","TemporalDeltaCensus","measure_temporal_delta",
    "support_for_energy","effective_rank",
    "IntegerResponseLinearState","SignedDigitResponseLinearState","balanced_ternary_scales","balanced_ternary_encode","SigmaDeltaResponseLinearState",
    "EvidenceLevel","ProgramNode","CandidateRoute","CompilationPolicy","ConsumerProgram",
    "LinearConsumerQuotient","ObservableLinearQuotient",
    "LinearNodeQuotient","discover_linear_graph_quotients",
    "EmpiricalQuotient","discover_empirical_quotient",
    "LinearOrderCertificate","QuantizationCellCertificate","ThresholdCertificate",
    "CompilationResult","QuotientCompiler","ingest_result_receipt",
    "LinearDeltaRoute","DeltaPortfolio","LinearRouteCost","LinearDeltaPlanner",
    "BeastDecodeBudget","ProjectedDecodeBudget","OrigamiDeltaMachineTarget",
]
