from __future__ import annotations
from dataclasses import dataclass,replace
from typing import Any
from .consumer_ir import ProgramNode,ConsumerProgram


@dataclass(frozen=True)
class RemathWirkungHint:
    canonical_name: str
    domain: str
    family: str
    rewrite: str
    persistent_state: tuple[str,...]
    preserves: tuple[str,...]
    destroys: tuple[str,...] = ()
    consumer_sensitive: bool = False
    temporal_rewrite: str = ""
    origami_state: tuple[str,...] = ()


def _get(obj: Any, name: str, default: Any="") -> Any:
    if isinstance(obj,dict):
        return obj.get(name,default)
    return getattr(obj,name,default)


def classify_remath_formula(formula_or_node: Any) -> RemathWirkungHint:
    """
    Dependency-light bridge from Remath FormulaSpec/GraphNode (or their dict form)
    into Wirkung rewrite families. Remath is not required at import time.
    """
    formula=_get(formula_or_node,"formula",formula_or_node)
    name=str(_get(formula,"canonical_name","")).strip()
    domain=str(_get(formula,"domain","")).strip().lower()
    low=name.lower()

    if "scaled dot-product attention" in low or low == "attention":
        return RemathWirkungHint(
            name,domain,"normalized_bilinear","dynamic_normalized_bilinear",
            ("output","log_partition","Q","K","V"),
            ("row_support","column_support","normalized_statistics"),
            consumer_sensitive=True,
        )
    if "softmax" in low or domain=="attention" and "normal" in low:
        return RemathWirkungHint(
            name,domain,"normalized_reduction","mass_gated_normalized_state",
            ("output","log_partition"),
            ("edited_term_support","normalized_statistics"),
        )
    if "rmsnorm" in low or "layernorm" in low or domain=="normalization":
        return RemathWirkungHint(
            name,domain,"normalization","structured_norm_rewrite",
            ("base_output","normalization_statistics"),
            ("row_scale","low_rank"),
        )
    if "matrix contraction" in low or "gemm" in low or "gemv" in low or domain=="linear_algebra":
        return RemathWirkungHint(
            name,domain,"linear_map","factor_propagation",
            ("base_output",),
            ("low_rank","sparse_rows","sparse_columns"),
            consumer_sensitive=True,
        )
    if "activation" in low or domain=="activation":
        return RemathWirkungHint(
            name,domain,"elementwise_nonlinearity","support_preserving_elementwise",
            (),
            ("sparse_rows","dirty_geometry"),
            ("low_rank",),
        )
    if any(k in low for k in ("sum","mean","moment","variance","gram","covariance")):
        return RemathWirkungHint(
            name,domain,"additive_reduction","sufficient_statistics",
            ("additive_summary",),
            ("additive_summary",),
        )
    if any(k in low for k in ("top-k","topk","argmax","argmin","max","min")):
        return RemathWirkungHint(
            name,domain,"order_reduction","certificate_or_reduction_tree",
            ("topk_or_tree",),
            ("order_certificate",),
        )
    if "fft" in low or "fourier" in low:
        return RemathWirkungHint(
            name,domain,"spectral_transform","response_basis_or_butterfly",
            ("spectrum","optional_stage_checkpoints"),
            ("response_basis","dirty_butterfly"),
        )
    return RemathWirkungHint(
        name,domain,"unknown","measure_then_route",(),(),
        consumer_sensitive=True,
    )


def temporalize_hint(hint:RemathWirkungHint)->RemathWirkungHint:
    """Attach the exact temporal Origami/Beast rewrite selected for a formula family."""
    mapping={
        "linear_map":(
            "temporal_linear_portfolio",
            ("previous_input","previous_output","activation_codes","basis_coefficients"),
        ),
        "normalization":(
            "temporal_rmsnorm_multilinear",
            ("previous_norm","previous_projection_outputs"),
        ),
        "normalized_bilinear":(
            "compiled_partition_attention_or_dynamic_normalized",
            ("base_query","quantized_key_signatures","partition_mass","partition_values"),
        ),
        "normalized_reduction":(
            "compiled_partition_attention_or_lazy_reconciliation",
            ("log_partition","numerator","signature_partitions"),
        ),
        "order_reduction":(
            "certified_topk_router",
            ("selected_experts","logit_intervals","certificate_margin"),
        ),
        "elementwise_nonlinearity":(
            "quantized_temporal_activation",
            ("previous_activation_codes","changed_code_support"),
        ),
        "additive_reduction":(
            "streaming_sufficient_statistics",
            ("sufficient_state",),
        ),
        "spectral_transform":(
            "requested_response_basis",
            ("requested_coefficients","sparse_input_delta"),
        ),
    }
    rewrite,state=mapping.get(hint.family,("measure_then_route",()))
    return replace(hint,temporal_rewrite=rewrite,origami_state=state)


def annotate_remath_trace(trace: Any) -> list[dict[str,Any]]:
    """Return compact Wirkung annotations for Remath execution-graph nodes."""
    nodes=_get(trace,"nodes",[])
    if isinstance(nodes,dict):
        nodes=list(nodes.values())
    out=[]
    for node in nodes:
        formula=_get(node,"formula",{})
        hint=temporalize_hint(classify_remath_formula(formula))
        out.append({
            "node_id":_get(node,"id",""),
            "node_name":_get(node,"name",""),
            "category":_get(node,"category",""),
            "canonical_name":hint.canonical_name,
            "wirkung_family":hint.family,
            "rewrite":hint.rewrite,
            "temporal_rewrite":hint.temporal_rewrite,
            "persistent_state":list(hint.persistent_state),
            "origami_state":list(hint.origami_state),
            "preserves":list(hint.preserves),
            "destroys":list(hint.destroys),
            "consumer_sensitive":hint.consumer_sensitive,
        })
    return out


def _dependency_ids(node:Any)->tuple[str,...]:
    raw=_get(node,"inputs",None)
    if raw is None:
        raw=_get(node,"dependencies",None)
    if raw is None:
        raw=_get(node,"parents",())
    if isinstance(raw,(str,bytes)):
        raw=(raw,)
    out=[]
    for value in raw or ():
        if isinstance(value,dict):
            dep=_get(value,"id",_get(value,"name",""))
        else:
            dep=_get(value,"id",value)
        dep=str(dep)
        if dep:
            out.append(dep)
    return tuple(out)


def consumer_program_from_remath_trace(
    trace:Any,*,changed:set[str]|frozenset[str],
    requested:set[str]|frozenset[str],
)->ConsumerProgram:
    """
    Lower a dependency-bearing Remath trace into the Quotient Compiler IR.

    The bridge preserves operator payload metadata such as matrices/shapes while
    using Remath's semantic classification to choose the compiler family.
    """
    raw_nodes=_get(trace,"nodes",[])
    if isinstance(raw_nodes,dict):
        raw_nodes=list(raw_nodes.values())
    nodes=[]
    family_map={
        "linear_map":"linear",
        "additive_reduction":"additive_reduction",
        "normalized_reduction":"normalized_reduction",
        "normalized_bilinear":"normalized_bilinear",
        "normalization":"normalization",
        "elementwise_nonlinearity":"elementwise",
        "order_reduction":"order_reduction",
        "spectral_transform":"spectral",
    }
    for raw in raw_nodes:
        node_id=str(_get(raw,"id",_get(raw,"name","")))
        if not node_id:
            raise ValueError("Remath node is missing id/name")
        category=str(_get(raw,"category","")).lower()
        inputs=_dependency_ids(raw)
        formula=_get(raw,"formula",{})
        hint=classify_remath_formula(formula)
        is_source=(category=="source" or (not inputs and not _get(raw,"formula",None)))
        family="source" if is_source else family_map.get(hint.family,hint.family)
        metadata={}
        shape=_get(raw,"shape",None)
        if shape is not None:
            metadata["shape"]=shape
            if family=="source":
                try:
                    metadata["dimension"]=int(shape[-1])
                except (TypeError,IndexError):
                    pass
        dimension=_get(raw,"dimension",None)
        if dimension is not None:
            metadata["dimension"]=int(dimension)
        matrix=_get(raw,"matrix",None)
        if matrix is not None:
            metadata["matrix"]=matrix
        metadata.update({
            "remath_name":str(_get(raw,"name","")),
            "remath_category":category,
            "canonical_name":hint.canonical_name,
            "wirkung_family":hint.family,
            "rewrite":hint.rewrite,
            "temporal_rewrite":temporalize_hint(hint).temporal_rewrite,
        })
        nodes.append(ProgramNode(node_id,family,inputs,metadata))
    return ConsumerProgram(
        tuple(nodes),frozenset(changed),frozenset(requested)
    )
