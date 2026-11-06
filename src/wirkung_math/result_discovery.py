from __future__ import annotations
from collections.abc import Mapping,Sequence
from .consumer_ir import CandidateRoute,EvidenceLevel


_SPEEDUP_KEYS=(
    "temporal_speedup_vs_fp32",
    "temporal_speedup_vs_full_sigma",
    "temporal_speedup",
    "speedup_vs_full",
    "speedup",
    "cpu_over_v100",
    "delta_vs_native_full",
    "temporal_vs_full_quant",
)
_SEMANTIC_KEYS=(
    "tokens_equal",
    "temporal_tokens_equal",
    "temporal_equal_full_sigma",
    "temporal_tokens_equal_full",
    "int_accumulator_exact",
    "all_tokens",
    "fp_equal",
    "valid",
)
_BYTES_KEYS=("state_bytes","table_bytes","hot_bytes")


def _scalar_metadata(node:Mapping)->dict:
    out={}
    for key,value in node.items():
        if isinstance(value,(str,int,float,bool)) or value is None:
            out[str(key)]=value
    return out


def _candidate_from_mapping(path:str,node:Mapping)->CandidateRoute|None:
    speedup_key=next(
        (key for key in _SPEEDUP_KEYS if key in node),None
    )
    semantic_keys=[key for key in _SEMANTIC_KEYS if key in node]
    if speedup_key is None or not semantic_keys:
        return None
    try:
        speedup=float(node[speedup_key])
    except (TypeError,ValueError):
        return None
    if speedup<=0:
        return None
    semantic_success=all(bool(node[key]) for key in semantic_keys)
    evidence=(
        EvidenceLevel.VALIDATED if semantic_success
        else EvidenceLevel.HEURISTIC
    )
    state_bytes=0
    for key in _BYTES_KEYS:
        if key in node:
            try:
                state_bytes=max(0,int(node[key]))
            except (TypeError,ValueError):
                state_bytes=0
            break
    metadata=_scalar_metadata(node)
    metadata.update({
        "receipt_path":path or "$",
        "speedup_key":speedup_key,
        "speedup":speedup,
        "semantic_keys":tuple(semantic_keys),
        "semantic_success":semantic_success,
        "cost_win":speedup>1.0,
        "cost_source":"measured_ratio",
    })
    return CandidateRoute(
        name=f"receipt:{path or '$'}",
        source="receipt:start",
        target="receipt:goal",
        setup_ms=0.0,
        update_ms=1.0/speedup,
        query_ms=0.0,
        state_bytes=state_bytes,
        evidence=evidence,
        device=str(node.get("device","unknown")),
        metadata=metadata,
    )


def ingest_result_receipt(receipt:dict)->tuple[CandidateRoute,...]:
    """
    Extract measured semantic/cost opportunities from arbitrary nested JSON.

    This parser deliberately never upgrades receipt evidence to PROVEN.
    Algebraic proof belongs to compiler rules/certificates, not measurement.
    """
    if not isinstance(receipt,dict):
        raise ValueError("receipt must be a mapping")
    found=[]

    def walk(value,path:str)->None:
        if isinstance(value,Mapping):
            candidate=_candidate_from_mapping(path,value)
            if candidate is not None:
                found.append(candidate)
            for key,child in value.items():
                child_path=f"{path}/{key}" if path else str(key)
                walk(child,child_path)
        elif isinstance(value,Sequence) and not isinstance(
            value,(str,bytes,bytearray)
        ):
            for i,child in enumerate(value):
                child_path=f"{path}[{i}]" if path else f"[{i}]"
                walk(child,child_path)

    walk(receipt,"")
    return tuple(found)
