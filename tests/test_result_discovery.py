from wirkung_math.consumer_ir import EvidenceLevel
from wirkung_math.result_discovery import ingest_result_receipt


def test_ingest_event_receipt_extracts_validated_cost_win():
    receipt={
        "event_runtime":{
            "runs":[{
                "bits":2,
                "temporal_speedup":1.25,
                "temporal_tokens_equal":True,
                "changed_block_fraction_median":.2,
            }]
        }
    }
    routes=ingest_result_receipt(receipt)
    assert len(routes)==1
    r=routes[0]
    assert r.evidence==EvidenceLevel.VALIDATED
    assert r.metadata["semantic_success"] is True
    assert r.metadata["cost_win"] is True
    assert r.metadata["speedup"]==1.25


def test_ingest_quality_preserving_sigma_delta_can_still_reject_on_cost():
    receipt={
        "test":{
            "fp_equal":True,
            "temporal_equal_full_sigma":True,
            "temporal_speedup_vs_fp32":.92,
            "event_fraction_median":.906,
        }
    }
    r=ingest_result_receipt(receipt)[0]
    assert r.metadata["semantic_success"] is True
    assert r.metadata["cost_win"] is False
    assert r.update_ms > 1.0


def test_ingest_failed_semantics_marks_heuristic_not_validated():
    receipt={"run":{"speedup":4.0,"tokens_equal":False}}
    r=ingest_result_receipt(receipt)[0]
    assert r.evidence==EvidenceLevel.HEURISTIC
    assert r.metadata["semantic_success"] is False


def test_ingest_certified_head_preserves_candidate_metadata():
    receipt={
        "key_results":[{
            "tokens_equal":True,
            "speedup":.63,
            "candidate_median":32,
            "candidate_p90":100,
        }]
    }
    r=ingest_result_receipt(receipt)[0]
    assert r.metadata["candidate_median"]==32
    assert r.metadata["cost_win"] is False
