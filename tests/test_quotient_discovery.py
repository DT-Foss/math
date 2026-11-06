import numpy as np

from wirkung_math.consumer_ir import EvidenceLevel
from wirkung_math.quotient_discovery import discover_empirical_quotient


def test_empirical_quotient_measures_consumer_classes_and_transition_reuse():
    states=np.arange(16,dtype=float).reshape(8,2)
    signatures=[0,0,1,1,0,0,1,1]
    q=discover_empirical_quotient(states,signatures)
    assert q.samples==8
    assert q.state_dimension==2
    assert q.distinct_classes==2
    assert q.compression_ratio==4.0
    assert abs(q.signature_entropy_bits-1.0)<1e-12
    assert q.changed_transitions==3
    assert q.evidence==EvidenceLevel.VALIDATED
    assert 0.0 < q.transition_reuse_fraction < 1.0


def test_empirical_quotient_accepts_vector_signatures():
    states=np.eye(4)
    signatures=np.array([[1,0],[1,0],[0,1],[0,1]])
    q=discover_empirical_quotient(states,signatures)
    assert q.distinct_classes==2
