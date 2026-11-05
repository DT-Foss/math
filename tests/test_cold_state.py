from wirkung_math.cold_state import ColdRowFetchModel,ColdBackedDeltaPlan

def test_32m_stats_can_be_cold_backed():
    p=ColdBackedDeltaPlan(ColdRowFetchModel(64,64,8),0.05300954217091203,61.09900665283203)
    assert 0.03 < p.fetch.transfer_ms(1.0) < 0.04
    assert p.speedup(1.0) > 700
    assert p.speedup(0.1) > 150
