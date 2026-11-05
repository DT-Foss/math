from wirkung_math.state_quotient import AdditiveStatsFootprint

def test_32m_by_64_hot_state_collapse():
    x=AdditiveStatsFootprint(33_554_432,64,8)
    assert x.raw_bytes==17_179_869_184
    assert x.sufficient_bytes==33_792
    assert x.compression_ratio>500_000
