from wirkung_math.hardware_router import SparseSpectrumV100Calibration,NormalizedQueryV100Calibration

def test_output_routes():
    a=SparseSpectrumV100Calibration()
    b=NormalizedQueryV100Calibration()
    assert a.route(16384)=="cpu_sparse_spectrum"
    assert a.route(65536)=="v100_full_fft"
    assert b.route(128)=="cpu_cached_queries"
    assert b.route(256)=="v100_requested_queries"
